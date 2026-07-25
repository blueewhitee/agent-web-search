"""Stage 5: BGE embedding model loader (D-018).

Pure module-level function, lazy-load + cache. Uses fastembed (ONNX
Runtime, ~50 MB) instead of sentence-transformers+PyTorch (~526 MB CUDA
wheel). fastembed's DefaultEmbedding IS BAAI/bge-small-en-v1.5.

Query prefix per BAAI model card v1.5:
    "Represent this sentence for searching relevant passages: "
    - Applies ONLY to queries, NOT to passages/chunks.
    - v1.5 improved without it, but prefix adds slight retrieval quality.
    - Docs: https://huggingface.co/BAAI/bge-small-en-v1.5

Returns None on import/download failure → caller sets low_confidence.
"""

from __future__ import annotations

# ── Query instruction prefix ──────────────────────────────────

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Lazy model cache ──────────────────────────────────────────

_model = None
_FALLBACK = False


def _load_model() -> None:
    """Lazy-load bge-small-en-v1.5 via fastembed, cache at module level."""
    global _model, _FALLBACK
    if _model is not None or _FALLBACK:
        return
    try:
        from fastembed import TextEmbedding

        # DefaultEmbedding == BAAI/bge-small-en-v1.5 (fastembed's default).
        _model = TextEmbedding()
    except (ImportError, OSError):
        _FALLBACK = True


def is_available() -> bool:
    """True if the embedding model loaded successfully."""
    _load_model()
    return not _FALLBACK and _model is not None


def embed_texts(texts: list[str]) -> "np.ndarray | None":
    """Batch encode texts to 384-dim L2-normalized vectors.

    No query prefix applied — use this for chunks/passages.
    Returns None if the model isn't available.
    """
    _load_model()
    if _FALLBACK or _model is None:
        return None
    import numpy as np

    if not texts:
        return np.zeros((0, 384), dtype=np.float32)
    # fastembed yields embeddings one at a time; collect into a matrix.
    return np.array(list(_model.embed(texts)), dtype=np.float32)


def embed_query(query: str) -> "np.ndarray | None":
    """Encode a query string with retrieval instruction prefix.

    Returns (384,) single vector, L2-normalized.
    Returns None if the model isn't available.
    """
    _load_model()
    if _FALLBACK or _model is None:
        return None
    import numpy as np

    vecs = list(_model.query_embed(_QUERY_PREFIX + query))
    return vecs[0]