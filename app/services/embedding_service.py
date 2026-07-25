"""Stage 5: BGE embedding model loader (D-018).

Pure module-level function, lazy-load + cache (same pattern as
_tokenizer in chunking_service.py). The model is 24M params,
384-dim, retrieval-trained.

Query prefix per BAAI model card v1.5:
    "Represent this sentence for searching relevant passages: "
    - Applies ONLY to queries, NOT to passages/chunks.
    - v1.5 improved without it, but prefix adds slight retrieval quality.
    - Docs: https://huggingface.co/BAAI/bge-small-en-v1.5

Returns None on import/download failure → caller sets low_confidence.
"""

from __future__ import annotations

import warnings

# ── Query instruction prefix ──────────────────────────────────

_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# ── Lazy model cache ──────────────────────────────────────────

_model = None
_FALLBACK = False


def _load_model() -> None:
    """Lazy-load bge-small-en-v1.5, cache at module level. ~100MB, ~500ms first call."""
    global _model, _FALLBACK
    if _model is not None or _FALLBACK:
        return
    try:
        from sentence_transformers import SentenceTransformer

        # Suppress the "`clean_up_tokenization_spaces` was ignored" warning
        # triggered by the underlying transformers tokenizer. Harmless noise.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            _model = SentenceTransformer("BAAI/bge-small-en-v1.5")
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
    return _model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def embed_query(query: str) -> "np.ndarray | None":
    """Encode a query string with retrieval instruction prefix.

    Returns (384,) single vector, L2-normalized.
    Returns None if the model isn't available.
    """
    _load_model()
    if _FALLBACK or _model is None:
        return None
    vec = _model.encode(
        [_QUERY_PREFIX + query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vec[0]  # (384,)
