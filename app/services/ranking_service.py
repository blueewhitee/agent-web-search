"""Stage 5: Cosine-similarity ranking over embedded chunks (D-003, D-008).

Pure function — takes a query, a list of chunk dicts, returns top-K
ranked by cosine similarity with scores.

Chunks are pooled across ALL results (5 pages → ~12 chunks → return
top 3). This is the D-008 differentiator: not "top results then
chunks within" but "top chunk from anywhere wins."

Returns None if the embedding model isn't available (caller sets
low_confidence=True on affected results, same pattern as D-016
extraction degradation).
"""

from __future__ import annotations

from app.services.embedding_service import embed_query, embed_texts, is_available


def rank_chunks(
    query: str,
    chunks: list[dict],
    top_k: int = 3,
) -> list[dict] | None:
    """Embed query + chunks, return top-K sorted by cosine similarity.

    Args:
        query:  Raw user query (prefix applied internally).
        chunks: List of chunk dicts with keys:
                {"text", "parent_text", "chunk_index", "source" (optional)}.
        top_k:  Number of top chunks to return.

    Returns:
        List of chunk dicts enriched with a "score" key, sorted descending.
        Returns None if embedding model unavailable.
    """
    import numpy as np

    if not chunks:
        return []

    if not is_available():
        return None

    chunk_texts = [c["text"] for c in chunks]

    chunk_vecs = embed_texts(chunk_texts)   # (n, 384)
    query_vec = embed_query(query)          # (384,)

    if chunk_vecs is None or query_vec is None:
        return None

    # Cosine similarity = dot product (both already L2-normalized).
    scores = chunk_vecs @ query_vec  # (n,)

    k = min(top_k, len(chunks))

    # Source diversity (D-020): one chunk per unique source URL in top-K.
    # Two passes: first collect one chunk per URL, then fill remaining
    # slots from any source if not enough unique URLs exist.
    all_indices = np.argsort(scores)[::-1]
    seen_urls: set[str] = set()
    diversified: list[int] = []

    for idx in all_indices:
        url = chunks[idx].get("source", {}).get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        diversified.append(int(idx))
        if len(diversified) >= k:
            break

    # Fallback: if not enough unique source URLs, fill remaining slots
    # with the next-best chunks regardless of source.
    if len(diversified) < k:
        for idx in all_indices:
            if idx not in diversified:
                diversified.append(int(idx))
                if len(diversified) >= k:
                    break

    ranked: list[dict] = []
    for idx in diversified:
        c = chunks[idx]
        ranked.append({
            "text": c["text"],
            "parent_text": c["parent_text"],
            "chunk_index": c["chunk_index"],
            "score": float(scores[idx]),
        })
        # Carry source metadata if present.
        if "source" in c:
            ranked[-1]["source"] = c["source"]

    return ranked
