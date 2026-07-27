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
    absolute_floor: float = 0.2,
    gap_threshold: float = 0.15,
) -> list[dict] | None:
    """Embed query + chunks, return top-K sorted by cosine similarity.

    Two-stage relevance filter runs BEFORE diversity selection (D-024):
      1. Absolute floor: if the top score < ``absolute_floor`` there is no
         signal at all (SearXNG returned all-garbage) -> return []. This is a
         sanity backstop, NOT precision work — BGE scores are cross-query
         stable only at the "noise" end (~0.2), so the floor is calibrated
         loosely from known-garbage pages.
      2. Relative gap: drop any chunk whose score is more than ``gap_threshold``
         below the top hit. This is the precision work — BGE's absolute scores
         aren't calibrated, but the gap from the best hit IS query-adaptive.

    Order: score all -> floor check -> gap filter -> D-020 diversity -> top_k.

    Args:
        query:          Raw user query (prefix applied internally).
        chunks:         List of chunk dicts with keys:
                        {"text", "parent_text", "chunk_index", "source" (optional)}.
        top_k:          Number of top chunks to return.
        absolute_floor: Backstop max-score cutoff (return [] below this).
        gap_threshold:  Drop chunks more than this below the top hit.

    Returns:
        List of chunk dicts enriched with a "score" key, sorted descending.
        Returns [] if the top score is below the floor (no signal) or no
        chunks survive the gap filter.
        Returns None if the embedding model is unavailable (caller degrades).
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

    # Sort once, descending. all_indices[0] is the best chunk.
    all_indices = np.argsort(scores)[::-1]
    max_score = float(scores[all_indices[0]])

    # ── D-024 filter stage 1: absolute floor (sanity backstop) ──────────
    # If even the best chunk is below the floor, SearXNG returned all-garbage.
    # Admit defeat (empty list) rather than serve noise to the LLM. The caller
    # (agent's LLM, per D-009) decides the user-facing "couldn't find anything"
    # message — the API just returns no ranked chunks.
    if max_score < absolute_floor:
        return []

    # ── D-024 filter stage 2: relative gap (precision work) ─────────────
    # Drop chunks more than `gap_threshold` below the top hit. BGE absolute
    # scores aren't cross-query calibrated, but the gap from the best hit is
    # query-adaptive. Keep at least the top-1 (it cleared the floor).
    candidates = [
        int(i) for i in all_indices
        if (max_score - float(scores[i])) <= gap_threshold
    ]
    # Defensive: floor passed so top-1 is always in candidates, but guard
    # against FP edge cases leaving the list empty.
    if not candidates:
        candidates = [int(all_indices[0])]

    k = min(top_k, len(candidates))

    # Source diversity (D-020): one chunk per unique source URL in top-K.
    # Two passes: first collect one chunk per URL, then fill remaining
    # slots from any source if not enough unique URLs exist. Operates on the
    # gap-filtered `candidates` (preserves score order) so diversity never
    # resurrects a chunk the gap filter dropped.
    seen_urls: set[str] = set()
    diversified: list[int] = []

    for idx in candidates:
        url = chunks[idx].get("source", {}).get("url", "")
        if url and url in seen_urls:
            continue
        if url:
            seen_urls.add(url)
        diversified.append(idx)
        if len(diversified) >= k:
            break

    # Fallback: if not enough unique source URLs, fill remaining slots
    # with the next-best gap-filtered chunks regardless of source.
    if len(diversified) < k:
        for idx in candidates:
            if idx not in diversified:
                diversified.append(idx)
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
