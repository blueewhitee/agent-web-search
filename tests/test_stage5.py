"""Tests for Stage 5 embedding + ranking service.

Tests require the bge-small-en-v1.5 model to be downloaded (first
run may take ~30-60s). All tests skip if the model isn't available
to avoid hanging in CI/offline environments.
"""

import pytest

from app.services.embedding_service import embed_query, embed_texts, is_available
from app.services.ranking_service import rank_chunks

# ── Skip helpers ──────────────────────────────────────────────

_model_available = is_available()
reason = "bge-small-en-v1.5 not available (no sentence-transformers or download failed)"

requires_model = pytest.mark.skipif(not _model_available, reason=reason)

# ── Embedding shape tests ─────────────────────────────────────


class TestEmbedding:
    def test_model_available_or_not(self):
        """Sanity: is_available() returns a bool.  Doesn't require model."""
        assert isinstance(is_available(), bool)

    @requires_model
    def test_embed_texts_shape(self):
        vecs = embed_texts(["hello world", "second text"])
        assert vecs is not None
        assert vecs.shape == (2, 384)

    @requires_model
    def test_embed_query_shape(self):
        vec = embed_query("what is asyncio")
        assert vec is not None
        assert vec.shape == (384,)

    @requires_model
    def test_embed_texts_normalized(self):
        """Vectors are L2-normalized → norm ≈ 1.0."""
        import numpy as np

        vecs = embed_texts(["test sentence for norm check"])
        assert vecs is not None
        norm = np.linalg.norm(vecs[0])
        assert abs(norm - 1.0) < 1e-4

    @requires_model
    def test_embed_query_normalized(self):
        import numpy as np

        vec = embed_query("normalize this")
        assert vec is not None
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-4

    @requires_model
    def test_embed_texts_single_string(self):
        """Single string should return (1, 384) not (384,)."""
        vecs = embed_texts(["one"])
        assert vecs is not None
        assert vecs.shape == (1, 384)

    @requires_model
    def test_embed_texts_empty_list(self):
        """Empty list should return empty array, not crash.

        SentenceTransformers returns (0,) for empty input — this is fine
        because rank_chunks() guards against empty chunks before calling
        embed_texts(), so the empty path is never hit in production.
        """
        vecs = embed_texts([])
        assert vecs is not None
        assert vecs.shape[0] == 0  # 0 rows, any number of columns


# ── Ranking tests ─────────────────────────────────────────────


class TestRanking:
    @requires_model
    def test_rank_returns_top_k(self):
        # All chunks relevant to the query → all survive the D-024 gap filter
        # → the top_k cap is the only thing limiting the count.
        chunks = [
            {"text": "Python is a popular programming language used for web development.", "parent_text": "...", "chunk_index": 0},
            {"text": "Python supports multiple programming paradigms including OOP.", "parent_text": "...", "chunk_index": 1},
            {"text": "Python has a large standard library and ecosystem.", "parent_text": "...", "chunk_index": 2},
            {"text": "Python is known for its readable syntax and simplicity.", "parent_text": "...", "chunk_index": 3},
        ]
        ranked = rank_chunks("what is python", chunks, top_k=2)
        assert ranked is not None
        assert len(ranked) == 2
        # Most relevant Python chunk should rank highest.
        assert "Python" in ranked[0]["text"]

    @requires_model
    def test_rank_scores_descending(self):
        chunks = [
            {"text": f"document about topic {i}", "parent_text": "...", "chunk_index": i}
            for i in range(5)
        ]
        ranked = rank_chunks("topic 2", chunks, top_k=3)
        assert ranked is not None
        scores = [r["score"] for r in ranked]
        # Scores should be strictly non-increasing.
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    @requires_model
    def test_rank_all_scores_in_negative_one_to_one(self):
        """Cosine similarity should lie in [-1, 1]."""
        chunks = [
            {"text": f"doc {i}", "parent_text": "...", "chunk_index": i}
            for i in range(3)
        ]
        ranked = rank_chunks("query", chunks, top_k=3)
        assert ranked is not None
        for r in ranked:
            assert -1.0 <= r["score"] <= 1.0

    @requires_model
    def test_rank_carries_source_metadata(self):
        chunks = [
            {
                "text": "Berlin is the capital.", "parent_text": "...",
                "chunk_index": 0, "source": {"url": "https://example.com", "title": "Facts"},
            }
        ]
        ranked = rank_chunks("capital of germany", chunks, top_k=1)
        assert ranked is not None
        assert "source" in ranked[0]
        assert ranked[0]["source"]["url"] == "https://example.com"

    @requires_model
    def test_rank_empty_chunks_returns_empty(self):
        assert rank_chunks("query", [], top_k=3) == []

    def test_rank_null_when_model_unavailable(self):
        """If is_available() is False, rank_chunks returns None."""
        if is_available():
            return  # model is loaded — nothing to test here
        assert rank_chunks("q", [{"text": "x", "parent_text": "", "chunk_index": 0}]) is None

    @requires_model
    def test_rank_top_k_exceeds_available(self):
        """top_k > len(chunks) should return all chunks."""
        chunks = [
            {"text": "a", "parent_text": "...", "chunk_index": 0},
            {"text": "b", "parent_text": "...", "chunk_index": 1},
        ]
        ranked = rank_chunks("query", chunks, top_k=10)
        assert ranked is not None
        assert len(ranked) == 2

    @requires_model
    def test_diversify_dedups_same_source_url(self):
        """Top-2 should pick different URLs when both are available."""
        chunks = [
            {"text": "Python is great.", "parent_text": "...", "chunk_index": 0,
             "source": {"url": "https://a.com/page"}},
            {"text": "Python is also great.", "parent_text": "...", "chunk_index": 1,
             "source": {"url": "https://a.com/page"}},
            {"text": "Python is popular.", "parent_text": "...", "chunk_index": 2,
             "source": {"url": "https://b.com/other"}},
        ]
        ranked = rank_chunks("why is python great", chunks, top_k=2)
        assert ranked is not None
        assert len(ranked) == 2
        # Both chunks should be from different URLs.
        urls = {r["source"]["url"] for r in ranked}
        assert len(urls) == 2

    @requires_model
    def test_diversify_fallback_when_too_few_sources(self):
        """Only 1 unique URL but top_k=2 → fill from same source."""
        chunks = [
            {"text": "A chunk about Python.", "parent_text": "...", "chunk_index": 0,
             "source": {"url": "https://only.com/page"}},
            {"text": "Another chunk about Python.", "parent_text": "...", "chunk_index": 1,
             "source": {"url": "https://only.com/page"}},
            {"text": "Third chunk about Python.", "parent_text": "...", "chunk_index": 2,
             "source": {"url": "https://only.com/page"}},
        ]
        ranked = rank_chunks("python", chunks, top_k=2)
        assert ranked is not None
        assert len(ranked) == 2  # fills second slot from same URL

    @requires_model
    def test_diversify_no_source_url_does_not_crash(self):
        """Chunks without source.url are treated as unique (no crash)."""
        chunks = [
            {"text": "Python is fun.", "parent_text": "...", "chunk_index": 0},
            {"text": "Python is nice.", "parent_text": "...", "chunk_index": 1},
        ]
        ranked = rank_chunks("python", chunks, top_k=2)
        assert ranked is not None
        assert len(ranked) == 2


# ── D-024 filter tests (deterministic — mocked embeddings) ────


def _mock_embeddings(monkeypatch, scores):
    """Patch ranking_service so chunk i gets cosine `scores[i]` vs the query.

    Vectors are unit-norm and aligned so dot(query, chunk_i) == scores[i].
    Makes the floor/gap filter logic testable without the real model.
    """
    import numpy as np
    from app.services import ranking_service

    def fake_embed_texts(texts):
        # unit vectors: [score, sqrt(1-score^2), 0] → dot with query = score
        return np.array(
            [[s, (1.0 - s * s) ** 0.5, 0.0] for s in scores],
            dtype=np.float32,
        )

    def fake_embed_query(q):
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    monkeypatch.setattr(ranking_service, "is_available", lambda: True)
    monkeypatch.setattr(ranking_service, "embed_texts", fake_embed_texts)
    monkeypatch.setattr(ranking_service, "embed_query", fake_embed_query)


def _chunks(n, urls=None):
    """Build n chunk dicts, optionally with source URLs for diversity tests."""
    out = []
    for i in range(n):
        c = {"text": f"chunk {i}", "parent_text": "...", "chunk_index": i}
        if urls:
            c["source"] = {"url": urls[i], "title": f"t{i}"}
        out.append(c)
    return out


class TestRankingFloorFilter:
    """D-024 stage 1: absolute floor — max score below floor → return []."""

    def test_all_noise_returns_empty(self, monkeypatch):
        _mock_embeddings(monkeypatch, [0.1, 0.05, 0.02])  # all < 0.2 floor
        ranked = rank_chunks("query", _chunks(3), top_k=3)
        assert ranked == []

    def test_floor_is_strictly_below(self, monkeypatch):
        # max == 0.2 exactly → NOT below → survives (boundary check).
        _mock_embeddings(monkeypatch, [0.2, 0.1])
        ranked = rank_chunks("query", _chunks(2), top_k=2, gap_threshold=1.0)
        assert ranked is not None
        assert len(ranked) >= 1  # top survives

    def test_floor_returns_empty_not_none(self, monkeypatch):
        """Floor path returns [] (data), not None (which means model unavailable)."""
        _mock_embeddings(monkeypatch, [0.05])
        ranked = rank_chunks("query", _chunks(1), top_k=1)
        assert ranked == []
        assert ranked is not None


class TestRankingGapFilter:
    """D-024 stage 2: relative gap — drop chunks far below the top hit."""

    def test_drops_chunks_beyond_gap(self, monkeypatch):
        # top=0.8, gap=0.15 → keep 0.8 & 0.7 (gap 0.1), drop 0.5 (gap 0.3)
        _mock_embeddings(monkeypatch, [0.8, 0.7, 0.5])
        ranked = rank_chunks("query", _chunks(3), top_k=3, gap_threshold=0.15)
        assert ranked is not None
        scores = [r["score"] for r in ranked]
        # 0.5 (gap 0.3 > 0.15) dropped; 0.8 & 0.7 kept. Use tolerance —
        # float32 makes 0.8 -> 0.8000000119...
        assert all(abs(s - 0.5) > 1e-3 for s in scores)
        assert any(abs(s - 0.8) < 1e-3 for s in scores)
        assert any(abs(s - 0.7) < 1e-3 for s in scores)

    def test_keeps_at_least_top1(self, monkeypatch):
        # top=0.5 (above floor), others way below → only top survives
        _mock_embeddings(monkeypatch, [0.5, 0.1, 0.05])
        ranked = rank_chunks("query", _chunks(3), top_k=3, gap_threshold=0.15)
        assert ranked is not None
        assert len(ranked) == 1
        assert abs(ranked[0]["score"] - 0.5) < 1e-3

    def test_all_close_survive(self, monkeypatch):
        # scores clustered → all within gap → all kept (then capped by top_k)
        _mock_embeddings(monkeypatch, [0.6, 0.58, 0.55])
        ranked = rank_chunks("query", _chunks(3), top_k=3, gap_threshold=0.15)
        assert ranked is not None
        assert len(ranked) == 3

    def test_gap_runs_before_diversity(self, monkeypatch):
        """A dropped chunk must NOT be resurrected by the diversity fill."""
        # top=0.8 (url a), second=0.6 (url a, dropped by gap), third=0.5 (url b, dropped)
        _mock_embeddings(monkeypatch, [0.8, 0.6, 0.5])
        urls = ["https://a.com", "https://a.com", "https://b.com"]
        ranked = rank_chunks("query", _chunks(3, urls), top_k=2, gap_threshold=0.15)
        assert ranked is not None
        # Only the top survived the gap → only 1 result, even though top_k=2
        # and a second URL exists. Diversity can't pull back dropped chunks.
        assert len(ranked) == 1
        assert ranked[0]["source"]["url"] == "https://a.com"
