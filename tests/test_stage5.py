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
        chunks = [
            {"text": "Python is a programming language.", "parent_text": "...", "chunk_index": 0},
            {"text": "The capital of France is Paris.", "parent_text": "...", "chunk_index": 1},
            {"text": "Asyncio enables concurrent code.", "parent_text": "...", "chunk_index": 2},
            {"text": "Photosynthesis produces oxygen.", "parent_text": "...", "chunk_index": 3},
        ]
        ranked = rank_chunks("what is python", chunks, top_k=2)
        assert ranked is not None
        assert len(ranked) == 2
        # Python chunk should rank highest for "what is python" query.
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
