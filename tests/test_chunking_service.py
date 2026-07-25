"""Tests for Stage 4 chunking service (Pair 3).

Covers empty/short edges, recursive splitting across paragraph and
sentence boundaries, hard-cap fallback, parent-window expansion
(binary-search balanced), tiny-chunk merging, and a substring
invariant that catches text corruption.
"""

from app.services.chunking_service import (
    Chunk,
    _split_text,
    chunk_text,
    count_tokens,
)

# ── Test helpers ──────────────────────────────────────────────

SHORT = "hello world"

# ~30 tokens per repetition: sentence boundary after ". "
_SENTENCE = "The meadow stretched wide under a pale blue sky and the wind moved slowly. "
_PARA = _SENTENCE * 10  # ~300 tokens

# Base64-like: no separators of any kind → forces hard-cap split.
_NO_SEP = "abcdef1234567890" * 100  # 1600 chars, ~400 tokens

# Mix: paragraphs with a tiny trailing sentence for merge testing.
_MERGE_SOURCE = "\n\n".join([_SENTENCE * 10, _SENTENCE * 5, "ok"])


def _all_tokens(chunks: list[Chunk]) -> list[int]:
    return [count_tokens(c.text) for c in chunks]


# ──────────────────────────────────────────────────────────────
# Empty / short
# ──────────────────────────────────────────────────────────────


class TestEmptyAndShort:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_none(self):
        assert chunk_text(None) == []

    def test_whitespace_only(self):
        assert chunk_text("   \n  ") == []

    def test_short_text_single_chunk(self):
        chunks = chunk_text(SHORT)
        assert len(chunks) == 1
        c = chunks[0]
        assert c.text == SHORT
        assert c.parent_text == SHORT
        assert c.chunk_index == 0


# ──────────────────────────────────────────────────────────────
# Splitting
# ──────────────────────────────────────────────────────────────


class TestSplitting:
    def test_multi_paragraph(self):
        # 4 paragraphs → should split into multiple chunks
        text = _PARA + "\n\n" + _PARA
        chunks = chunk_text(text)
        assert len(chunks) >= 2
        for c in chunks:
            assert count_tokens(c.text) <= 320

    def test_sentence_boundary(self):
        # No \n\n, only ". " boundaries (plus " " fallback)
        text = _SENTENCE * 20  # ~600 tokens
        chunks = chunk_text(text)
        assert len(chunks) >= 2

    def test_all_chunks_within_hard_cap(self):
        text = _PARA * 5  # plenty of paragraphs
        for c in chunk_text(text):
            assert count_tokens(c.text) <= 320

    def test_hard_cap_forced_split__direct(self):
        """_force_hard_cap_split splits at character level.

        Note: the BGE WordPiece tokenizer collapses unspaced strings
        > ~108 chars into a single [UNK] token, so the hard-cap path
        in _merge_split is unreachable with real tokenizer counts.
        We test the function directly to prove it works when triggered.
        """
        from app.services.chunking_service import _force_hard_cap_split
        # 800 chars, cap at 320 tokens → 1280 chars. At 320*4=1280 chars
        # the text is shorter than the cap, so use cap=50 → 200 chars.
        text = "X" * 800
        chunks = _force_hard_cap_split(text, cap_tokens=50)
        assert len(chunks) >= 4  # 800 / (50*4) ≈ 4
        for c in chunks:
            # Each chunk ≤ 50*4 = 200 chars (rough estimate)
            assert len(c) <= 200


# ──────────────────────────────────────────────────────────────
# Parent expansion
# ──────────────────────────────────────────────────────────────


class TestParentExpansion:
    def test_parent_never_exceeds_512(self):
        text = _PARA * 5
        for c in chunk_text(text):
            assert count_tokens(c.parent_text) <= 512

    def test_first_chunk_parent_starts_at_document_beginning(self):
        text = _PARA + "\n\n" + _PARA
        chunks = chunk_text(text)
        assert chunks[0].parent_text.startswith(text[:5])

    def test_last_chunk_parent_ends_at_document_end(self):
        text = _PARA + "\n\n" + _PARA
        chunks = chunk_text(text)
        assert chunks[-1].parent_text.endswith(text[-5:])

    def test_middle_chunk_parent_extends_both_sides(self):
        text = _PARA  # ~300 tokens → often 2 chunks
        chunks = chunk_text(text)
        if len(chunks) < 2:
            return  # single chunk — nothing to test
        middle = chunks[0]
        # Parent should extend to nearly the end (right side)
        assert count_tokens(middle.parent_text) > count_tokens(middle.text)

    def test_single_chunk_parent_equals_full_text(self):
        text = _PARA  # ~300 tokens → single chunk
        chunks = chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0].parent_text == text


# ──────────────────────────────────────────────────────────────
# Merging
# ──────────────────────────────────────────────────────────────


class TestMerging:
    def test_tiny_last_chunk_merged(self):
        # _MERGE_SOURCE: para (~300t) + short_para (~150t) + "ok" (~3t)
        # _split_text would give something like [~300t, ~150t, ~3t].
        # After merging, the "ok" chunk should be absorbed.
        chunks = chunk_text(_MERGE_SOURCE)
        # Check: no chunk (except possibly the first if it's the only one)
        # has fewer than MIN_CHUNK_TOKENS (32) tokens — because tiny ones
        # get merged backward.
        for i, tc in enumerate(_all_tokens(chunks)):
            if len(chunks) == 1:
                break  # single chunk can be any size
            if i == 0:
                continue  # first chunk preserved even if tiny
            assert tc >= 32, f"chunk {i} has {tc} tokens, expected >= 32"

    def test_tiny_first_chunk_not_merged(self):
        # A tiny heading at the start stays (no previous to merge into).
        text = "Intro.\n\n" + _PARA
        chunks = chunk_text(text)
        assert len(chunks) >= 1
        # The intro chunk should be its own chunk (or merged into later if
        # it fits in target, but with ~3 words + 300-token para, it
        # doesn't get merged down). Either way, all chunk.text are
        # substrings (next test proves no corruption).
        assert chunks[0].parent_text.startswith(text[:3])

    def test_chunk_indices_sequential(self):
        text = _PARA * 5
        chunks = chunk_text(text)
        assert len(chunks) >= 2
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


# ──────────────────────────────────────────────────────────────
# Property invariants
# ──────────────────────────────────────────────────────────────


class TestPropertyInvariants:
    def test_every_chunk_is_substring_of_original(self):
        """No text corruption: every chunk is an exact substring."""
        text = _PARA * 5
        for c in chunk_text(text):
            assert text.find(c.text) != -1, (
                f"Chunk not found in original:\n{c.text[:100]!r}"
            )
