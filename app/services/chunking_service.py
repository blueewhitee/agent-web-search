"""Stage 4: Text chunking with recursive character splitting.

Pure function, stateless. Lazily-cached BGE tokenizer for accurate token counting.
Splits text into ~256-token chunks, hard-capped at 320 tokens.
Each chunk gets an expanded ~512-token parent window for LLM context.

Public API:
    chunk_text(text) -> list[Chunk]
    count_tokens(text) -> int
"""

from __future__ import annotations

from dataclasses import dataclass

# ──────────────────────────────────────────────
# Public types
# ──────────────────────────────────────────────


@dataclass
class Chunk:
    """A chunk of text with surrounding context.

    text:        ~256 tokens — fed to the embedding model for retrieval.
    parent_text: ~512 tokens — returned to the LLM for answer generation.
    chunk_index: preserves original ordering within the document.
    """

    text: str
    parent_text: str
    chunk_index: int


# ──────────────────────────────────────────────
# Tokenizer
# ──────────────────────────────────────────────

_tokenizer = None
_TOKENIZER_FALLBACK = False


def _load_tokenizer():
    """Load BGE tokenizer once, cache at module level. ~2MB, ~50ms first call."""
    global _tokenizer, _TOKENIZER_FALLBACK
    if _tokenizer is not None:
        return
    try:
        from transformers import AutoTokenizer

        _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    except ImportError:
        _TOKENIZER_FALLBACK = True


def count_tokens(text: str) -> int:
    """Count tokens in text using BGE tokenizer. Falls back to 1 token ≈ 4 chars."""
    _load_tokenizer()
    if _TOKENIZER_FALLBACK or _tokenizer is None:
        return len(text) // 4
    # add_special_tokens=False: count text-only tokens.
    # The embedding model adds its own [CLS]/[SEP] at call time;
    # counting without them gives accurate sizing against the 512 limit.
    return len(_tokenizer.encode(text, add_special_tokens=False))


# ──────────────────────────────────────────────
# Pair 1: Recursive character splitting
# ──────────────────────────────────────────────

SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " "]
TARGET_TOKENS = 256
HARD_CAP_TOKENS = 320
MIN_CHUNK_TOKENS = 32
PARENT_TOKENS = 512


def _force_hard_cap_split(text: str, cap_tokens: int) -> list[str]:
    """Last resort: split at character level when text resists all separators."""
    char_cap = cap_tokens * 4
    chunks: list[str] = []
    for i in range(0, len(text), char_cap):
        piece = text[i : i + char_cap].strip()
        if piece:
            chunks.append(piece)
    return chunks


def _merge_split(
    text: str,
    sep_idx: int = 0,
    target: int = TARGET_TOKENS,
    hard_cap: int = HARD_CAP_TOKENS,
) -> list[str]:
    """Recursively split text into chunks of ~target tokens."""
    if not text.strip():
        return []

    if count_tokens(text) <= target:
        return [text]

    if sep_idx >= len(SEPARATORS):
        return _force_hard_cap_split(text, hard_cap)

    separator = SEPARATORS[sep_idx]
    pieces = text.split(separator)

    chunks: list[str] = []
    current = ""

    for piece in pieces:
        if not piece:
            continue

        candidate = current + separator + piece if current else piece

        if count_tokens(candidate) <= target:
            current = candidate
        else:
            if current:
                chunks.append(current)

            if count_tokens(piece) <= target:
                current = piece
            else:
                sub_chunks = _merge_split(piece, sep_idx + 1, target, hard_cap)
                chunks.extend(sub_chunks)
                current = ""

    if current.strip():
        chunks.append(current)

    return chunks


def _split_text(
    text: str,
    target: int = TARGET_TOKENS,
    hard_cap: int = HARD_CAP_TOKENS,
) -> list[str]:
    """Split text into chunks of ~target tokens."""
    if not text or not text.strip():
        return []
    return _merge_split(text, sep_idx=0, target=target, hard_cap=hard_cap)


# ──────────────────────────────────────────────
# Pair 2: Positions + Merging + Parent Windows
# ──────────────────────────────────────────────


def _compute_positions(
    text: str, chunks: list[str]
) -> list[tuple[str, int, int]]:
    """Find each chunk's (start, end) character offset in the original text.

    Uses sequential offset tracking so duplicate chunks map to
    the correct occurrence.
    """
    positions: list[tuple[str, int, int]] = []
    search_offset = 0
    for chunk in chunks:
        pos = text.find(chunk, search_offset)
        if pos == -1:
            pos = search_offset  # defensive fallback
        end_pos = pos + len(chunk)
        positions.append((chunk, pos, end_pos))
        search_offset = end_pos
    return positions


def _merge_tiny_chunks(
    chunks: list[tuple[str, int, int]],
    min_tokens: int = MIN_CHUNK_TOKENS,
) -> list[tuple[str, int, int]]:
    """Merge chunks smaller than min_tokens into the previous chunk.

    Works on (text, char_start, char_end) tuples so position spans
    stay accurate. Iterates backward so indices don't shift under pop().
    First chunk (index 0) is never merged — rare heading case.
    """
    if not chunks:
        return []

    merged = list(chunks)
    i = len(merged) - 1

    while i > 0:
        text_i, start_i, end_i = merged[i]
        if count_tokens(text_i) < min_tokens:
            prev_text, prev_start, prev_end = merged[i - 1]
            merged[i - 1] = (
                prev_text + " " + text_i,
                prev_start,
                end_i,
            )
            merged.pop(i)
        i -= 1

    return merged


def _expand_parent(
    text: str,
    chunk_start: int,
    chunk_end: int,
    parent_tokens: int = PARENT_TOKENS,
) -> str:
    """Return the widest window around [chunk_start, chunk_end] that fits
    within parent_tokens. Binary search — O(log n) tokenizer calls.
    """
    total_len = len(text)
    chunk_tok = count_tokens(text[chunk_start:chunk_end])
    budget = parent_tokens - chunk_tok

    if budget <= 0:
        return text[chunk_start:chunk_end]

    left_budget = budget // 2

    lo, hi = 0, chunk_start
    while lo < hi:
        mid = (lo + hi) // 2
        if count_tokens(text[mid:chunk_start]) <= left_budget:
            hi = mid
        else:
            lo = mid + 1
    start = lo

    actual_left = count_tokens(text[start:chunk_start])
    remaining = parent_tokens - actual_left - chunk_tok

    lo, hi = chunk_end, total_len
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count_tokens(text[start:mid]) <= parent_tokens:
            lo = mid
        else:
            hi = mid - 1
    end = lo

    return text[start:end]


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────


def chunk_text(text: str | None) -> list[Chunk]:
    """Split text into chunks with parent context for retrieval.

    Pipeline: split → compute positions → merge tiny → expand
    parents → assemble Chunks.

    Returns empty list for None / empty / whitespace-only input.
    """
    if not text or not text.strip():
        return []

    raw_texts = _split_text(text)
    with_positions = _compute_positions(text, raw_texts)
    merged = _merge_tiny_chunks(with_positions)

    chunks: list[Chunk] = []
    for i, (chunk_str, start, end) in enumerate(merged):
        parent = _expand_parent(text, start, end)
        chunks.append(
            Chunk(text=chunk_str, parent_text=parent, chunk_index=i)
        )

    return chunks
