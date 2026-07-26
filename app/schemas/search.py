from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    render_js: bool = False         # D-006: opt-in crawl4ai per request
    include_content: bool = False   # D-002: opt-in raw_content in response
    # D-0XX: optional SearXNG category override. When None (default), the
    # router auto-detects intent and routes to news/it/general accordingly.
    # Explicitly set to e.g. ["news"] to force a specific category.
    categories: list[str] | None = None
    time_range: str | None = None   # "day"|"week"|"month"|"year" (news only)


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str  # SearXNG's "content" field — renamed for clarity
    # D-003: upstream popularity metadata, NOT final ranking. The name is
    # load-bearing: `score` is reserved for Stage 5 semantic similarity.
    searxng_score: float
    low_confidence: bool = False           # D-001: fetch failed/robots-blocked/timeout — not a 504
    # D-002: extracted clean text (Stage 3 complete). Plain UTF-8, ≥200 chars when
    # extraction succeeded; None when extraction failed or the caller didn't opt in.
    # Populated only when SearchRequest.include_content=True.
    raw_content: str | None = None
    # Stage 4: chunks for retrieval. Populated when include_content=True.
    chunks: list[dict] | None = None


class ChunkSource(BaseModel):
    """Provenance: where this chunk came from."""
    url: str
    title: str
    searxng_score: float  # D-003 upstream metadata


class RankedChunk(BaseModel):
    """A ranked chunk returned to the caller (D-008, D-013).

    text:        ~256 tokens — the embedding-targeted snippet.
    parent_text: ~512 tokens — full context for the LLM to read.
    score:       cosine similarity against the query (D-003).
    source:      provenance metadata (url, title, searxng_score).
    """
    text: str
    parent_text: str
    chunk_index: int
    score: float
    source: ChunkSource


class SearchResponse(BaseModel):
    query: str
    ranked_chunks: list[RankedChunk] = []
    unresponsive_engines: list[str] = []
