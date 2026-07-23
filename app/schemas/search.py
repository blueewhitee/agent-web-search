from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    render_js: bool = False         # D-006: opt-in crawl4ai per request
    include_content: bool = False   # D-002: opt-in raw_content in response


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


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    unresponsive_engines: list[str] = []
