# Agent Web Search

Self-hosted, privacy-first **search API for AI agents**. Like Tavily/Exa — but free, self-hostable, and open source.

Takes a query → SearXNG → fetch → extract → chunk → embed → rank → returns clean JSON with ranked chunks for LLM consumption.

## Quick start

```bash
git clone https://github.com/blueewhitee/agent-web-search.git
cd agent-web-search
docker compose up
```

Full stack starts (SearXNG + API) on `localhost:8000`.

## Usage

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "python asyncio tutorial", "include_content": true}'
```

### Parameters

| Param | Type | Default | Description |
|---|---|---|---|
| `query` | string | — | Search query (1–1000 chars) |
| `include_content` | bool | false | Return extracted + ranked chunks |
| `render_js` | bool | false | Enable crawl4ai for JS-heavy pages |

### Response

```json
{
  "query": "python asyncio tutorial",
  "ranked_chunks": [
    {
      "text": "256-token chunk for embedding",
      "parent_text": "512-token window for LLM context",
      "chunk_index": 0,
      "score": 0.877,
      "source": {
        "url": "https://docs.python.org/3/library/asyncio.html",
        "title": "asyncio — Asynchronous I/O",
        "searxng_score": 1.0
      }
    }
  ],
  "unresponsive_engines": ["brave", "google"]
}
```

For `include_content=false`, returns `"ranked_chunks": []` (only metadata available).

## Pipeline

```
query → SearXNG → dedup → fetch → extract → scrub → chunk → embed → rank → JSON
```

| Stage | What | Why |
|---|---|---|
| 0 | Query rewrite | Deferred to the calling harness (D-009) |
| 1 | SearXNG | Multi-engine meta-search |
| 2 | Fetch + dedup | Concurrent fetch with graceful degradation |
| 3 | Extract | trafilatura + readability-lxml. Markdown output |
| 3.5 | Scrub | Prompt-injection detection + redaction |
| 4 | Chunk | 256-token recursive split, 512-token parent window |
| 5 | Embed + rank | bge-small-en-v1.5, cosine similarity, source diversity |
| 6 | Assemble | Chunk-centric JSON response |

## Configuration

Set via environment variables (or `.env`):

| Env | Default | Description |
|---|---|---|
| `SEARXNG_BASE_URL` | `http://localhost:8080` | SearXNG endpoint |
| `TOP_K_FETCH` | `5` | URLs to fetch per query |
| `TOP_K_RETURN` | `3` | Chunks to return |
| `PER_URL_TIMEOUT` | `6.0` | Seconds per URL fetch |
| `BATCH_DEADLINE` | `8.0` | Batch fetch timeout |

## Development

```bash
# Install
uv sync

# Run
uv run uvicorn main:app --reload

# Test
uv run pytest
```

Requires SearXNG running on port 8080 (`docker compose up searxng`).

**94 tests, 0 failures.**

## Project structure

```
app/
  api/search.py         # Router (orchestrator)
  core/config.py        # Settings
  schemas/search.py     # Request/response models
  services/
    searxng_service.py  # Stage 1
    fetch_service.py    # Stage 2
    robots_cache.py     # Robots.txt cache
    _crawl4ai_adapter.py
    extraction_service.py  # Stage 3
    scrubber_service.py    # Stage 3.5
    chunking_service.py    # Stage 4
    embedding_service.py   # Stage 5
    ranking_service.py     # Stage 5
  utils/url.py          # URL normalize + dedup
tests/
main.py                 # FastAPI app + lifespan
```

## Why not agent-search?

Our 6-stage pipeline from scratch (~1.5K LOC) is the interview artifact. [agent-search](https://github.com/brcrusoe72/agent-search) returns full-page text blobs to the LLM. We chunk, embed, rank, and return parent-context windows — the same way production RAG systems work.

## License

MIT
