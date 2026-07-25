# Agent Web Search

Self-hosted search API for AI agents. Think Tavily/Exa — free, self-hostable, BYOK.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)


## Quick start

```bash
git clone https://github.com/blueewhitee/agent-web-search.git
cd agent-web-search
docker compose up
```

Full stack running on `http://localhost:8000`.

## Usage

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "python asyncio tutorial", "include_content": true}'
```

### Parameters

| Param | Default | Description |
|---|---|---|
| `query` | — | Search query (1–1000 chars) |
| `include_content` | false | Return extracted + ranked chunks |
| `render_js` | false | Enable crawl4ai for JS-heavy pages |

### Response

```json
{
  "query": "python asyncio tutorial",
  "ranked_chunks": [
    {
      "text": "256-token chunk for embedding",
      "parent_text": "512-token window for LLM context",
      "score": 0.877,
      "source": {
        "url": "https://docs.python.org/3/library/asyncio.html",
        "title": "asyncio — Asynchronous I/O",
        "searxng_score": 1.0
      }
    }
  ]
}
```

## Pipeline

```
query → SearXNG → fetch → extract → scrub → chunk → embed → rank → JSON
```

| Stage | What |
|---|---|
| 1 | SearXNG multi-engine meta-search |
| 2 | Concurrent fetch + URL dedup + graceful degradation |
| 3 | Text extraction (trafilatura → readability-lxml), markdown output |
| 3.5 | Prompt-injection detection + redaction |
| 4 | 256-token recursive chunking with 512-token parent context |
| 5 | bge-small-en-v1.5 embedding + cosine ranking + source diversity |

## Configuration

Environment variables (or `.env`):

| Env | Default | Description |
|---|---|---|
| `SEARXNG_BASE_URL` | `http://localhost:8080` | SearXNG endpoint |
| `TOP_K_FETCH` | `5` | URLs to fetch per query |
| `TOP_K_RETURN` | `3` | Chunks to return |
| `PER_URL_TIMEOUT` | `6` | Seconds per URL fetch |
| `BATCH_DEADLINE` | `8` | Batch fetch timeout |

## Development

```bash
uv sync
uv run uvicorn main:app --reload
uv run pytest    # 94 tests, 0 failures
```

Requires SearXNG on port 8080 (`docker compose up searxng`).

## License

MIT
