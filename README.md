<div align="center">

<!-- Replace with a real demo screenshot/GIF (MCP tool used inside an agent).
     This is the #1 star-growth signal per 2026 README audits (+35% star rate). -->

<p align="center">
  <img
    src="https://github.com/user-attachments/assets/eefb4a02-c622-4689-a063-6599585d77a7"
    alt="Agent Web Search used as a web_search tool inside Claude Desktop"
    width="720"
  />
  <br>
  <em>Agent Web Search used as a <code>web_search</code> tool inside Claude Desktop, returning ranked chunks with cosine similarity scores.</em>
</p>

# Agent Web Search

**Self-hosted search API for AI agents — free, private, drop-in replacement for Tavily/Exa.**

6-stage pipeline: search → fetch → extract → scrub → chunk → embed → rank.  
Returns Chonkie-chunked, cosine-ranked JSON ready for RAG.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/blueewhitee/agent-web-search?style=social)](https://github.com/blueewhitee/agent-web-search)
[![Python](https://img.shields.io/badge/Python-3.12+-blue?logo=python)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue?logo=docker)](https://docs.docker.com/)
[![MCP](https://img.shields.io/badge/MCP-ready-purple)](https://modelcontextprotocol.io)
[![PyPI](https://img.shields.io/pypi/v/agent-web-search?logo=pypi&logoColor=white)](https://pypi.org/project/agent-web-search/)
[![Tests](https://img.shields.io/badge/tests-175%20passed-brightgreen)](#development)

</div>

## Quick start

```bash
git clone https://github.com/blueewhitee/agent-web-search.git
cd agent-web-search
docker compose up
```

Full stack on `http://localhost:8000` — SearXNG + FastAPI + embedding model, **zero API keys required**.

```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "python asyncio tutorial", "include_content": true}'
```

**Or use the Python SDK** (one `pip install`):

```python
from agent_web_search import AgentWebSearch

with AgentWebSearch(base_url="http://localhost:8000") as client:
    response = client.search("python asyncio", include_content=True)
    for chunk in response.ranked_chunks:
        print(f"{chunk.score:.3f}  {chunk.source.url}")
```

**Wire it into an MCP agent** (Claude Desktop, Cursor, Pi):

```json
{
  "mcpServers": {
    "nature-search": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_server.server"],
      "env": { "NATURE_SE_API_URL": "http://localhost:8000" }
    }
  }
}
```

<details>
<summary>Table of Contents</summary>

- [Why Agent Web Search?](#why-agent-web-search)
- [Pipeline](#pipeline)
- [Features](#features)
- [API Reference](#api-reference)
- [MCP Integration](#mcp-integration)
- [Python SDK](#python-sdk)
- [SearXNG Configuration](#searxng-configuration)
- [Configuration](#configuration)
- [Development](#development)
- [FAQ](#faq)
- [Contributing](#contributing)
- [License](#license)

</details>

## Why Agent Web Search?

Tavily and Exa are excellent SaaS products. Agent Web Search is for when you want the same output shape but need it **self-hosted, air-gapped, or free** — with full control over which engines are queried and how content is extracted.

| | Agent Web Search | Tavily | Exa |
|---|---|---|---|
| **Cost** | Free, self-hosted | $0.05/query (free tier limited) | $0.005/query |
| **Privacy** | Your machine, your data | Cloud API | Cloud API |
| **Engine control** | Full SearXNG config | Black box | Black box |
| **MCP native** | ✅ Built-in | ❌ (requires wrapper) | ❌ (requires wrapper) |
| **JS rendering** | ✅ crawl4ai opt-in | ❌ | ❌ |
| **Prompt scrubbing** | ✅ Built-in | ❌ | ❌ |
| **Category routing** | ✅ Intent-aware | ❌ | ❌ |
| **RAG chunk output** | ✅ 256/512-token with scores | ❌ (raw content) | ❌ (raw content) |

## Pipeline

```
query → SearXNG → fetch → extract → scrub → chunk → embed → rank → JSON
  │        │        │        │        │       │        │        │
  │        │        │        │        │       │        │        └─ cosine ranking
  │        │        │        │        │       │        │           + source diversity
  │        │        │        │        │       │        └─ bge-small-en-v1.5
  │        │        │        │        │       │           (384-dim embeddings)
  │        │        │        │        │       └─ 256-token recursive chunks
  │        │        │        │        │          + 512-token parent windows
  │        │        │        │        └─ prompt-injection detection
  │        │        │        │           + automated redaction
  │        │        │        └─ trafilatura → readability-lxml
  │        │        │           two-tier extraction, markdown output
  │        │        └─ concurrent fetch + URL dedup
  │        │           graceful degradation on timeout
  │        └─ intent routing → it | news | general
  │           word-boundary regex, no false positives
  └─ 5+ engines (duckduckgo, mojeek, qwant, stackoverflow,
     github, mdn, bing news, reuters)
```

| Stage | Technology | What happens |
|---|---|---|
| 1 — Search | SearXNG with intent routing | `CODE` → stackoverflow/github/mdn, `NEWS` → bing news/reuters, `DEFAULT` → duckduckgo/qwant |
| 2 — Fetch | `httpx` + `asyncio` | Concurrent fetch, URL dedup, 6s per-URL timeout, 8s batch deadline |
| 3 — Extract | `trafilatura` → `readability-lxml` | Two-tier: trafilatura for clean pages, readability-lxml fallback |
| 3.5 — Scrub | In-house scrubber | Prompt-injection detection, automated redaction |
| 4 — Chunk | `Chonkie` | 256-token recursive split, 32-token overlap, 512-token parent window |
| 5 — Rank | `sentence-transformers` | `bge-small-en-v1.5` embeddings, cosine similarity, per-URL diversity cap |

## Features

| Feature | Description |
|---|---|
| 🔀 **Intent-aware routing** | Code questions hit StackOverflow/GitHub/MDN. News queries hit Reuters/Bing News. Everything else uses general search. Deterministic word-boundary regex — no LLM call, no false positives. |
| 📄 **Content extraction with quality filter** | Two-tier pipeline (trafilatura → readability-lxml) drops index pages, aggregator shells, and author-list soup via density heuristics. |
| 🛡️ **Prompt-injection scrubbing** | Detects and redacts prompt-injection patterns in fetched content before it reaches your LLM. |
| 🧩 **Chunk + embed + rank** | 256-token Chonkie chunks with 512-token parent windows, `bge-small-en-v1.5` embeddings, cosine-ranked with source diversity. Output shape matches what RAG pipelines expect. |
| 🔌 **MCP native** | Thin stdio wrapper. Connect Claude Desktop, Cursor, or Pi in 30 seconds — no translation layer. |
| 🐍 **Python SDK** | `pip install agent-web-search`. Type-safe sync + async client, Pydantic v2, same vibe as the OpenAI SDK. [Repo →](https://github.com/blueewhitee/agent-web-search-sdk) |
| 🎬 **JS rendering (opt-in)** | `render_js: true` triggers crawl4ai for JavaScript-heavy pages. Not bundled by default; `pip install crawl4ai` when you need it. |
| 🐳 **Docker all-in-one** | `docker compose up` brings up SearXNG, the FastAPI, and the embedding model. No GPU required. |

## API Reference

### `POST /search`

| Param | Type | Default | Description |
|---|---|---|---|
| `query` | `string` | *required* | 1–1000 character search query |
| `include_content` | `bool` | `false` | Fetch, extract, embed, and rank page content |
| `render_js` | `bool` | `false` | JS rendering via crawl4ai (`pip install crawl4ai`) |
| `categories` | `string` | auto-detected | Override intent routing: `"it"`, `"news"`, `"general"` |
| `time_range` | `string` | auto-detected | `"day"`, `"week"`, `"month"`, `"year"` (used with `news`) |

**Response** (`include_content: true`):

```json
{
  "query": "python asyncio tutorial",
  "ranked_chunks": [
    {
      "text": "256-token chunk for embedding models",
      "parent_text": "512-token window for LLM context",
      "score": 0.877,
      "source": {
        "url": "https://docs.python.org/3/library/asyncio.html",
        "title": "asyncio — Asynchronous I/O",
        "searxng_score": 1.0
      }
    }
  ],
  "unresponsive_engines": []
}
```

**Response** (`include_content: false`):

```json
{
  "query": "python asyncio",
  "results": [
    { "url": "https://docs.python.org/3/library/asyncio.html", "title": "asyncio — Asynchronous I/O", "score": 1.0 }
  ]
}
```

### `GET /health`

```json
{ "status": "ok" }
```

<!-- A reproducible proof shot: terminal screenshot of the curl command + JSON above. -->
> 💡 Drop a terminal screenshot of the `curl` request and its ranked JSON response at `docs/curl-demo.png` and it renders right here.

## MCP Integration

The MCP server exposes `web_search` as a tool for any MCP-compatible client. It is a **thin stdio wrapper** — all business logic lives in the FastAPI. No duplicate logic, no drift. Results return as formatted markdown with cosine similarity scores, source URLs, and parent context windows.

**Pi** (project-local `.mcp.json`):
```json
{
  "mcpServers": {
    "nature-search": {
      "command": "uv",
      "args": ["run", "python", "-m", "mcp_server.server"],
      "env": { "NATURE_SE_API_URL": "http://localhost:8000" },
      "directTools": true
    }
  }
}
```

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "nature-search": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/agent-web-search", "python", "-m", "mcp_server.server"],
      "env": { "NATURE_SE_API_URL": "http://localhost:8000" }
    }
  }
}
```

**Cursor** (`.cursor/mcp.json`): Same format as Claude Desktop.

## Python SDK

The `agent-web-search` PyPI package gives you a type-safe Python client with sync and async surfaces — same look as the OpenAI SDK.

```bash
pip install agent-web-search
```

```python
from agent_web_search import AgentWebSearch, AsyncAgentWebSearch

# Sync
with AgentWebSearch(base_url="http://localhost:8000") as client:
    response = client.search("python asyncio", include_content=True)
    for chunk in response.ranked_chunks:
        print(f"{chunk.score:.3f}  {chunk.source.title}")

# Async (great for batching multiple searches)
import asyncio
async with AsyncAgentWebSearch() as client:
    results = await asyncio.gather(
        client.search("python asyncio"),
        client.search("rust borrow checker"),
        client.search("docker healthcheck"),
    )
```

The SDK is a **thin wrapper** — no pipeline logic, no duplicate code. It calls the same HTTP API you'd `curl`. Full docs, tests, and source: [agent-web-search-sdk](https://github.com/blueewhitee/agent-web-search-sdk).

## SearXNG Configuration

The bundled `searxng/config/settings.yml` is pre-tuned for reliability: engines that cause 30s timeouts (Google, Bing, Yahoo, Startpage, Brave) are **disabled**. The active engine lineup:

| Category | Engines |
|---|---|
| `general` | DuckDuckGo, Mojeek, Qwant |
| `it` | StackOverflow, GitHub, MDN, AskUbuntu, SuperUser, Docker Hub |
| `news` | Bing News, Reuters |
| `science` | Wikipedia, arXiv |
| `files` | GitHub, Docker Hub |

Plus sanctioned APIs: Wikipedia, GitHub, StackOverflow, MDN, ArXiv, Docker Hub. Edit `searxng/config/settings.yml` to add your own engines.

## Configuration

Environment variables (or `.env`):

| Env | Default | Description |
|---|---|---|
| `SEARXNG_BASE_URL` | `http://localhost:8080` | SearXNG JSON API endpoint |
| `TOP_K_FETCH` | `5` | URLs to fetch per query |
| `TOP_K_RETURN` | `3` | Chunks to return |
| `PER_URL_TIMEOUT` | `6` | Seconds per URL fetch |
| `BATCH_DEADLINE` | `8` | Total fetch deadline |

## Development

```bash
uv sync
uv run uvicorn main:app --reload
uv run pytest    # 175 tests, 0 failures
```

Requires SearXNG on port 8080:

```bash
docker compose up searxng
```

Project structure:

```
app/
  api/search.py              # FastAPI router
  services/
    searxng_service.py        # SearXNG JSON API client + intent routing
    fetch_service.py            # Concurrent page fetcher
    extraction_service.py       # trafilatura + readability-lxml + density filter
    scrub_service.py            # Prompt-injection detection
    chunk_service.py            # Chonkie recursive chunker
    embed_service.py            # bge-small-en-v1.5 embeddings
    rank_service.py             # Cosine similarity + diversity
    intent_service.py           # Category routing engine (codes/news/general)
mcp_server/
  server.py                    # MCP stdio wrapper (thin, no business logic)
tests/
  test_intent_service.py        # 36 tests for intent routing
  test_search_router.py
  test_extraction_service.py
  ...                          # 140 tests total
```

## FAQ

**Is this a Tavily replacement?** — Yes, output shape matches what RAG pipelines expect, self-hosted and free. See the comparison table above.

**Do I need API keys?** — No. SearXNG + the local embedding model run entirely on your machine. API-keyed providers are optional.

**Does it work offline / air-gapped?** — The stack runs locally, but SearXNG still queries live external search engines, so it needs outbound internet to those engines. The model and API run without external services.

**Why are Google / Bing / Brave disabled?** — Those engines cause 30s timeouts in SearXNG's default config. We keep DuckDuckGo, Mojeek, Qwant plus sanctioned APIs (Wikipedia, GitHub, StackOverflow, MDN, arXiv) that return reliably. Re-enable any engine in `searxng/config/settings.yml`.

**Does it support JavaScript-heavy pages?** — Yes, opt-in via `render_js: true`, which uses crawl4ai. Not bundled by default: `pip install crawl4ai`.

**How is this different from just running SearXNG?** — SearXNG gives you URLs. Agent Web Search fetches each page, extracts clean text, scrubs prompt injections, chunks by semantic boundaries, embeds with `bge-small-en-v1.5`, and cosine-ranks chunks against your query — the exact preprocessing an AI agent needs.

## Contributing

PRs are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full process.

Quick version:

1. Fork → `git checkout -b feat/your-feature`
2. `uv sync` then `uv run pytest` — keep all 140 tests green
3. Open a PR describing what changed and why

We respond to PRs within 48 hours.

<div align="center">

[![Star History](https://api.star-history.com/svg?repos=blueewhitee/agent-web-search&type=Date)](https://star-history.com/#blueewhitee/agent-web-search&Date)

</div>

## License

[MIT](LICENSE) — use it, fork it, ship it.
