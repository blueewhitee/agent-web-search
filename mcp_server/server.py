"""MCP server for Nature-based Search API.

Thin stdio wrapper — delegates all work to the FastAPI. No business logic.
Connect any MCP-compatible agent (Claude Desktop, Cursor, Pi, etc.) to get
web search as a tool.

Usage:
    uv run python -m mcp_server.server          # stdio (default)
    mcp dev mcp_server/server.py                # inspector (test in browser)
    NATURE_SE_API_URL=http://localhost:8000 uv run python -m mcp_server.server
"""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

# ── Server ────────────────────────────────────────────────────

mcp = FastMCP("nature_search_mcp")
API_URL = os.getenv("NATURE_SE_API_URL", "http://localhost:8000")

# ── Input schema ──────────────────────────────────────────────


class SearchInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description=(
            "Search query. IMPORTANT (D-009): if the user used vague words "
            "like 'it', 'this', or 'that', resolve them to specific terms "
            "using the conversation context BEFORE calling this tool."
        ),
    )
    render_js: bool = Field(
        default=False,
        description=(
            "Use browser rendering for JavaScript-heavy pages. "
            "Slower; only for pages that require JavaScript to load content."
        ),
    )


# ── Tool ──────────────────────────────────────────────────────


@mcp.tool(
    name="web_search",
    annotations={
        "title": "Web Search",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
)
async def web_search(params: SearchInput) -> str:
    """Search the web and return ranked, content-rich results.

    Queries multiple search engines, fetches full page content, extracts clean
    text, scrubs prompt injections, chunks by semantic boundaries, and ranks
    chunks by cosine similarity to the query.

    Use this when the user asks about current information, facts,
    documentation, or anything that requires up-to-date web results.

    Returns markdown-formatted ranked chunks with cosine similarity scores,
    source URLs, and surrounding parent context for each chunk.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{API_URL}/search",
                json={
                    "query": params.query,
                    "include_content": True,
                    "render_js": params.render_js,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError:
        return (
            f"Error: Search API unreachable at {API_URL}. "
            "Start it with: docker compose up"
        )
    except httpx.HTTPStatusError as e:
        detail = ""
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text[:200]
        return (
            f"Error: Search API returned HTTP {e.response.status_code}\n"
            f"Details: {detail}"
        )
    except httpx.TimeoutException:
        return (
            "Error: Search API request timed out (>30s). "
            "The upstream search engines may be slow — try again."
        )

    chunks = data.get("ranked_chunks", [])
    query = data.get("query", params.query)
    unresponsive = data.get("unresponsive_engines", [])

    if not chunks:
        note = ""
        if unresponsive:
            note = f"\n\n⚠️ Unresponsive search engines: {', '.join(unresponsive)}"
        return f"No results returned for: {query}{note}"

    lines = [f"## Web Search Results: {query}", ""]

    for i, c in enumerate(chunks, 1):
        src = c["source"]
        lines.append(f"### {i}. [score: {c['score']:.2f}] {src['title']}")
        lines.append(f"**Source:** {src['url']}")
        lines.append("")
        lines.append(f"> {c['text']}")
        lines.append("")
        lines.append("**Parent context:**")
        lines.append(c["parent_text"])
        lines.append("")
        lines.append("---")
        lines.append("")

    if unresponsive:
        lines.append(f"⚠️ Unresponsive search engines: {', '.join(unresponsive)}")

    return "\n".join(lines)


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
