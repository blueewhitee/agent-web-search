import httpx

from app.schemas.search import SearchResult


class SearXNGService:
    """Thin async wrapper around a self-hosted SearXNG JSON endpoint.

    The service owns no state beyond the injected httpx client and base URL,
    so a single shared AsyncClient (pooled, created once in main.py lifespan)
    is reused across requests.
    """

    def __init__(
        self,
        base_url: str,
        client: httpx.AsyncClient,
        timeout: float = 8.0,
    ) -> None:
        self._base_url = base_url
        self._client = client
        self._timeout = timeout

    async def search(
        self,
        query: str,
        top_k: int = 5,
        categories: list[str] | tuple[str, ...] | None = None,
        time_range: str | None = None,
    ) -> tuple[list[SearchResult], list[str]]:
        """Run a query against SearXNG and return (normalized results, unresponsive engines).

        - Missing `results` key → empty list (SearXNG returned nothing; not an error).
        - Individual result missing `url` → skipped (SearXNG can return malformed rows).
        - HTTP error → let `raise_for_status()` bubble; the router maps it to 502.
        - Timeout → caller (router) catches `httpx.TimeoutException` → 504.
        - ``categories`` / ``time_range`` are optional; when both are None (default)
          behaviour is identical to today (SearXNG defaults to `general` category).
        """
        params: dict = {"q": query, "format": "json"}
        if categories:
            params["categories"] = ",".join(categories)
        if time_range:
            params["time_range"] = time_range
        response = await self._client.get(
            f"{self._base_url}/search", params=params, timeout=self._timeout
        )
        response.raise_for_status()
        data = response.json()

        results = [
            SearchResult(
                url=r["url"],
                title=r.get("title", ""),
                snippet=r.get("content", ""),
                searxng_score=float(r.get("score", 0.0)),
            )
            for r in data.get("results", [])[:top_k]
            if "url" in r
        ]

        # SearXNG returns unresponsive_engines as [name, error_msg] pairs.
        unresponsive = [
            name for name, _ in data.get("unresponsive_engines", [])
        ]
        return results, unresponsive
