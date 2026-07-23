"""Concurrent page fetch with dual deadlines (D-001, D-006, D-012).

Per-URL deadline (settings.per_url_timeout) and a whole-fan-out deadline
(settings.batch_deadline). Any failure — robots-disallowed, timeout,
HTTP error, or batch-deadline cancellation — degrades to a
`low_confidence=True` FetchResult rather than raising. The router merges
these flags onto SearchResults; the API never 504s on a fetch failure.
"""

import asyncio

import httpx

from app.core.config import Settings
from app.schemas.fetch import FetchResult
from app.services.robots_cache import RobotsCache


class FetchService:
    def __init__(
        self,
        client: httpx.AsyncClient,
        robots: RobotsCache,
        settings: Settings,
    ) -> None:
        self._client = client
        self._robots = robots
        self._settings = settings

    async def fetch_one(self, url: str, render_js: bool = False) -> FetchResult:
        """Fetch a single URL. Never raises — failures return low_confidence."""
        # 1. Robots check (fail-open upstream, but explicit disallow → skip)
        if not await self._robots.is_allowed(url):
            return FetchResult(url=url, low_confidence=True, method="robots-blocked")

        # 2. crawl4ai opt-in (D-006)
        if render_js:
            try:
                from app.services._crawl4ai_adapter import fetch_with_crawl4ai

                html = await fetch_with_crawl4ai(url, self._settings.user_agent)
                return FetchResult(url=url, html=html, status=200, method="crawl4ai")
            except Exception:
                # crawl4ai missing/broken → fall back to httpx; honest about
                # the fallback in `method` (httpx path sets it below).
                pass

        # 3. httpx fetch with per-URL deadline
        try:
            r = await asyncio.wait_for(
                self._client.get(
                    url,
                    headers={"User-Agent": self._settings.user_agent},
                    follow_redirects=True,
                ),
                timeout=self._settings.per_url_timeout,
            )
            r.raise_for_status()
            return FetchResult(url=url, html=r.text, status=r.status_code, method="httpx")
        except asyncio.TimeoutError:
            return FetchResult(url=url, low_confidence=True, method="httpx-failed")
        except httpx.HTTPError:
            return FetchResult(url=url, low_confidence=True, method="httpx-failed")

    async def fetch_many(
        self, urls: list[str], render_js: bool = False
    ) -> list[FetchResult]:
        """Fetch many URLs concurrently within the batch deadline.

        Output is aligned with input order. Tasks still pending when the
        batch deadline fires are cancelled and reported as
        `method="batch-timeout"`, `low_confidence=True`.
        """
        if not urls:
            return []
        tasks = [asyncio.create_task(self.fetch_one(u, render_js)) for u in urls]
        done, pending = await asyncio.wait(
            tasks, timeout=self._settings.batch_deadline
        )
        for t in pending:
            t.cancel()

        # Align output with input order
        results: list[FetchResult] = []
        for t, url in zip(tasks, urls):
            if t in done and not t.cancelled() and not t.exception():
                results.append(t.result())
            else:
                results.append(
                    FetchResult(url=url, low_confidence=True, method="batch-timeout")
                )
        return results
