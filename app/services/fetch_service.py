"""Concurrent page fetch with dual deadlines (D-001, D-006, D-012).

Per-URL deadline (settings.per_url_timeout) and a whole-fan-out deadline
(settings.batch_deadline). Any failure — robots-disallowed, timeout,
HTTP error, or batch-deadline cancellation — degrades to a
`low_confidence=True` FetchResult rather than raising. The router merges
these flags onto SearchResults; the API never 504s on a fetch failure.

D-026: page fetching uses curl_cffi (AsyncSession, impersonate='chrome')
instead of httpx. Wikipedia/StackOverflow/MDN and other major content sites
anti-bot on the TLS fingerprint (JA3 hash of the TLS handshake), NOT the
User-Agent string — Python's `ssl` produces a robotic handshake that httpx
inherits. curl_cffi uses libcurl with Chrome's exact TLS fingerprint, so
requests pass the same anti-bot checks a real Chrome browser does. This is
strictly MORE honest than spoofing a UA string: the TLS layer is genuinely
Chrome's, not a lie in a header. httpx is retained for SearXNG and
robots.txt (local/owned services, no anti-bot) but those don't flow
through this service's page-fetch path.
"""

import asyncio

import httpx  # retained for SearXNG/robots clients; not used for page fetch

from app.core.config import Settings
from app.schemas.fetch import FetchResult
from app.services.robots_cache import RobotsCache


class FetchService:
    def __init__(
        self,
        client: httpx.AsyncClient,
        robots: RobotsCache,
        settings: Settings,
        curl_session: "object | None" = None,
    ) -> None:
        # `client` is the shared httpx client (kept for robots.txt fetches via
        # the RobotsCache; the page-fetch path uses curl_cffi instead).
        self._client = client
        self._robots = robots
        self._settings = settings
        # curl_cffi AsyncSession for page fetches (D-026). Lazy-init on first
        # use so the import + session creation don't slow app startup, and so
        # tests that monkeypatch fetch_one don't require curl_cffi installed.
        self._curl_session = curl_session
        self._curl_initialized = curl_session is not None

    async def _get_curl(self):
        """Lazy-create the curl_cffi AsyncSession on first page fetch.

        impersonate='chrome' is what makes major sites (Wikipedia, SO, MDN)
        return 200 instead of 403 — it mimics Chrome's exact TLS handshake.
        """
        if not self._curl_initialized:
            from curl_cffi import AsyncSession
            self._curl_session = AsyncSession(impersonate='chrome')
            self._curl_initialized = True
        return self._curl_session

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
                # crawl4ai missing/broken → fall back to curl_cffi; honest
                # about the fallback in `method` (set below).
                pass

        # 3. curl_cffi fetch with per-URL deadline (D-026)
        try:
            session = await self._get_curl()
            r = await asyncio.wait_for(
                session.get(
                    url,
                    headers={"User-Agent": self._settings.user_agent},
                    allow_redirects=True,
                ),
                timeout=self._settings.per_url_timeout,
            )
            # curl_cffi Response has .status_code and .text like httpx.
            if r.status_code >= 400:
                return FetchResult(url=url, low_confidence=True, method="curl_cffi-failed")
            return FetchResult(url=url, html=r.text, status=r.status_code, method="curl_cffi")
        except asyncio.TimeoutError:
            return FetchResult(url=url, low_confidence=True, method="curl_cffi-failed")
        except Exception:
            # curl_cffi errors (CurlError, etc.) — degrade, never raise.
            return FetchResult(url=url, low_confidence=True, method="curl_cffi-failed")

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
