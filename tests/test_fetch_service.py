"""Tests for FetchService (D-001, D-006, D-012).

MockTransport-based — no network. Covers the graceful-degradation
contract: every failure mode returns a low_confidence FetchResult
rather than raising. The robots check uses an empty robots.txt
(allow-all) so fetch behavior can be isolated.
"""

import asyncio

import httpx
import pytest
import pytest_asyncio

from app.core.config import Settings
from app.services.fetch_service import FetchService
from app.services.robots_cache import RobotsCache


class _AsyncMockTransport(httpx.AsyncBaseTransport):
    """Async transport whose handler can `await` — so asyncio.wait_for
    can genuinely interrupt slow responses (the Phase 4 lesson: a
    sync time.sleep() in a MockTransport blocks the event loop)."""

    def __init__(self, handler):
        self._handler = handler

    async def handle_async_request(self, request):
        return await self._handler(request)


ROBOTS_EMPTY = ""  # allow-all


async def _handler(request):
    # robots.txt requests -> empty (allow all)
    if request.url.path == "/robots.txt":
        return httpx.Response(200, text=ROBOTS_EMPTY)
    host = request.url.host
    if host == "ok.test":
        return httpx.Response(200, text="<html>ok</html>")
    if host == "notfound.test":
        return httpx.Response(404, text="nope")
    if host == "slow.test":
        await asyncio.sleep(5)
        return httpx.Response(200, text="never")
    if host == "drop.test":
        raise httpx.ConnectError("unreachable")
    return httpx.Response(200, text="generic")


@pytest.fixture
def settings():
    s = Settings()
    s.per_url_timeout = 0.3
    s.batch_deadline = 0.5
    return s


@pytest_asyncio.fixture
async def fetch_service(settings):
    transport = _AsyncMockTransport(_handler)
    async with httpx.AsyncClient(transport=transport) as client:
        robots = RobotsCache(client=client, user_agent=settings.user_agent, timeout=1.0)
        yield FetchService(client=client, robots=robots, settings=settings)


class TestFetchOne:
    @pytest.mark.asyncio
    async def test_successful_fetch(self, fetch_service):
        r = await fetch_service.fetch_one("https://ok.test/x")
        assert r.low_confidence is False
        assert r.html == "<html>ok</html>"
        assert r.method == "httpx"
        assert r.status == 200

    @pytest.mark.asyncio
    async def test_404_low_confidence(self, fetch_service):
        r = await fetch_service.fetch_one("https://notfound.test/x")
        assert r.low_confidence is True
        assert r.method == "httpx-failed"
        assert r.html is None

    @pytest.mark.asyncio
    async def test_timeout_low_confidence(self, fetch_service):
        r = await fetch_service.fetch_one("https://slow.test/x")
        assert r.low_confidence is True
        assert r.method == "httpx-failed"

    @pytest.mark.asyncio
    async def test_connection_error_low_confidence(self, fetch_service):
        r = await fetch_service.fetch_one("https://drop.test/x")
        assert r.low_confidence is True
        assert r.method == "httpx-failed"

    @pytest.mark.asyncio
    async def test_render_js_falls_back_to_httpx(self, fetch_service):
        # crawl4ai is not installed -> ImportError -> httpx fallback.
        r = await fetch_service.fetch_one("https://ok.test/x", render_js=True)
        assert r.low_confidence is False
        assert r.method == "httpx"
        assert r.html == "<html>ok</html>"


class TestFetchMany:
    @pytest.mark.asyncio
    async def test_batch_deadline_cancels_slow(self, fetch_service, settings):
        settings.batch_deadline = 0.4
        results = await fetch_service.fetch_many(
            ["https://ok.test/x", "https://slow.test/x"]
        )
        assert len(results) == 2
        # Order preserved
        assert results[0].url == "https://ok.test/x"
        assert results[0].low_confidence is False
        assert results[1].url == "https://slow.test/x"
        assert results[1].low_confidence is True
        assert results[1].method == "httpx-failed" or results[1].method == "batch-timeout"

    @pytest.mark.asyncio
    async def test_empty_urls(self, fetch_service):
        assert await fetch_service.fetch_many([]) == []
