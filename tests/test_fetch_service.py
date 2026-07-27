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


class _MockCurlSession:
    """Mimics curl_cffi.AsyncSession for tests — no network, no curl_cffi dep.

    Production fetch_one uses curl_cffi (D-026); tests inject this mock via
    the `curl_session` param so the graceful-degradation contract is tested
    without curl_cffi installed. Returns httpx.Response objects, which expose
    the same .status_code / .text attributes curl_cffi responses do, so the
    production code paths (r.status_code >= 400, r.text) work unchanged.
    """

    def __init__(self, handler):
        self._handler = handler

    async def get(self, url, headers=None, allow_redirects=True):
        request = httpx.Request("GET", url, headers=headers or {})
        return await self._handler(request)

    async def close(self):
        pass


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
        mock_curl = _MockCurlSession(_handler)
        yield FetchService(
            client=client, robots=robots, settings=settings, curl_session=mock_curl
        )


class TestFetchOne:
    @pytest.mark.asyncio
    async def test_successful_fetch(self, fetch_service):
        r = await fetch_service.fetch_one("https://ok.test/x")
        assert r.low_confidence is False
        assert r.html == "<html>ok</html>"
        assert r.method == "curl_cffi"
        assert r.status == 200

    @pytest.mark.asyncio
    async def test_404_low_confidence(self, fetch_service):
        r = await fetch_service.fetch_one("https://notfound.test/x")
        assert r.low_confidence is True
        assert r.method == "curl_cffi-failed"
        assert r.html is None

    @pytest.mark.asyncio
    async def test_timeout_low_confidence(self, fetch_service):
        r = await fetch_service.fetch_one("https://slow.test/x")
        assert r.low_confidence is True
        assert r.method == "curl_cffi-failed"

    @pytest.mark.asyncio
    async def test_connection_error_low_confidence(self, fetch_service):
        # The mock handler raises httpx.ConnectError; the production code catches
        # broad Exception → degrades to low_confidence. Verify the contract.
        r = await fetch_service.fetch_one("https://drop.test/x")
        assert r.low_confidence is True
        assert r.method == "curl_cffi-failed"

    @pytest.mark.asyncio
    async def test_render_js_falls_back_to_curl_cffi(self, fetch_service):
        # crawl4ai is not installed -> ImportError -> curl_cffi fallback.
        r = await fetch_service.fetch_one("https://ok.test/x", render_js=True)
        assert r.low_confidence is False
        assert r.method == "curl_cffi"
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
        assert results[1].method == "curl_cffi-failed" or results[1].method == "batch-timeout"

    @pytest.mark.asyncio
    async def test_empty_urls(self, fetch_service):
        assert await fetch_service.fetch_many([]) == []
