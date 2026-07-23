"""Per-host robots.txt cache (D-012).

Fetches and caches robots.txt per host for the lifetime of the process
(no TTL in v1 — acceptable for a search API that runs briefly per query).
Fail-open on fetch error: an unreachable robots.txt must not kill the
whole batch, so a bad host degrades to "allow all".
"""

import urllib.parse
from urllib.robotparser import RobotFileParser

import httpx


class RobotsCache:
    def __init__(
        self,
        client: httpx.AsyncClient,
        user_agent: str,
        timeout: float = 3.0,
    ) -> None:
        self._client = client
        self._user_agent = user_agent
        self._timeout = timeout
        self._cache: dict[str, RobotFileParser] = {}

    async def is_allowed(self, url: str) -> bool:
        """Return True if `user_agent` may fetch `url` per robots.txt.

        First access for a host triggers a one-time fetch; subsequent
        accesses hit the in-memory cache. Unreachable/errored robots.txt
        → allow all (fail-open).
        """
        parsed = urllib.parse.urlsplit(url)
        host = parsed.netloc
        if host not in self._cache:
            await self._load(host, parsed.scheme or "https")
        return self._cache[host].can_fetch(self._user_agent, url)

    async def _load(self, host: str, scheme: str) -> None:
        robots_url = f"{scheme}://{host}/robots.txt"
        try:
            r = await self._client.get(
                robots_url, timeout=self._timeout, follow_redirects=True
            )
            parser = RobotFileParser()
            # Non-200 → treat as empty (no rules = allow all).
            parser.parse(r.text.splitlines() if r.status_code == 200 else [])
            self._cache[host] = parser
        except httpx.HTTPError:
            # Fail-open: unreachable robots.txt → allow all. Don't let one
            # bad host kill the batch.
            parser = RobotFileParser()
            parser.parse([])
            self._cache[host] = parser
