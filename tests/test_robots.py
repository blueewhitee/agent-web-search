"""Unit tests for robots.txt parsing (D-012).

Pure parsing tests — no I/O. Uses urllib.robotparser directly with
fixture robots.txt strings. Encodes the RFC 9309 specific-group
precedence rule and the UA version-stripping behavior (the Phase 3
lesson): robots.txt rules use the unversioned product name, and the
queried UA's version is stripped via split('/')[0] internally.
"""

from urllib.robotparser import RobotFileParser


def _parse(robots_text: str) -> RobotFileParser:
    parser = RobotFileParser()
    parser.parse(robots_text.splitlines())
    return parser


ROBOTS_FIXTURE = """User-agent: NatureSE-Bot
Disallow: /private

User-agent: *
Disallow: /search
"""


class TestRobotsParsing:
    def test_specific_group_allows_public(self):
        p = _parse(ROBOTS_FIXTURE)
        # Queried UA is versioned; parser strips via split('/')[0] -> "naturese-bot"
        assert p.can_fetch("NatureSE-Bot/0.1 (search-research)", "https://site.test/public") is True

    def test_specific_group_blocks_private(self):
        p = _parse(ROBOTS_FIXTURE)
        assert p.can_fetch("NatureSE-Bot/0.1", "https://site.test/private") is False

    def test_specific_group_precedence_ignores_default_group(self):
        # RFC 9309: once a specific user-agent group matches, the * default
        # group is NOT consulted. So /search (a default-group rule) is allowed
        # for NatureSE-Bot even though it's blocked for *.
        p = _parse(ROBOTS_FIXTURE)
        assert p.can_fetch("NatureSE-Bot/0.1", "https://site.test/search") is True

    def test_default_group_fallback_blocks_search(self):
        p = _parse(ROBOTS_FIXTURE)
        # Generic UA -> falls through to * group -> /search blocked
        assert p.can_fetch("GenericCrawler/2.0", "https://site.test/search") is False

    def test_default_group_allows_unruled_path(self):
        p = _parse(ROBOTS_FIXTURE)
        assert p.can_fetch("GenericCrawler/2.0", "https://site.test/public") is True

    def test_empty_robots_allows_all(self):
        # Fail-open equivalent: no rules -> allow everything.
        p = _parse("")
        assert p.can_fetch("AnyBot/1.0", "https://site.test/anything") is True

    def test_disallow_root_blocks_all(self):
        p = _parse("User-agent: *\nDisallow: /\n")
        assert p.can_fetch("AnyBot/1.0", "https://site.test/anything") is False

    def test_allow_overrides_disallow(self):
        # Explicit Allow for a path wins over a broader Disallow.
        robots = """User-agent: *
Disallow: /private
Allow: /public
"""
        p = _parse(robots)
        assert p.can_fetch("AnyBot/1.0", "https://site.test/public") is True
        assert p.can_fetch("AnyBot/1.0", "https://site.test/private") is False
