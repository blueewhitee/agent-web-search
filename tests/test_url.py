"""Unit tests for URL normalization + dedup (D-004). Pure, no I/O."""

from app.schemas.search import SearchResult
from app.utils.url import dedup_results, normalize_url


def _n(url: str) -> str:
    """Helper: accept bare hostnames by prepending https."""
    if "://" not in url:
        url = "https://" + url
    return normalize_url(url)


class TestNormalizeUrl:
    def test_mobile_subdomain_collapses(self):
        # en.wikipedia.org/wiki/Foo <-> en.m.wikipedia.org/wiki/Foo -> same
        assert _n("en.wikipedia.org/wiki/Foo") == _n(
            "https://en.m.wikipedia.org/wiki/Foo"
        )

    def test_trailing_slash_collapses(self):
        # example.com/foo <-> example.com/foo/ -> same
        assert _n("https://example.com/foo") == _n("https://example.com/foo/")

    def test_utm_param_stripped(self):
        # example.com/x?utm_source=twitter <-> example.com/x -> same
        assert _n("https://example.com/x") == _n(
            "https://example.com/x?utm_source=twitter"
        )

    def test_tracking_exact_params_stripped(self):
        # gclid/fbclid/ref/source/_ga/mc_cid all dropped
        assert _n("https://example.com/x?gclid=abc&keep=1") == _n(
            "https://example.com/x?keep=1&fbclid=zzz"
        )

    def test_content_driving_param_q_differs(self):
        # example.com/?q=hello <-> example.com/?q=world -> DIFFERENT
        assert _n("https://example.com/?q=hello") != _n(
            "https://example.com/?q=world"
        )

    def test_content_driving_param_id_differs(self):
        # example.com/?id=123 <-> example.com/?id=456 -> DIFFERENT
        assert _n("https://example.com/?id=123") != _n(
            "https://example.com/?id=456"
        )

    def test_www_subdomain_stripped(self):
        assert _n("https://www.example.com/x") == _n("https://example.com/x")

    def test_bare_m_subdomain_stripped(self):
        assert _n("https://m.example.com/x") == _n("https://example.com/x")

    def test_scheme_lowercased(self):
        assert _n("HTTPS://Example.com/x") == _n("https://example.com/x")

    def test_fragment_dropped(self):
        assert _n("https://example.com/x#frag") == _n("https://example.com/x")

    def test_root_path_preserved(self):
        # Trailing slash on root '/' must not be stripped to empty
        assert _n("https://example.com/").endswith("/")


class TestDedupResults:
    def test_empty_list(self):
        assert dedup_results([]) == []

    def test_keeps_highest_score(self):
        # Two entries for same normalized URL; higher searxng_score wins.
        results = [
            SearchResult(url="https://example.com/foo", title="a", snippet="", searxng_score=1.0),
            SearchResult(url="https://example.com/foo/", title="b", snippet="", searxng_score=2.0),
            SearchResult(url="https://example.com/bar", title="c", snippet="", searxng_score=0.5),
        ]
        d = dedup_results(results)
        assert len(d) == 2
        winner = next(r for r in d if "foo" in r.url)
        assert winner.title == "b"

    def test_ties_keep_first_seen(self):
        # Equal scores: first-seen wins (stable order).
        results = [
            SearchResult(url="https://example.com/foo", title="first", snippet="", searxng_score=1.0),
            SearchResult(url="https://example.com/foo/", title="second", snippet="", searxng_score=1.0),
        ]
        d = dedup_results(results)
        assert len(d) == 1
        assert d[0].title == "first"

    def test_preserves_order_of_winners(self):
        results = [
            SearchResult(url="https://example.com/a", title="a", snippet="", searxng_score=1.0),
            SearchResult(url="https://example.com/b", title="b", snippet="", searxng_score=2.0),
            SearchResult(url="https://example.com/a/", title="a-dup", snippet="", searxng_score=3.0),
        ]
        d = dedup_results(results)
        # Winner for example.com/a should still appear before example.com/b
        # because original first-seen index of a (0) < b (1).
        assert [r.title for r in d] == ["a-dup", "b"]
