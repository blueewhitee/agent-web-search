"""Unit tests for query-intent routing (intent_service.py).

Pure-function tests — no network, no I/O.
"""

import pytest

from app.services.intent_service import (
    detect,
    IntentParams,
    _is_mdn_topical,
    filter_mdn_results,
)


class TestDetectNews:
    def test_explicit_news_keyword(self):
        p = detect("latest news on OpenAI")
        assert p.categories == ("news",)
        assert p.time_range == "week"

    def test_today_is_day_range(self):
        p = detect("OpenAI crash today")
        assert p.categories == ("news",) and p.time_range == "day"

    def test_tonight_is_day_range(self):
        p = detect("election results tonight")
        assert p.time_range == "day"

    def test_yesterday_is_day_range(self):
        p = detect("what happened yesterday")
        assert p.time_range == "day"

    def test_breaking_is_day_range(self):
        p = detect("breaking news about the protest")
        assert p.time_range == "day"

    def test_this_month_is_month_range(self):
        p = detect("AI regulation news this month")
        assert p.time_range == "month"

    def test_announced_triggers_news(self):
        p = detect("OpenAI announced a new model")
        assert p.categories == ("news",)

    def test_released_triggers_news(self):
        p = detect("Apple released new iPhone")
        assert p.categories == ("news",)

    def test_scandal_triggers_news(self):
        p = detect("political scandal latest")
        assert p.categories == ("news",)

    def test_protest_triggers_news(self):
        p = detect("protests erupt in Delhi")
        assert p.categories == ("news",)

    def test_outage_triggers_news(self):
        p = detect("ChatGPT outage latest updates")
        assert p.categories == ("news",)

    def test_default_time_range_is_week(self):
        p = detect("news about AI")
        assert p.time_range == "week"


class TestDetectCode:
    def test_python_query(self):
        p = detect("python asyncio gather example")
        assert p.categories == ("it",)
        assert p.time_range is None

    def test_error_string(self):
        p = detect("AttributeError 'NoneType' object has no attribute 'split'")
        assert p.categories == ("it",)

    def test_file_extension_tsconfig(self):
        p = detect("configure tsconfig.json strict mode")
        assert p.categories == ("it",)

    def test_file_extension_py(self):
        p = detect("open main.py and edit the class")
        assert p.categories == ("it",)

    def test_file_extension_rs(self):
        p = detect("how do I compile lib.rs")
        assert p.categories == ("it",)

    def test_docker(self):
        p = detect("docker compose restart searxng")
        assert p.categories == ("it",)

    def test_kubernetes(self):
        p = detect("kubernetes pod crashloopbackoff")
        assert p.categories == ("it",)

    def test_how_to(self):
        p = detect("how to install nginx on ubuntu")
        assert p.categories == ("it",)

    def test_npm(self):
        p = detect("npm install react-hook-form")
        assert p.categories == ("it",)

    def test_git(self):
        p = detect("git rebase interactive squash")
        assert p.categories == ("it",)

    def test_traceback_error(self):
        p = detect("Traceback (most recent call last): File app.py line 42")
        assert p.categories == ("it",)

    def test_compiler_error(self):
        p = detect("compiler error expected type string but found int")
        assert p.categories == ("it",)

    def test_code_language_names(self):
        for lang in ("javascript", "typescript", "rust", "go", "golang",
                     "java", "kotlin", "swift", "ruby", "php", "bash",
                     "powershell", "lua", "scala", "perl"):
            p = detect(f"{lang} tutorial")
            assert p.categories == ("it",), f"'{lang}' should trigger code intent"

    def test_library_names(self):
        for lib in ("react", "vue", "angular", "django", "flask",
                    "fastapi", "tailwind"):
            p = detect(f"{lib} component example")
            assert p.categories == ("it",), f"'{lib}' should trigger code intent"


class TestDetectGeneral:
    def test_capital_of_france(self):
        p = detect("capital of France")
        assert p.categories == ("general",)
        assert p.time_range is None

    def test_cockroach_janta_party(self):
        p = detect("Cockroach Janta Party")
        assert p.categories == ("general",)
        assert p.time_range is None

    def test_openai(self):
        p = detect("OpenAI")
        assert p.categories == ("general",)
        assert p.time_range is None

    def test_plain_wikipedia_question(self):
        p = detect("what is photosynthesis")
        assert p.categories == ("general",)


class TestPrecedence:
    def test_code_wins_over_news(self):
        """Both signals present — code takes precedence (docs over news)."""
        p = detect("python 3.13 released announcement")
        assert p.categories == ("it",)

    def test_js_error_news(self):
        # "error" is code, "news" doesn't trigger here
        p = detect("react error boundary tutorial")
        assert p.categories == ("it",)


class TestIntentParams:
    def test_as_params_without_time_range(self):
        p = IntentParams(categories=("it",), time_range=None)
        assert p.as_params() == {"categories": "it"}

    def test_as_params_with_time_range(self):
        p = IntentParams(categories=("news",), time_range="week")
        assert p.as_params() == {"categories": "news", "time_range": "week"}

    def test_as_params_multiple_categories(self):
        p = IntentParams(categories=("news", "general"), time_range="day")
        assert p.as_params() == {
            "categories": "news,general",
            "time_range": "day",
        }

    def test_intent_params_is_frozen(self):
        p = IntentParams(categories=("it",), time_range=None)
        with pytest.raises(Exception):
            p.categories = ("news",)  # type: ignore[misc]


# ── MDN topical filter (D-026) ─────────────────────────────────────────────
# Backed by the Run-2 eval finding: ~29% of all local results were MDN
# pages unrelated to the query ("rust borrow checker" -> "Using an external
# spell checker"; "python 3.13" -> "Firefox 13 release notes"). MDN's
# glossary indexes every concept, so it bleeds into non-web queries.


class _FakeResult:
    """Duck-typed on .url — same shape SearXNGService returns."""

    def __init__(self, url: str, title: str = "x") -> None:
        self.url = url
        self.title = title


class TestIsMdnTopical:
    def test_python_query_not_topical(self):
        assert _is_mdn_topical("what is a context manager in python") is False

    def test_rust_query_not_topical(self):
        assert _is_mdn_topical("rust borrow checker explained") is False

    def test_docker_query_not_topical(self):
        assert _is_mdn_topical("docker compose healthcheck example") is False

    def test_python_release_not_topical(self):
        assert _is_mdn_topical("python 3.13 release announcement") is False

    def test_general_knowledge_not_topical(self):
        assert _is_mdn_topical("what is the capital of Australia") is False
        assert _is_mdn_topical("how does photosynthesis work") is False

    def test_regex_is_topical(self):
        # CORRECT MDN hit lives here — must NOT be filtered out
        assert _is_mdn_topical("regex negative lookahead syntax") is True

    def test_regexp_keyword_topical(self):
        assert _is_mdn_topical("grep vs regexp difference") is True

    def test_html_topical(self):
        assert _is_mdn_topical("html form submit input types") is True

    def test_css_topical(self):
        assert _is_mdn_topical("css flexbox vs grid which to use") is True

    def test_javascript_topical(self):
        assert _is_mdn_topical("javascript fetch vs axios") is True

    def test_js_word_boundary_required(self):
        # "js" inside another word must NOT trigger allow
        assert _is_mdn_topical("objects destructuring in python") is False

    def test_json_topical(self):
        assert _is_mdn_topical("parse json in javascript") is True

    def test_canonical_web_api_topical(self):
        assert _is_mdn_topical("how to use websocket") is True
        assert _is_mdn_topical("css grid layout tutorial") is True

    def test_ambiguous_fetch_NOT_topical(self):
        # "fetch" alone is ambiguous (HTTP fetch API vs the verb); we
        # deliberately exclude it from the allow-list to avoid over-allowing.
        # Query here avoids other web keywords (no "url"/"json"/etc.) so only
        # "fetch" would have triggered — which it must NOT.
        assert _is_mdn_topical("how to fetch remote data in python requests") is False

    def test_empty_query_not_topical(self):
        assert _is_mdn_topical("") is False


class TestFilterMdnResults:
    MDN1 = "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Glossary/Python"
    MDN2 = "https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Releases/13"
    SO1 = "https://stackoverflow.com/questions/123/python-context-manager"
    SU1 = "https://superuser.com/questions/456/login-data"

    def test_drops_mdn_for_python_query(self):
        out = filter_mdn_results(
            [_FakeResult(self.MDN1), _FakeResult(self.SO1), _FakeResult(self.SU1)],
            "what is a context manager in python",
        )
        urls = [r.url for r in out]
        assert self.MDN1 not in urls
        assert self.SO1 in urls
        assert self.SU1 in urls

    def test_keeps_mdn_for_regex_query(self):
        # The known-good MDN hit on regex assertions must survive
        mdn_regex = "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Regular_Expressions/Lookahead"
        so = "https://stackoverflow.com/questions/regex"
        out = filter_mdn_results(
            [_FakeResult(mdn_regex), _FakeResult(so)],
            "regex negative lookahead syntax",
        )
        urls = [r.url for r in out]
        assert mdn_regex in urls
        assert so in urls

    def test_preserves_order(self):
        mdn = "https://developer.mozilla.org/x"
        a = "https://github.com/a"
        b = "https://gitlab.com/b"
        out = filter_mdn_results(
            [_FakeResult(a), _FakeResult(mdn), _FakeResult(b)],
            "python typing generic class",
        )
        assert [r.url for r in out] == [a, b]

    def test_all_mdn_dropped_yields_empty(self):
        out = filter_mdn_results(
            [_FakeResult(self.MDN1), _FakeResult(self.MDN2)],
            "python 3.13 release announcement",
        )
        assert out == []

    def test_empty_input_returns_empty(self):
        assert filter_mdn_results([], "anything") == []

    def test_no_mdn_returns_unchanged(self):
        rs = [_FakeResult(self.SO1), _FakeResult(self.SU1)]
        out = filter_mdn_results(rs, "what is a context manager in python")
        assert [r.url for r in out] == [self.SO1, self.SU1]

    def test_mdn_dev_host_also_filtered(self):
        # alternate MDN host substring
        rs = [_FakeResult("https://mdn.dev/foo"), _FakeResult(self.SO1)]
        out = filter_mdn_results(rs, "docker compose healthcheck example")
        assert [r.url for r in out] == [self.SO1]

    def test_case_insensitive_url_match(self):
        # Hosts come through with mixed case from SearXNG
        rs = [_FakeResult("https://Developer.Mozilla.org/x"), _FakeResult(self.SO1)]
        out = filter_mdn_results(rs, "rust trait")
        assert [r.url for r in out] == [self.SO1]

