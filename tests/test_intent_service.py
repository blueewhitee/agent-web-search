"""Unit tests for query-intent routing (intent_service.py).

Pure-function tests — no network, no I/O.
"""

import pytest

from app.services.intent_service import detect, IntentParams


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
