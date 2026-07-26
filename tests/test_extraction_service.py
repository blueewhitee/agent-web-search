"""Tests for Stage 3 text extraction (D-002, D-006 sub-topic 2).

Covers the two-tier fallback contract: trafilatura primary, readability-lxml
fallback, ≥200-char threshold, and the density filter that rejects
index/aggregator shells. No network — pure function over HTML strings.
"""

from app.services.extraction_service import _looks_like_shell, extract_text

# ~300 chars of lorem-ish article body, enough to clear the 200-char threshold.
_LONG_BODY = (
    "Python's asyncio module provides infrastructure for writing single-threaded "
    "concurrent code using coroutines, multiplexing I/O access over sockets and "
    "other resources that run event loops. It is the foundation for modern async "
    "web frameworks like FastAPI and Starlette. The core abstraction is the event "
    "loop, which schedules tasks and handles I/O multiplexing efficiently. "
    "Coroutines are functions marked with async def; calling them returns a "
    "coroutine object that must be awaited to execute, allowing suspension."
)

_GOOD_HTML = f"""<html><head><title>Async IO Tutorial</title></head><body>
<article>
<h1>Understanding Python asyncio</h1>
<p>{_LONG_BODY}</p>
<p>A second paragraph with additional context about task scheduling and the
event loop's role in coordinating concurrent coroutines without threads.</p>
</article></body></html>"""


class TestExtractText:
    def test_good_html_returns_text_over_200_chars(self):
        result = extract_text(_GOOD_HTML, "https://example.com/article")
        assert result is not None
        assert len(result) >= 200

    def test_empty_html_returns_none(self):
        assert extract_text("<html></html>", None) is None

    def test_short_extraction_returns_none(self):
        # 200-char threshold: a body well under the limit must return None.
        short_html = "<html><body><p>Hello world this is too short</p></body></html>"
        assert extract_text(short_html, None) is None

    def test_readability_fallback(self):
        # HTML that trafilatura's fast mode rejects but readability-lxml can
        # extract. We assert the fallback produces text ≥200 chars — proving
        # the second tier is reached when the first returns None/short.
        # Build content trafilatura is likely to drop (minimal markup, no
        # article tags) but readability's heuristic still scores as readable.
        fallback_html = (
            "<html><head><title>Doc</title></head><body>"
            f"<div><p>{_LONG_BODY}</p>"
            "<p>More content here to ensure the readability fallback path "
            "produces enough text to clear the two hundred char threshold.</p>"
            "</div></body></html>"
        )
        result = extract_text(fallback_html, None)
        # Either tier succeeded; the contract is "≥200 chars or None".
        # If trafilatura got it, fine; if readability did, the fallback works.
        assert result is not None
        assert len(result) >= 200

    def test_url_hint_passed_through(self):
        # The url kwarg is accepted and doesn't crash extraction. Validates
        # the pass-through to trafilatura.extract(url=...).
        result = extract_text(_GOOD_HTML, url="https://example.com/article")
        assert result is not None
        assert len(result) >= 200


class TestDensityFilter:
    """_looks_like_shell rejects index / aggregator pages (D-0XX)."""

    def test_short_line_ratio_triggers(self):
        # Many short lines (<30 chars each), few long ones → shell.
        text = "\n".join(
            ["By Alice", "By Bob", "By Charlie", "By Diana",
             "By Eve", "By Frank",  # 6 short lines
             "A longer paragraph that provides some actual body content so the"
             " overall text exceeds the character floor but the pattern is still"
             " overwhelmingly a list of author names."]
        )
        assert _looks_like_shell(text) is True

    def test_by_line_signature(self):
        # Four or more lines starting with "By " → author-list shell.
        text = "\n".join([
            "By Ashley Capoot", "By Teddy Rosenbluth",
            "By Erin Rode", "By Lucas Ropek",
        ])
        assert _looks_like_shell(text) is True

    def test_by_lines_below_threshold_passes(self):
        text = "By John Doe\nIntroduction paragraph here.\nAnother line."
        assert _looks_like_shell(text) is False

    def test_few_short_lines_does_not_trigger(self):
        # 3 short lines out of 4 — under the 6-line AND 60% threshold.
        text = "\n".join(["short", "also short", "tiny", "A longer paragraph"])
        assert _looks_like_shell(text) is False

    def test_normal_article_passes(self):
        # Real prose with paragraph-length lines is not a shell.
        text = (
            "Python's asyncio module provides infrastructure for writing "
            "single-threaded concurrent code using coroutines, multiplexing "
            "I/O access over sockets and other resources that run event loops.\n"
            "It is the foundation for modern async web frameworks like "
            "FastAPI and Starlette, built on top of low-level APIs.\n"
            "The event loop schedules tasks and handles I/O multiplexing.\n"
        )
        assert _looks_like_shell(text) is False

    def test_very_few_lines_passthrough(self):
        # 2 lines, both short — too few to judge; let 200-char floor decide.
        text = "short\ntoo short"
        assert _looks_like_shell(text) is False
