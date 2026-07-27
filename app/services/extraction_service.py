"""Stage 3 text extraction (D-002, D-006 sub-topic 2).

Two-tier: trafilatura primary, readability-lxml fallback. Returns extracted
text when ≥ 200 chars, else None. Caller (router) maps None to
`low_confidence=True` and leaves `raw_content` unset.

trafilatura is synchronous; callers wrap in `asyncio.to_thread` to keep the
event loop responsive. Encoding is handled transparently by trafilatura via
charset_normalizer (https://github.com/adbar/trafilatura/blob/v2.1.0/trafilatura/utils.py#L168-L201).
"""

_MIN_TEXT_LEN = 200  # D-006 sub-topic 2: "JS-page tell" threshold
_MAX_HTML_LEN = 100_000  # D-025: cap HTML before trafilatura (15x latency win, 100% content retained)


def _looks_like_shell(text: str) -> bool:
    """Heuristic: index / aggregator page (author-name lists, link soup).

    Targets a failure mode where SearXNG returns an aggregator topic page or
    category index and trafilatura extracts *something* (author names, short
    link captions) but it is not a real article body.

    The 200-char floor in ``extract_text`` catches empty shells.  This check
    catches slightly larger shells that pass the floor but are still junk.

    Returns True when the text looks more like an index than prose.
    """
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= 2:
        return False  # too little to judge; the 200-char floor catches empties
    short = sum(1 for ln in lines if len(ln.strip()) < 30)
    if len(lines) >= 6 and short / len(lines) >= 0.6:
        return True
    # author-list signature ("By Ashley Capoot", "By ...")
    by_lines = sum(1 for ln in lines if ln.strip().lower().startswith("by "))
    if by_lines >= 4:
        return True
    return False


def extract_text(html: str, url: str | None = None) -> str | None:
    """Two-tier extraction. Returns text ≥ 200 chars, else None."""
    # Cap input HTML before parsing. trafilatura/lxml parse cost scales with
    # HTML size; a 465KB page takes ~2.4s vs ~0.16s at 100KB (15x). Article
    # body is almost always in the first 100KB — nav/boilerplate/footer come
    # later. Empirically verified (D-025): 100KB retains 100% of article text
    # on docs.python.org; 50KB loses 42%. Safe for quality, huge for latency.
    if len(html) > _MAX_HTML_LEN:
        html = html[:_MAX_HTML_LEN]
    text = _try_trafilatura(html, url)
    if text is not None and len(text) >= _MIN_TEXT_LEN:
        return text if not _looks_like_shell(text) else None
    text = _try_readability(html)
    if text is not None and len(text) >= _MIN_TEXT_LEN:
        return text if not _looks_like_shell(text) else None
    return None


def _try_trafilatura(html: str, url: str | None) -> str | None:
    from trafilatura import extract  # lazy import: trafilatura is heavy
    # output_format="markdown": preserves headings, bullets, and tables
    # that plain text loses. Markdown is what downstream LLMs read best
    # (2026 benchmarks: markdown is 35-38% more token-efficient than JSON
    # for LLM context, and LLMs are trained heavily on it). The calling
    # harness decides the final prompt format; we just hand it structured
    # text rather than flattened prose.
    return extract(html, url=url, output_format="markdown", fast=True,
                   include_comments=False, include_tables=True)


def _try_readability(html: str) -> str | None:
    from readability import Document  # lazy import: secondary extractor
    from lxml import html as lxml_html
    try:
        doc = Document(html)
        article_html = doc.summary()
        # readability returns an HTML fragment; strip tags to plain text.
        fragment = lxml_html.fromstring(article_html)
        text = fragment.text_content()
    except Exception:
        return None
    text = " ".join(text.split())  # collapse whitespace
    return text if text else None
