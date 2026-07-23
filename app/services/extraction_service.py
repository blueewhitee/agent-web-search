"""Stage 3 text extraction (D-002, D-006 sub-topic 2).

Two-tier: trafilatura primary, readability-lxml fallback. Returns extracted
text when ≥ 200 chars, else None. Caller (router) maps None to
`low_confidence=True` and leaves `raw_content` unset.

trafilatura is synchronous; callers wrap in `asyncio.to_thread` to keep the
event loop responsive. Encoding is handled transparently by trafilatura via
charset_normalizer (https://github.com/adbar/trafilatura/blob/v2.1.0/trafilatura/utils.py#L168-L201).
"""

_MIN_TEXT_LEN = 200  # D-006 sub-topic 2: "JS-page tell" threshold


def extract_text(html: str, url: str | None = None) -> str | None:
    """Two-tier extraction. Returns text ≥ 200 chars, else None."""
    text = _try_trafilatura(html, url)
    if text is not None and len(text) >= _MIN_TEXT_LEN:
        return text
    text = _try_readability(html)
    if text is not None and len(text) >= _MIN_TEXT_LEN:
        return text
    return None


def _try_trafilatura(html: str, url: str | None) -> str | None:
    from trafilatura import extract  # lazy import: trafilatura is heavy
    return extract(html, url=url, output_format="txt", fast=True,
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
