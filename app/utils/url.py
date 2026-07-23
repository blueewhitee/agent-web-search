"""URL normalization + result dedup (D-004).

Pure functions, no I/O. Normalization collapses URLs that point at the
same content despite cosmetic differences (mobile subdomain, trailing
slash, tracking params). Dedup keeps the highest-scoring result per
normalized URL.
"""

from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from app.schemas.search import SearchResult

# Tracking / referral params to strip. `utm_*` matched by prefix; rest by exact key.
_TRACKING_PREFIXES = ("utm_",)
_TRACKING_EXACT = frozenset(
    {"gclid", "fbclid", "ref", "source", "_ga", "mc_cid"}
)

# Subdomain labels to drop wherever they appear in the netloc. Label-based
# (not prefix) so `en.m.wikipedia.org` collapses with `en.wikipedia.org`;
# the plan's test case requires language-prefixed mobile to dedup.
_STRIPPABLE_LABELS = frozenset({"www", "m", "mobile"})


def normalize_url(url: str) -> str:
    """Normalize a URL for dedup purposes.

    Steps:
      1. urlsplit into components.
      2. Strip mobile/www subdomain prefix from netloc.
      3. Strip trailing '/' from path (preserve root '/').
      4. Drop tracking params (utm_* prefix + exact set), sort remaining keys.
      5. Lowercase scheme + netloc; keep path case-sensitive.
      6. Reassemble, drop fragment.
    URLs with no scheme/netloc are returned stripped of fragment only —
    SearXNG emits full URLs so this is a defensive fallback.
    """
    parsed = urlsplit(url)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()

    # Subdomain strip: drop www/m/mobile labels, but keep >=2 labels so
    # degenerate hosts like `m.com` aren't reduced to a bare TLD.
    labels = netloc.split(".")
    cleaned = [lab for lab in labels if lab not in _STRIPPABLE_LABELS]
    if len(cleaned) >= 2:
        netloc = ".".join(cleaned)

    # Path: strip trailing slash unless it's the root
    path = parsed.path
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    # Query: drop tracking params, sort remaining
    if parsed.query:
        pairs = parse_qs(parsed.query, keep_blank_values=False)
        filtered = {}
        for key, values in pairs.items():
            key_lower = key.lower()
            if key_lower in _TRACKING_EXACT:
                continue
            if any(key_lower.startswith(p) for p in _TRACKING_PREFIXES):
                continue
            filtered[key] = values
        # parse_qs returns lists; take first value per key, sort by key
        sorted_pairs = sorted(
            (k, v[0] if v else "") for k, v in filtered.items()
        )
        query = urlencode(sorted_pairs)
    else:
        query = ""

    return urlunsplit((scheme, netloc, path, query, ""))


def dedup_results(results: list[SearchResult]) -> list[SearchResult]:
    """Deduplicate search results by normalized URL.

    Walks results in order; on collision keeps the entry with the higher
    `searxng_score` (ties keep first-seen). Returns winners in their
    original index order.
    """
    seen: dict[str, SearchResult] = {}
    for r in results:
        key = normalize_url(r.url)
        existing = seen.get(key)
        if existing is None:
            seen[key] = r
            continue
        # Collision: keep higher score; ties keep first-seen (existing)
        if r.searxng_score > existing.searxng_score:
            seen[key] = r
    return list(seen.values())
