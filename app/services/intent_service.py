"""Query-intent routing for SearXNG category selection (Stage 1 enhancement).

Pure function: query string -> which SearXNG ``categories`` + ``time_range`` to use.

Rationale: after disabling google/bing/brave/startpage in settings.yml (to stop
30s timeouts), the default ``general`` category only has ``duckduckgo``, which is
weak for (a) breaking news and (b) deep technical content. SearXNG has purpose-
built categories whose engines actually work without scraping google:

  ``news`` -> bing news + reuters (real article URLs, ~1.7s, tested 2026-07-26)
  ``it``   -> stackoverflow, github, mdn, askubuntu, superuser, docker hub
              ONLY matches technical queries; returns 0 for generic words.
  ``general`` -> duckduckgo (evergreen / Wikipedia-style content)

Precedence: CODE > NEWS > GENERAL.  In a coding-agent session, when a query
carries both a technical signal and a news signal (e.g. "python 3.13 released"),
``it`` returns the authoritative docs/release notes, which is what the user
usually wants, and ``it`` engines are the most reliable.

No I/O, no state — easy to unit-test.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class IntentParams:
    categories: tuple[str, ...]   # e.g. ("it",)  or  ("news",)  or  ("general",)
    time_range: str | None        # "day" | "week" | "month" | "year" | None

    def as_params(self) -> dict[str, str]:
        """Shape ready to merge into SearXNG request params."""
        out: dict[str, str] = {"categories": ",".join(self.categories)}
        if self.time_range:
            out["time_range"] = self.time_range
        return out


# ── keyword sets (lowercased; matched as substrings) ─────────────────────

_NEWS_HINTS: set[str] = {
    "news", "latest", "today", "tonight", "yesterday", "this week",
    "this month", "announced", "announces", "announcement", "released",
    "release", "controversy", "controversial", "scandal", "protest",
    "protests", "election", "crash", "outage", "outages", "happened",
    "breaking", "rumor", "rumour", "reportedly", "reports",
    "live updates", "live blog",
}

_CODE_LANGS: set[str] = {
    "python", "javascript", "js", "typescript", "ts", "rust", "go", "golang",
    "java", "kotlin", "swift", "c++", "cpp", "c#", "csharp", "ruby", "php",
    "bash", "shell", "sh", "sql", "powershell", "lua", "scala", "perl",
    "html", "css", "scss",
}

_CODE_LIBS: set[str] = {
    "react", "vue", "angular", "svelte", "nextjs", "next.js", "nuxt",
    "django", "flask", "fastapi", "starlette", "spring", "express",
    "node", "node.js", "npm", "pip", "cargo", "yarn", "pnpm", "tailwind",
    "bootstrap", "jquery",
}

_CODE_HINTS: set[str] = {
    "error", "exception", "traceback", "stacktrace", "stack trace",
    "undefined", "nullpointer", "null pointer", "segfault",
    "segmentation fault", "how to", "how do i", "docs", "documentation",
    "api", "function", "method", "class", "import", "docker", "kubernetes",
    "k8s", "helm", "git", "github", "gitlab", "regex", "regexp", "syntax",
    "compiler", "typeerror", "nameerror", "attributeerror", "install",
    "installation", "dependency", "dependencies", "package", "module",
    "library", "framework", "cli", "command line", "stackoverflow",
    "stack overflow", "vs code", "vscode", "ide", "lint", "linting",
    "pydantic", "pytest", "unittest",
}

_FILE_EXT_RE: re.Pattern[str] = re.compile(
    r"\.(py|js|ts|tsx|jsx|rs|go|java|kt|kts|cpp|cc|c|h|hpp|rb|php|sh|"
    r"sql|ps1|lua|toml|yaml|yml|json|cs)\b",
    re.IGNORECASE,
)


# ── public API ──────────────────────────────────────────────────────────


def detect(query: str) -> IntentParams:
    """Classify a query into SearXNG ``categories`` + ``time_range``.

    Deterministic, side-effect free.  Call from the router before Stage 1.
    """
    q = query.lower()

    # 1) code intent (highest precedence — see module docstring)
    if _is_code(q):
        return IntentParams(categories=("it",), time_range=None)

    # 2) news intent
    if _is_news(q):
        return IntentParams(
            categories=("news",), time_range=_news_time_range(q)
        )

    # 3) default
    return IntentParams(categories=("general",), time_range=None)


# ── helpers ─────────────────────────────────────────────────────────────


def _is_code(q: str) -> bool:
    if _FILE_EXT_RE.search(q):
        return True
    for tok in _CODE_LANGS | _CODE_LIBS | _CODE_HINTS:
        if _token_match(q, tok):
            return True
    return False


def _is_news(q: str) -> bool:
    return any(_token_match(q, tok) for tok in _NEWS_HINTS)


# ── matching helper ────────────────────────────────────────────────────
# Single-word tokens (no space) use \b word-boundary regex so that "api"
# doesn't match inside "capital" and "sh" doesn't match inside "crash".
# Multi-word tokens ("how to", "live updates") use plain substring match
# — they are long enough that substring false-positives are negligible.


def _token_match(query: str, token: str) -> bool:
    if " " in token:
        return token in query
    return re.search(r"\b" + re.escape(token) + r"\b", query) is not None


def _news_time_range(q: str) -> str:
    if any(w in q for w in ("today", "tonight", "yesterday", "breaking")):
        return "day"
    if "this month" in q:
        return "month"
    return "week"
