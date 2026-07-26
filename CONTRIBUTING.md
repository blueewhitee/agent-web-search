# Contributing to Agent Web Search

Thanks for taking the time to contribute. This project is small and self-hosted, so the process is intentionally lightweight.

## Setup

```bash
git clone https://github.com/blueewhitee/agent-web-search.git
cd agent-web-search
uv sync
```

You also need SearXNG on `localhost:8080` to run the integration tests:

```bash
docker compose up searxng -d
```

## Before you submit a PR

1. **Fork** the repo and create a branch: `git checkout -b feat/your-feature`.
2. **Run the tests** and make sure they all pass:
   ```bash
   uv run pytest
   ```
   We keep 140 tests green. If you add behaviour, add a test for it.
3. **Keep intent routing deterministic.** The `intent_service.py` router is pure (no I/O, no state) and covered by 36 tests. New keyword sets must use word-boundary matching (`_token_match`) — see the module docstring for why substring matching causes false positives (`"api"` inside `"capital"`).
4. **Don't re-enable Google / Bing / Brave.** They're disabled in `searxng/config/settings.yml` because they cause 30s timeouts. If you want to experiment, do it in your own fork and measure first.

## What we're looking for

Good first issues:

- **Evals harness** — precision@k measurement over a fixed query set.
- **Python SDK** — thin wrapper around the HTTP API.
- **More extraction fallbacks** — Playwright / Jina Reader for stubborn JS pages.
- **Engine tuning** — measurable ranking improvements from new SearXNG engines or weighting.

Less good:

- Cosmetic README tweaks without a reason.
- Re-architecting the pipeline — talk first.

## Commit messages

Conventional Commits style:

```
feat(intent): route "rust async" to the it category
fix(fetch): drop body timeout to 6s for slow engines
docs: clarify SearXNG engine table
```

## Response time

We review and respond to PRs within **48 hours**. If a PR sits longer, ping it once — we may have missed the notification.

## License

By contributing, you agree your contributions are licensed under the project's [MIT License](LICENSE).