from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App configuration loaded from env / .env.

    No secrets here — SearXNG is a local self-hosted backend.
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")

    searxng_base_url: str = "http://localhost:8080"
    searxng_timeout: float = 8.0
    top_k_fetch: int = 5  # caps SearXNG results pulled from upstream (fetch wider than return)
    user_agent: str = "AgentWebSearch-Bot/0.1 (agent-web-search)"  # D-012
    per_url_timeout: float = 6.0  # D-001: per-URL fetch deadline
    batch_deadline: float = 8.0   # D-001: whole fan-out deadline
    top_k_return: int = 3        # D-013: results returned to caller
    render_js_default: bool = False  # D-006
    # Hardening fixes:
    extract_timeout: float = 5.0          # per-extraction deadline (trafilatura/lxml hang guard)
    max_chunks_per_url: int = 10          # cap chunks embedded per page (D-025: 20→10; answer is almost always in first ~5 chunks, tail is footer/related)
    searxng_empty_fallback: bool = True   # retry `general` when auto-detected category returns 0
    # Ranking thresholds (D-024): PLACEHOLDER defaults — calibrate from the
    # evals harness (known-relevant + known-garbage query sets) before trusting.
    rank_absolute_floor: float = 0.2  # backstop: max cosine < this -> no signal, return []
    rank_gap_threshold: float = 0.15  # drop chunks more than this below the top hit (relative gap)


settings = Settings()
