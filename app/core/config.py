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


settings = Settings()
