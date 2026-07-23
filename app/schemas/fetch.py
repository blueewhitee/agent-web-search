from pydantic import BaseModel


class FetchResult(BaseModel):
    """Internal-only fetch outcome. Never exposed on the API surface.

    `low_confidence` (public) collapses all failure modes to a single bool;
    `method` + `status` retain detail for internal logging.
    """

    url: str
    html: str | None = None
    status: int | None = None
    low_confidence: bool = False
    method: str = "httpx"  # "httpx" | "crawl4ai" | "httpx-failed" | "robots-blocked" | "batch-timeout"
