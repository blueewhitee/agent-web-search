"""Thin lazy-import adapter for crawl4ai (D-006).

crawl4ai is optional and heavy (Playwright-backed). This module is
imported only when a caller passes `render_js=True`. If crawl4ai is
missing or its API has shifted, the ImportError (or any Exception)
propagates to `FetchService.fetch_one`, which falls back to httpx —
no user-visible error.

API verified against crawl4ai v0.9.0 (commit c66f327):
  - AsyncWebCrawler(config: BrowserConfig = None, ...)
  - crawler.arun(url: str, config: CrawlerRunConfig = None) -> CrawlResultContainer
  - CrawlResultContainer.__getattr__ delegates to results[0], so `.html` works
  - BrowserConfig(headless: bool = True, user_agent: str = <default>, ...)
"""

async def fetch_with_crawl4ai(url: str, user_agent: str) -> str:
    """Fetch a URL with JS rendering via crawl4ai. Returns page HTML.

    Lazy import: if crawl4ai is not installed, ImportError propagates
    and the caller falls back to httpx.
    """
    from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

    browser_cfg = BrowserConfig(user_agent=user_agent, headless=True)
    run_cfg = CrawlerRunConfig()
    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        result = await crawler.arun(url=url, config=run_cfg)
        # CrawlResultContainer delegates .html to the first CrawlResult.
        return result.html or ""
