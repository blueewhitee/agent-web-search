from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.search import router as search_router
from app.core.config import settings
from app.services.fetch_service import FetchService
from app.services.robots_cache import RobotsCache
from app.services.searxng_service import SearXNGService


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient() as client:
        app.state.http_client = client
        app.state.searxng_service = SearXNGService(
            base_url=settings.searxng_base_url,
            client=client,
            timeout=settings.searxng_timeout,
        )
        app.state.robots_cache = RobotsCache(
            client=client,
            user_agent=settings.user_agent,
        )
        app.state.fetch_service = FetchService(
            client=client,
            robots=app.state.robots_cache,
            settings=settings,
        )
        yield


app = FastAPI(title="Nature-based Search API", version="0.1.0", lifespan=lifespan)
app.include_router(search_router)
