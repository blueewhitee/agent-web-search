import asyncio

import httpx
from fastapi import APIRouter, HTTPException, Request

from app.core.config import settings
from app.schemas.search import ChunkSource, RankedChunk, SearchRequest, SearchResponse
from app.services.chunking_service import chunk_text
from app.services.extraction_service import extract_text
from app.services.ranking_service import rank_chunks
from app.services.scrubber_service import scrub_content
from app.services.fetch_service import FetchService
from app.services.searxng_service import SearXNGService
from app.utils.url import dedup_results

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(body: SearchRequest, request: Request) -> SearchResponse:
    """Stage 1 (SearXNG) → Stage 2 (dedup + concurrent fetch) orchestrator.

    The router stays the orchestrator — matches the existing Stage 1 pattern.
    A PipelineService abstraction is deferred until Stage 3 lands and the
    orchestrator grows further.
    """
    searxng: SearXNGService = request.app.state.searxng_service
    fetch: FetchService = request.app.state.fetch_service

    # Stage 1: SearXNG (error map from D-014)
    try:
        results, unresponsive = await searxng.search(
            body.query, top_k=settings.top_k_fetch
        )
    except httpx.TimeoutException:
        raise HTTPException(
            504,
            detail={
                "upstream": "searxng",
                "error": "timeout",
                "timeout_s": settings.searxng_timeout,
            },
        )
    except httpx.TransportError as e:
        # ConnectError / NetworkError / etc. — upstream unreachable, same
        # 504 meaning as a timeout (gateway could not get a response).
        raise HTTPException(
            504,
            detail={
                "upstream": "searxng",
                "error": "unreachable",
                "error_type": type(e).__name__,
            },
        )
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            502,
            detail={"upstream": "searxng", "status": e.response.status_code},
        )

    # Stage 2a: dedup (D-004)
    results = dedup_results(results)

    # Stage 2b: fetch with dual deadlines (D-001, D-006, D-012)
    fetch_results = await fetch.fetch_many(
        [r.url for r in results], render_js=body.render_js
    )
    fetch_by_url = {fr.url: fr for fr in fetch_results}

    # Stage 2c: merge fetch-quality flags only (raw_content deferred to Stage 3)
    for r in results:
        fr = fetch_by_url.get(r.url)
        if fr:
            r.low_confidence = fr.low_confidence

    # Stage 3: concurrent text extraction (D-002, D-006 sub-topic 2)
    async def _extract_for(r):
        fr = fetch_by_url.get(r.url)
        html = fr.html if fr else None
        if html is None:
            return None
        return await asyncio.to_thread(extract_text, html, r.url)

    extraction_results = await asyncio.gather(*(_extract_for(r) for r in results))
    for r, text in zip(results, extraction_results):
        # Additive quality signal: extraction failure ADDS low_confidence
        # (fetch-stage failures already set it; we don't clear it either way).
        if text is None:
            r.low_confidence = True
        # D-002: raw_content is extracted text (opt-in). None if caller didn't ask
        # or extraction failed — Pydantic will drop the field from the JSON
        # response when None per the `exclude_none` pattern (see Phase 4).
        if body.include_content:
            r.raw_content = text

    # Stage 3.5: scrub prompt injection (D-010)
    for r in results:
        if r.raw_content is not None:
            scrub_result = scrub_content(r.raw_content)
            r.raw_content = scrub_result.content

    # Stage 4: chunk extracted text for retrieval
    for r in results:
        if r.raw_content is not None:
            chunks = chunk_text(r.raw_content)
            r.chunks = [
                {"text": c.text, "parent_text": c.parent_text, "chunk_index": c.chunk_index}
                for c in chunks
            ]

    # Stage 5: pool chunks across all results, rank globally, return top-K
    all_chunks = []
    for r in results:
        if r.chunks:
            for c in r.chunks:
                all_chunks.append({
                    **c,
                    "source": {
                        "url": r.url,
                        "title": r.title,
                        "searxng_score": r.searxng_score,
                    },
                })

    ranked = rank_chunks(body.query, all_chunks, top_k=settings.top_k_return)

    if ranked is None:
        # Model unavailable — degrade to flat D-013 fallback
        ranked = []
        for r in results[: settings.top_k_return]:
            for c in r.chunks or []:
                ranked.append({
                    **c,
                    "score": 0.0,
                    "source": {
                        "url": r.url,
                        "title": r.title,
                        "searxng_score": r.searxng_score,
                    },
                })
        ranked = ranked[: settings.top_k_return]

    ranked_chunks = [
        RankedChunk(
            text=c["text"],
            parent_text=c["parent_text"],
            chunk_index=c["chunk_index"],
            score=c["score"],
            source=ChunkSource(**c["source"]),
        )
        for c in ranked
    ]

    return SearchResponse(
        query=body.query,
        ranked_chunks=ranked_chunks,
        unresponsive_engines=unresponsive,
    )
