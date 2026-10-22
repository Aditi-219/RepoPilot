"""RAG (Retrieval-Augmented Generation) & search router for RepoPilot.

This module provides endpoints for:
1. POST /api/repositories/{id}/search — Hybrid semantic + full-text code retrieval
2. POST /api/repositories/{id}/ask    — Codebase Q&A with grounded AI answers
"""

import asyncio
import logging

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.llm_service import ask_with_llm
from app.reranker import rerank
from app.routers.helpers import get_repository_or_404, require_indexed
from app.schemas import AskQuery, AskResponse, CodeCitation, SearchQuery, SearchResponse
from app.search import hybrid_search

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/repositories",
    tags=["RAG & Search"],
)


@router.post(
    "/{repository_id}/search",
    response_model=SearchResponse,
    summary="Hybrid semantic + full-text search",
)
def search_repository(
    repository_id: int,
    payload: SearchQuery,
    db: Session = Depends(get_db),
) -> SearchResponse:
    """Search code chunks using both vector similarity and full-text search.

    How it works:
    1. Verifies repository is fully indexed (status == 'completed').
    2. Runs vector cosine similarity search via pgvector on 384-dim BERT embeddings.
    3. Runs PostgreSQL full-text search (tsvector/tsquery) on chunk text.
    4. Merges both rankings using Reciprocal Rank Fusion (RRF).
    5. Optionally applies reranking for maximum precision.

    Args:
        repository_id: Database ID of the repository to search.
        payload:       Search query string, limit, and rerank boolean flag.
        db:            Active database session.

    Returns:
        SearchResponse: Ranked list of matching code citations with line ranges.
    """
    repo = get_repository_or_404(db, repository_id)
    require_indexed(repo)

    # 1. Retrieve candidates — fetch 3x the requested limit so the reranker has options
    candidates = hybrid_search(db, repository_id, payload.query, limit=payload.limit * 3)

    # 2. Rerank or slice candidates to user limit
    if payload.rerank and candidates:
        results = rerank(payload.query, candidates, top_k=payload.limit)
    else:
        results = candidates[: payload.limit]

    citations = [
        CodeCitation(
            chunk_id=r.chunk_id,
            repository_id=r.repository_id,
            file_path=r.file_path,
            start_line=r.start_line,
            end_line=r.end_line,
            content=r.content,
            language=r.language,
            score=round(r.score, 4),
        )
        for r in results
    ]

    logger.info(
        "Search complete: repo=%d query=%r results=%d",
        repository_id, payload.query, len(citations),
    )

    return SearchResponse(
        query=payload.query,
        repository_id=repository_id,
        results=citations,
    )


@router.post(
    "/{repository_id}/ask",
    response_model=AskResponse,
    summary="Ask questions about the codebase (RAG Q&A)",
)
async def ask_repository(
    repository_id: int,
    payload: AskQuery,
    db: Session = Depends(get_db),
) -> AskResponse:
    """Answer questions about the codebase grounded in retrieved code citations.

    The complete RAG Pipeline:
    1. Retrieve relevant code chunks via hybrid search (pgvector + PostgreSQL full-text).
    2. Deduplicate and score chunks via reranker.
    3. Construct a grounded context prompt with precise file paths and line ranges.
    4. Call local Ollama (if running) or Google Gemini API.
    5. Return the explanation along with expandable citations for the user to verify.

    Args:
        repository_id: Database ID of the repository.
        payload:       Question string and top_k snippet count.
        db:            Active database session.

    Returns:
        AskResponse: Generated answer, citations list, and model identifier.
    """
    repo = get_repository_or_404(db, repository_id)
    require_indexed(repo)

    # 1. Hybrid retrieval (runs in thread pool to prevent blocking the async event loop)
    candidates = await asyncio.to_thread(
        hybrid_search,
        db,
        repository_id,
        payload.question,
        payload.top_k * 3,
    )

    # 2. Rerank candidates
    snippets = await asyncio.to_thread(
        rerank,
        payload.question,
        candidates,
        payload.top_k,
    )

    # 3. LLM generation with graceful degradation if API key/Ollama is missing
    try:
        answer, model_used = await ask_with_llm(payload.question, snippets)
    except RuntimeError as exc:
        answer = f"⚠️ {exc}"
        model_used = "none"
        logger.warning("LLM notice: %s", exc)
    except Exception as exc:
        answer = f"⚠️ LLM generation failed ({type(exc).__name__}: {exc}). Showing retrieved citations below."
        model_used = "error"
        logger.exception("LLM error for repo %d: %s", repository_id, exc)

    citations = [
        CodeCitation(
            chunk_id=r.chunk_id,
            repository_id=r.repository_id,
            file_path=r.file_path,
            start_line=r.start_line,
            end_line=r.end_line,
            content=r.content,
            language=r.language,
            score=round(r.score, 4),
        )
        for r in snippets
    ]

    logger.info(
        "Ask complete: repo=%d citations=%d model=%s",
        repository_id, len(citations), model_used,
    )

    return AskResponse(
        question=payload.question,
        answer=answer,
        citations=citations,
        model=model_used,
        repository_id=repository_id,
    )
