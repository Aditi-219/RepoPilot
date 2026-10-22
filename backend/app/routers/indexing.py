"""Indexing and chunk inspection router for RepoPilot.

This module provides endpoints for:
1. POST /api/repositories/{id}/index        — Trigger background ingestion & vectorization
2. GET  /api/repositories/{id}/chunks       — View paginated code chunks
3. GET  /api/repositories/{id}/chunks/count — Total count of indexed chunks
"""

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.database import get_db
from app.indexer import index_repository
from app.models import CodeChunk
from app.routers.helpers import get_repository_or_404
from app.schemas import ChunkCountResponse, ChunkResponse, IndexResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/repositories",
    tags=["Indexing"],
)


@router.post(
    "/{repository_id}/index",
    response_model=IndexResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Trigger code ingestion & vector indexing pipeline",
)
async def trigger_indexing(
    repository_id: int,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> IndexResponse:
    """Trigger the background indexing pipeline for a repository.

    Why 202 Accepted?
    Indexing an entire GitHub codebase (downloading files, calculating tiktoken
    tokens, running local BERT neural embeddings) takes time.
    Instead of making the HTTP request hang and timeout, we return immediately
    with 202 Accepted. FastAPI runs `index_repository()` in a background worker thread.

    Lifecycle transition:
    - pending / failed -> indexing (set immediately)
    - indexing -> completed (when all chunks are embedded & saved)
    - indexing -> failed (if unrecoverable network or parsing errors occur)

    Args:
        repository_id:    Database ID of the repository to index.
        background_tasks: FastAPI background task manager.
        db:               Active database session.

    Returns:
        IndexResponse: Status confirmation and repository ID.
    """
    repository = get_repository_or_404(db, repository_id)

    # Don't allow starting another index if one is actively running
    if repository.status == "indexing":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Repository '{repository.full_name}' is already being indexed. Please wait for it to finish.",
        )

    # Enqueue background task
    background_tasks.add_task(index_repository, repository_id)
    logger.info("Indexing enqueued for repo %d (%s)", repository_id, repository.full_name)

    return IndexResponse(
        message=f"Indexing started for '{repository.full_name}'. Check status via GET /api/repositories/{repository_id}.",
        repository_id=repository_id,
        status="indexing",
    )


@router.get(
    "/{repository_id}/chunks",
    response_model=list[ChunkResponse],
    summary="List indexed code chunks (paginated)",
)
def list_chunks(
    repository_id: int,
    skip: int = Query(default=0, ge=0, description="Offset for pagination"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of chunks to return"),
    db: Session = Depends(get_db),
) -> list[CodeChunk]:
    """Inspect the chunks created for a repository.

    Useful for developers to verify chunking quality, line ranges, and language detection.

    Args:
        repository_id: Database ID of the repository.
        skip:          Number of rows to skip (pagination offset).
        limit:         Maximum number of rows to return (1-100).
        db:            Active database session.

    Returns:
        list[ChunkResponse]: List of chunk records (excluding raw 384-dim vector floats).
    """
    get_repository_or_404(db, repository_id)

    statement = (
        select(CodeChunk)
        .where(CodeChunk.repository_id == repository_id)
        .order_by(CodeChunk.file_path, CodeChunk.start_line)
        .offset(skip)
        .limit(limit)
    )
    return list(db.scalars(statement).all())


@router.get(
    "/{repository_id}/chunks/count",
    response_model=ChunkCountResponse,
    summary="Get total count of indexed chunks",
)
def count_chunks(
    repository_id: int,
    db: Session = Depends(get_db),
) -> ChunkCountResponse:
    """Return the total number of code chunks stored for a repository.

    Args:
        repository_id: Database ID of the repository.
        db:            Active database session.

    Returns:
        ChunkCountResponse: The chunk count.
    """
    get_repository_or_404(db, repository_id)

    count = db.scalar(
        select(func.count(CodeChunk.id)).where(CodeChunk.repository_id == repository_id)
    )

    return ChunkCountResponse(
        repository_id=repository_id,
        chunk_count=count or 0,
    )
