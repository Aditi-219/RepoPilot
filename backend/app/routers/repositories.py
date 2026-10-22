"""Repository management router for RepoPilot.

This module provides REST API endpoints for:
1. POST   /api/repositories         — Register a new GitHub repository
2. GET    /api/repositories         — List all registered repositories
3. GET    /api/repositories/{id}    — Get detailed metadata for a single repository
4. PATCH  /api/repositories/{id}/status — Update repository processing status
5. DELETE /api/repositories/{id}    — Remove a repository and its indexed code chunks
"""

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import get_db
from app.github_service import fetch_repository
from app.models import Repository
from app.routers.helpers import get_repository_or_404
from app.schemas import RepositoryCreate, RepositoryResponse, StatusUpdate

logger = logging.getLogger(__name__)

# APIRouter lets us group related routes under a common prefix and tag.
# Any endpoint defined below will automatically start with '/api/repositories'.
router = APIRouter(
    prefix="/api/repositories",
    tags=["Repositories"],
)


@router.post(
    "",
    response_model=RepositoryResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new GitHub repository",
)
async def create_repository(
    payload: RepositoryCreate,
    db: Session = Depends(get_db),
) -> Repository:
    """Register a new GitHub repository in RepoPilot.

    What happens step-by-step:
    1. Validates the provided URL to ensure it points to a public github.com repo.
    2. Calls the GitHub REST API to fetch live metadata (stars, language, description).
    3. Checks PostgreSQL to ensure the repo has not already been registered.
    4. Saves the repo record with default status 'pending' (ready for indexing).

    Args:
        payload: Pydantic model containing the repository URL from the user.
        db:      Active database session injected by FastAPI's get_db dependency.

    Returns:
        Repository: The newly created database record.
    """
    # 1. Fetch live repository metadata from GitHub REST API
    metadata: dict[str, Any] = await fetch_repository(str(payload.url))

    # 2. Prevent duplicate entries by checking GitHub's unique numeric repo ID
    existing = db.scalar(
        select(Repository).where(Repository.github_id == metadata["github_id"])
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Repository '{metadata['full_name']}' has already been registered.",
        )

    # 3. Create the database record with default 'pending' lifecycle status
    repository = Repository(
        **metadata,
        status="pending",
    )
    db.add(repository)

    # 4. Safely commit the transaction to PostgreSQL
    try:
        db.commit()
        db.refresh(repository)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save repository to the database.",
        )

    logger.info("Registered new repository: %s (id=%d)", repository.full_name, repository.id)
    return repository


@router.get(
    "",
    response_model=list[RepositoryResponse],
    summary="List all registered repositories",
)
def list_repositories(
    limit: int = Query(default=50, ge=1, le=100, description="Max repositories to return"),
    db: Session = Depends(get_db),
) -> list[Repository]:
    """Retrieve all registered repositories, ordered by newest first.

    Args:
        limit: Maximum number of records to return (between 1 and 100).
        db:    Active database session.

    Returns:
        list[Repository]: List of repository database records.
    """
    statement = (
        select(Repository)
        .order_by(Repository.created_at.desc())
        .limit(limit)
    )
    return list(db.scalars(statement).all())


@router.get(
    "/{repository_id}",
    response_model=RepositoryResponse,
    summary="Get repository details by ID",
)
def get_repository(
    repository_id: int,
    db: Session = Depends(get_db),
) -> Repository:
    """Retrieve metadata for a single repository by its internal database ID.

    Args:
        repository_id: Primary key of the repository in the database.
        db:            Active database session.

    Returns:
        Repository: The matched database record.
    """
    return get_repository_or_404(db, repository_id)


@router.patch(
    "/{repository_id}/status",
    response_model=RepositoryResponse,
    summary="Update repository processing status",
)
def update_repository_status(
    repository_id: int,
    payload: StatusUpdate,
    db: Session = Depends(get_db),
) -> Repository:
    """Manually update the lifecycle status of a repository.

    Allowed statuses: 'pending', 'indexing', 'completed', 'failed'.

    Args:
        repository_id: Internal ID of the repository to update.
        payload:       Schema containing the new status string.
        db:            Active database session.

    Returns:
        Repository: The updated database record.
    """
    repository = get_repository_or_404(db, repository_id)
    repository.status = payload.status
    db.commit()
    db.refresh(repository)
    logger.info("Updated repository %d status to: %s", repository.id, payload.status)
    return repository


@router.delete(
    "/{repository_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a repository and its indexed data",
)
def delete_repository(
    repository_id: int,
    db: Session = Depends(get_db),
) -> None:
    """Delete a repository from RepoPilot.

    Because the `code_chunks` table has `ondelete='CASCADE'` on its foreign key,
    deleting the repository row automatically deletes all associated vector chunks.

    Args:
        repository_id: Internal database ID of the repository to delete.
        db:            Active database session.
    """
    repository = get_repository_or_404(db, repository_id)
    db.delete(repository)
    db.commit()
    logger.info("Deleted repository %d (%s)", repository_id, repository.full_name)
