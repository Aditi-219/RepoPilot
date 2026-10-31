"""Shared helper functions for RepoPilot API routers.

This module contains common validation and retrieval functions used across
different route handlers (repositories, indexing, RAG).
"""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models import Repository


def get_repository_or_404(db: Session, repository_id: int) -> Repository:
    """Fetch a repository from PostgreSQL by ID or raise an HTTP 404 error.

    Why this helper matters:
    Instead of writing `db.get(...)` and `if repository is None: raise ...`
    in every single endpoint, we centralize this logic here.

    Args:
        db:            The active SQLAlchemy database session.
        repository_id: The primary key integer of the repository.

    Returns:
        Repository: The matching database record if found.

    Raises:
        HTTPException(404): If no repository with this ID exists in the database.
    """
    repository = db.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository with ID {repository_id} not found.",
        )
    return repository


def require_indexed(repository: Repository) -> None:
    """Ensure a repository has finished its indexing pipeline before running queries.

    Why this helper matters:
    RAG search and question-answering require vector embeddings and token chunks
    to exist in the database. If a user tries to search an un-indexed or failed repo,
    we stop them early with HTTP 409 Conflict and an actionable message.

    Args:
        repository: The Repository database record to check.

    Raises:
        HTTPException(409): If the repository's status is not 'completed'.
    """
    if repository.status != "completed":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Repository '{repository.full_name}' is not fully indexed "
                f"(current status: '{repository.status}'). "
                "Trigger indexing first via POST /api/repositories/{id}/index."
            ),
        )
