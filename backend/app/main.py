"""Main application entry point for the RepoPilot FastAPI backend.

This module initializes the FastAPI app, configures CORS, handles database table
creation on startup, and exposes REST endpoints for repository management.
"""

import os
from contextlib import asynccontextmanager
from typing import Any

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import Base, engine, get_db
from app.github_service import fetch_repository
from app.models import Repository
from app.schemas import RepositoryCreate, RepositoryResponse, StatusUpdate

# Load environment configuration
load_dotenv()

# Determine allowed frontend origins for Cross-Origin Resource Sharing (CORS)
# Defaults to standard Vite dev server ports (localhost and 127.0.0.1)
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
ALLOWED_ORIGINS = [
    FRONTEND_URL,
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager.

    Code before 'yield' runs when the server boots up.
    Code after 'yield' runs when the server shuts down.
    """
    # Create database tables automatically if they do not exist yet.
    # Note: In production or larger stages, Alembic migrations are used instead.
    Base.metadata.create_all(bind=engine)
    yield


# Initialize FastAPI instance
app = FastAPI(
    title="RepoPilot API",
    description=(
        "Backend API for RepoPilot: AI-powered codebase intelligence, "
        "hybrid retrieval, and bug debugging copilot."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS Middleware
# Allows the React frontend to make HTTP requests from a different origin/port
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)


# -----------------------------------------------------------------------------
# Root & Health Check Endpoints
# -----------------------------------------------------------------------------

@app.get("/", tags=["General"])
def root() -> dict[str, str]:
    """Root endpoint welcoming developers and linking to interactive docs."""
    return {
        "message": "Welcome to RepoPilot API",
        "docs": "/docs",
        "version": "0.1.0",
    }


@app.get("/api/health", tags=["General"])
def health() -> dict[str, str]:
    """Health check endpoint used by uptime monitors or container orchestrators."""
    return {"status": "healthy"}


# -----------------------------------------------------------------------------
# Repository Management Endpoints
# -----------------------------------------------------------------------------

@app.post(
    "/api/repositories",
    response_model=RepositoryResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Repositories"],
)
async def create_repository(
    payload: RepositoryCreate,
    db: Session = Depends(get_db),
) -> Repository:
    """Register a new GitHub repository in RepoPilot.

    Steps performed:
    1. Validates and parses the GitHub URL.
    2. Calls the GitHub REST API to fetch live repo metadata (stars, language, etc.).
    3. Checks PostgreSQL to ensure the repo hasn't already been added.
    4. Saves the repo with initial status 'pending' to await indexing.

    Args:
        payload: RepositoryCreate containing the public GitHub URL.
        db: Injected SQLAlchemy database session.

    Returns:
        Repository: Newly created repository database record.
    """
    # 1. Fetch live repository metadata from GitHub API
    metadata: dict[str, Any] = await fetch_repository(str(payload.url))

    # 2. Check for duplicate repository by GitHub ID
    existing = db.scalar(
        select(Repository).where(
            Repository.github_id == metadata["github_id"]
        )
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Repository '{metadata['full_name']}' has already been registered.",
        )

    # 3. Create the database record with default 'pending' status
    repository = Repository(
        **metadata,
        status="pending",
    )
    db.add(repository)

    # 4. Commit transaction safely
    try:
        db.commit()
        db.refresh(repository)
    except Exception:
        db.rollback()  # Rollback changes if an error occurs during commit
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save repository to the database.",
        )

    return repository


@app.get(
    "/api/repositories",
    response_model=list[RepositoryResponse],
    tags=["Repositories"],
)
def list_repositories(
    limit: int = Query(default=50, ge=1, le=100, description="Max repositories to return"),
    db: Session = Depends(get_db),
) -> list[Repository]:
    """List registered repositories, ordered by newest first.

    Args:
        limit: Maximum number of records to return (1-100).
        db: Injected database session.

    Returns:
        list[Repository]: List of repository records.
    """
    statement = (
        select(Repository)
        .order_by(Repository.created_at.desc())
        .limit(limit)
    )
    return list(db.scalars(statement).all())


@app.get(
    "/api/repositories/{repository_id}",
    response_model=RepositoryResponse,
    tags=["Repositories"],
)
def get_repository(
    repository_id: int,
    db: Session = Depends(get_db),
) -> Repository:
    """Retrieve details for a single repository by its database ID.

    Args:
        repository_id: Primary key of the repository.
        db: Injected database session.

    Returns:
        Repository: Found repository record.
    """
    repository = db.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository with ID {repository_id} not found.",
        )
    return repository


@app.patch(
    "/api/repositories/{repository_id}/status",
    response_model=RepositoryResponse,
    tags=["Repositories"],
)
def update_repository_status(
    repository_id: int,
    payload: StatusUpdate,
    db: Session = Depends(get_db),
) -> Repository:
    """Update the indexing lifecycle status of a repository.

    Used by the indexing worker or manual UI controls to move a repo
    between 'pending', 'indexing', 'completed', and 'failed'.

    Args:
        repository_id: ID of the repository to update.
        payload: StatusUpdate schema with the new status.
        db: Injected database session.

    Returns:
        Repository: Updated repository record.
    """
    repository = db.get(Repository, repository_id)
    if repository is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository with ID {repository_id} not found.",
        )

    repository.status = payload.status
    db.commit()
    db.refresh(repository)

    return repository