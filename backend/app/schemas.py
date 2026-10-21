"""Pydantic schemas for request validation and response serialization.

These models define the data contracts between the FastAPI backend and external
clients (such as the React frontend or API consumers).
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

# Reusable type alias for repository lifecycle states
RepositoryStatus = Literal[
    "pending",
    "indexing",
    "completed",
    "failed",
]


class RepositoryCreate(BaseModel):
    """Payload schema for registering a new GitHub repository."""

    url: HttpUrl = Field(
        ...,
        description="Public GitHub repository URL (e.g. 'https://github.com/psf/requests')",
        examples=["https://github.com/psf/requests"],
    )


class RepositoryResponse(BaseModel):
    """Response schema returned when querying repository details."""

    id: int = Field(..., description="Internal RepoPilot database ID")
    github_id: int = Field(..., description="GitHub repository ID")
    full_name: str = Field(..., description="Repository owner/name slug")
    url: str = Field(..., description="GitHub web URL")
    description: str | None = Field(None, description="Repository description")
    stars: int = Field(..., description="Stargazer count")
    language: str | None = Field(None, description="Primary programming language")
    status: RepositoryStatus = Field(
        ...,
        description="Current indexing status: 'pending', 'indexing', 'completed', 'failed'",
    )
    created_at: datetime = Field(..., description="Registration timestamp")

    # Enables automatic mapping from SQLAlchemy ORM instances to Pydantic models.
    # (e.g., db_repo.full_name -> RepositoryResponse.full_name)
    model_config = ConfigDict(from_attributes=True)


class StatusUpdate(BaseModel):
    """Payload schema for manually updating a repository's indexing status."""

    status: RepositoryStatus = Field(
        ...,
        description="Updated lifecycle state for the repository",
    )


# -----------------------------------------------------------------------------
# Phase 2 — Indexing Schemas
# -----------------------------------------------------------------------------

class IndexResponse(BaseModel):
    """Response returned immediately when an indexing job is triggered.

    The job runs in the background; this response just confirms it started.
    """

    message: str = Field(..., description="Human-readable confirmation message")
    repository_id: int = Field(..., description="ID of the repository being indexed")
    status: str = Field(..., description="New status after triggering ('indexing')")


class ChunkResponse(BaseModel):
    """Response schema for a single indexed code chunk.

    Returned by the chunks listing endpoint. The embedding vector is excluded
    from API responses — it is only used internally for vector similarity search.
    """

    id: int = Field(..., description="Chunk primary key")
    repository_id: int = Field(..., description="Parent repository ID")
    file_path: str = Field(..., description="Relative file path in the repository")
    start_line: int = Field(..., description="First line number of this chunk (1-indexed)")
    end_line: int = Field(..., description="Last line number of this chunk (1-indexed)")
    content: str = Field(..., description="Raw source text of the chunk")
    language: str | None = Field(None, description="Detected programming language")
    created_at: datetime = Field(..., description="Timestamp when this chunk was indexed")

    model_config = ConfigDict(from_attributes=True)


class ChunkCountResponse(BaseModel):
    """Response for the chunk count endpoint."""

    repository_id: int = Field(..., description="Repository ID")
    chunk_count: int = Field(..., description="Total number of indexed chunks")