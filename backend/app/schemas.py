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