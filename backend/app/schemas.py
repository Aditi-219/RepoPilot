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
    status: RepositoryStatus = Field(..., description="New status after triggering ('indexing')")


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


# -----------------------------------------------------------------------------
# Phase 3 — RAG Search & Q&A Schemas
# -----------------------------------------------------------------------------

class SearchQuery(BaseModel):
    """Request body for the hybrid search endpoint.

    The search combines pgvector semantic similarity and PostgreSQL full-text
    search, fused via Reciprocal Rank Fusion.
    """

    query: str = Field(
        ...,
        description="Natural language search query about the codebase",
        examples=["Where is authentication handled?"],
    )
    limit: int = Field(
        default=8,
        ge=1,
        le=50,
        description="Number of results to return after retrieval and reranking",
    )
    rerank: bool = Field(
        default=True,
        description="Whether to apply cross-encoder reranking for higher precision",
    )


class CodeCitation(BaseModel):
    """A single retrieved code chunk with full source attribution.

    Used as citations in both search and Q&A responses. The score field
    holds the RRF score before reranking, or the cross-encoder logit after.
    """

    chunk_id: int = Field(..., description="Database primary key of the code chunk")
    repository_id: int = Field(..., description="Parent repository ID")
    file_path: str = Field(..., description="Relative file path in the repository")
    start_line: int = Field(..., description="First line of this chunk (1-indexed)")
    end_line: int = Field(..., description="Last line of this chunk (1-indexed)")
    content: str = Field(..., description="Raw source text of the chunk")
    language: str | None = Field(None, description="Detected programming language")
    score: float = Field(..., description="Relevance score (RRF or cross-encoder logit)")


class SearchResponse(BaseModel):
    """Response from the hybrid search endpoint."""

    query: str = Field(..., description="The original search query")
    repository_id: int = Field(..., description="Repository that was searched")
    results: list[CodeCitation] = Field(
        ...,
        description="Ranked list of matching code chunks with citations",
    )


class AskQuery(BaseModel):
    """Request body for the RAG Q&A endpoint.

    Triggers the full pipeline: hybrid retrieval -> cross-encoder reranking
    -> prompt construction -> Gemini generation.
    """

    question: str = Field(
        ...,
        description="Natural language question about the codebase",
        examples=["Why might this API return a 404?"],
    )
    top_k: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Number of code snippets to retrieve and use as LLM context",
    )


class AskResponse(BaseModel):
    """Response from the RAG Q&A endpoint.

    Contains the AI-generated answer grounded in retrieved code snippets,
    plus the full citations so the frontend can display clickable file links.
    """

    question: str = Field(..., description="The original question")
    answer: str = Field(
        ...,
        description=(
            "AI-generated answer grounded in retrieved code snippets. "
            "If no LLM key is configured, contains a notice and the citations "
            "alone are still returned."
        ),
    )
    citations: list[CodeCitation] = Field(
        ...,
        description="Code snippets used as context for generation, in ranked order",
    )
    model: str = Field(
        ...,
        description="LLM model name used for generation, or 'none' if no key is set",
    )
    repository_id: int = Field(..., description="Repository that was queried")
