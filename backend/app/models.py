"""SQLAlchemy ORM database models for RepoPilot.

This module defines the relational schema for repositories registered in
RepoPilot. It uses SQLAlchemy 2.0 type-annotated 'Mapped' columns.
"""

from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Repository(Base):
    """Represents a GitHub repository registered for intelligence and indexing.

    Table Name: 'repositories'

    Lifecycle Statuses:
    - 'pending': Added by user, awaiting indexing pipeline.
    - 'indexing': Currently parsing AST, chunking code, and computing embeddings.
    - 'completed': Ready for hybrid retrieval, codebase Q&A, and bug investigation.
    - 'failed': Encountered an error during fetch, clone, or embedding generation.
    """

    __tablename__ = "repositories"

    # Internal primary key ID (auto-incrementing integer)
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    # GitHub's unique repository ID (prevents re-adding the same repo)
    github_id: Mapped[int] = mapped_column(
        Integer,
        unique=True,
        index=True,
    )

    # Repository slug in format 'owner/repo_name' (e.g., 'facebook/react')
    full_name: Mapped[str] = mapped_column(
        String(200),
        index=True,
    )

    # Web URL to the GitHub repository (e.g., 'https://github.com/facebook/react')
    url: Mapped[str] = mapped_column(
        String(500),
        unique=True,
    )

    # Repository description fetched from GitHub API (can be None)
    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # Star count from GitHub stargazers_count
    stars: Mapped[int] = mapped_column(
        Integer,
        default=0,
    )

    # Primary programming language (e.g., 'Python', 'TypeScript')
    language: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    # Current processing state in the RepoPilot pipeline
    status: Mapped[str] = mapped_column(
        String(30),
        default="pending",
    )

    # Timestamp when this record was inserted (handled by PostgreSQL func.now())
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        """String representation for easy debugging and log inspection."""
        return (
            f"<Repository(id={self.id}, full_name='{self.full_name}', "
            f"status='{self.status}')>"
        )