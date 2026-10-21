"""SQLAlchemy ORM database models for RepoPilot.

This module defines the relational schema for repositories registered in
RepoPilot. It uses SQLAlchemy 2.0 type-annotated 'Mapped' columns.
"""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

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

    # One-to-many relationship: a repo has many indexed code chunks
    chunks: Mapped[list["CodeChunk"]] = relationship(
        "CodeChunk",
        back_populates="repository",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """String representation for easy debugging and log inspection."""
        return (
            f"<Repository(id={self.id}, full_name='{self.full_name}', "
            f"status='{self.status}')>"
        )


class CodeChunk(Base):
    """Represents a single indexed chunk of code or documentation from a repository.

    Each repository file is split into overlapping token-bounded chunks. Every
    chunk stores:
    - The raw source text (for display and keyword search)
    - A 384-dimensional vector embedding (for semantic similarity search)
    - File path and line range metadata (for source attribution)

    Table Name: 'code_chunks'
    """

    __tablename__ = "code_chunks"

    # Primary key
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    # Foreign key linking this chunk to its parent repository
    repository_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("repositories.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )

    # Relative file path within the repository (e.g., 'src/utils/parser.py')
    file_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # Line numbers for source attribution (1-indexed)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)

    # Raw text content of this chunk (shown in search results)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Detected programming language from the file extension (e.g., 'python', 'typescript')
    language: Mapped[str | None] = mapped_column(String(50), nullable=True)

    # 384-dimensional embedding vector from sentence-transformers/all-MiniLM-L6-v2
    # Used for cosine similarity search via pgvector
    embedding: Mapped[list[float]] = mapped_column(
        Vector(384),
        nullable=False,
    )

    # Timestamp when this chunk was indexed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
    )

    # Many-to-one back-reference to the parent repository
    repository: Mapped["Repository"] = relationship(
        "Repository",
        back_populates="chunks",
    )

    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"<CodeChunk(id={self.id}, repo_id={self.repository_id}, "
            f"file='{self.file_path}', lines={self.start_line}-{self.end_line})>"
        )