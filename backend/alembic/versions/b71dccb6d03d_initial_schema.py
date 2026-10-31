"""initial_schema

Revision ID: b71dccb6d03d
Revises:
Create Date: 2026-10-29 16:21:29.839377

This is the baseline migration for RepoPilot. It captures the schema that was
previously created by SQLAlchemy's Base.metadata.create_all(). Existing
databases should be stamped at this revision with:

    alembic stamp b71dccb6d03d

Fresh installs should run the full upgrade:

    alembic upgrade head

Note: The pgvector extension and LangGraph checkpoint tables (checkpoints,
checkpoint_blobs, checkpoint_writes, checkpoint_migrations) are created by
their respective libraries, not by this migration.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = "b71dccb6d03d"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create RepoPilot application tables and indexes from scratch.

    Requires the pgvector extension to be installed in the target database:
        CREATE EXTENSION IF NOT EXISTS vector;
    """
    # Ensure pgvector extension is present
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ------------------------------------------------------------------
    # repositories
    # ------------------------------------------------------------------
    op.create_table(
        "repositories",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column("github_id", sa.Integer(), nullable=False, unique=True, index=True),
        sa.Column("full_name", sa.String(200), nullable=False, index=True),
        sa.Column("url", sa.String(500), nullable=False, unique=True),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("stars", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("language", sa.String(100), nullable=True),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # ------------------------------------------------------------------
    # code_chunks
    # ------------------------------------------------------------------
    op.create_table(
        "code_chunks",
        sa.Column("id", sa.Integer(), primary_key=True, index=True),
        sa.Column(
            "repository_id",
            sa.Integer(),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("start_line", sa.Integer(), nullable=False),
        sa.Column("end_line", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("language", sa.String(50), nullable=True),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # HNSW index for fast cosine similarity search (pgvector)
    op.create_index(
        "idx_code_chunks_embedding_hnsw",
        "code_chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
        postgresql_with={"m": "16", "ef_construction": "64"},
    )

    # GIN index for full-text keyword search
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_code_chunks_content_gin
        ON code_chunks
        USING gin(to_tsvector('english', content))
        """
    )

    # ------------------------------------------------------------------
    # chat_sessions
    # ------------------------------------------------------------------
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "repository_id",
            sa.Integer(),
            sa.ForeignKey("repositories.id", ondelete="CASCADE"),
            nullable=False,
            index=True,
        ),
        sa.Column(
            "title",
            sa.String(255),
            nullable=False,
            server_default="New Conversation",
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )


def downgrade() -> None:
    """Drop all RepoPilot application tables in reverse dependency order."""
    op.drop_table("chat_sessions")
    op.drop_index("idx_code_chunks_content_gin", table_name="code_chunks")
    op.drop_index(
        "idx_code_chunks_embedding_hnsw",
        table_name="code_chunks",
        postgresql_using="hnsw",
    )
    op.drop_table("code_chunks")
    op.drop_table("repositories")
