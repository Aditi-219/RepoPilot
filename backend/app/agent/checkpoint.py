"""Checkpointer management for durable LangGraph conversation persistence.

Manages PostgreSQL-backed checkpoint storage using AsyncPostgresSaver, with automatic
fallback to an in-memory MemorySaver for local development and isolated testing.
"""

import logging
import os
import sys
from typing import Any

if sys.platform == "win32":
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)

_checkpointer: BaseCheckpointSaver | None = None
_pool: Any = None


def _get_postgres_conn_info() -> str | None:
    """Normalize the database URL for pure psycopg connection."""
    url = os.getenv("DATABASE_URL", "")
    if not url:
        return None
    # Strip SQLAlchemy dialect prefix if present (e.g. 'postgresql+psycopg://' -> 'postgresql://')
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql://", 1)
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql://", 1)
    return url


async def init_checkpointer() -> BaseCheckpointSaver:
    """Initialize and set up the global LangGraph checkpointer.

    Attempts to connect to PostgreSQL via AsyncConnectionPool and runs checkpointer.setup()
    to ensure checkpointer tables (checkpoints, checkpoint_blobs, checkpoint_writes) exist.
    If the database is unreachable, safely falls back to MemorySaver.
    """
    global _checkpointer, _pool
    if _checkpointer is not None:
        return _checkpointer

    conn_info = _get_postgres_conn_info()
    if conn_info:
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
            from psycopg_pool import AsyncConnectionPool

            _pool = AsyncConnectionPool(
                conninfo=conn_info,
                max_size=10,
                open=False,
                kwargs={"autocommit": True},
            )
            await _pool.open()
            saver = AsyncPostgresSaver(_pool)
            await saver.setup()
            _checkpointer = saver
            logger.info("Durable PostgreSQL LangGraph checkpointer initialized successfully.")
            return _checkpointer
        except Exception as exc:
            logger.warning(
                "Could not initialize PostgreSQL checkpointer (%s). Falling back to MemorySaver.",
                exc,
            )

    # In-memory fallback
    _checkpointer = MemorySaver()
    logger.info("In-memory LangGraph MemorySaver active.")
    return _checkpointer


async def close_checkpointer() -> None:
    """Cleanly close the checkpointer connection pool during application shutdown."""
    global _checkpointer, _pool
    if _pool is not None:
        try:
            await _pool.close()
            logger.info("Checkpointer connection pool closed.")
        except Exception as exc:
            logger.warning("Error closing checkpointer connection pool: %s", exc)
        finally:
            _pool = None
    _checkpointer = None


def get_checkpointer() -> BaseCheckpointSaver:
    """Return the active checkpointer instance (or MemorySaver fallback)."""
    global _checkpointer
    if _checkpointer is None:
        _checkpointer = MemorySaver()
    return _checkpointer


def set_checkpointer(checkpointer: BaseCheckpointSaver | None) -> None:
    """Override the active checkpointer (primarily used in automated unit tests)."""
    global _checkpointer
    _checkpointer = checkpointer
