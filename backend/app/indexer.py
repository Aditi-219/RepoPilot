"""Repository indexing pipeline for RepoPilot.

This module orchestrates the full ingestion pipeline:
  1. Mark repository status as 'indexing'
  2. Fetch the full file tree from GitHub
  3. Download each supported file's content
  4. Chunk each file into overlapping token-bounded segments
  5. Generate 384-dim embeddings for each chunk in worker threads
  6. Bulk-insert all CodeChunk rows into PostgreSQL
  7. Mark repository status as 'completed'
  8. On any failure: mark status as 'failed' and log the error

This function runs as a background task with an isolated database session.
"""

import asyncio
import logging

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.chunker import Chunk, chunk_file
from app.database import SessionLocal
from app.embedder import embed_chunks
from app.github_service import fetch_file_content, fetch_repo_tree, parse_github_url
from app.models import CodeChunk, Repository

logger = logging.getLogger(__name__)

# How many chunks to embed and insert in a single DB transaction.
BATCH_SIZE = 64


def _set_status(db: Session, repo: Repository, status: str) -> None:
    """Update repository status and commit immediately."""
    repo.status = status
    db.commit()
    logger.info("Repository %d (%s) status → %s", repo.id, repo.full_name, status)


async def index_repository(repository_id: int) -> None:
    """Run the full indexing pipeline for a single repository.

    Uses an isolated database session from SessionLocal() to ensure it
    never interferes with or depends on FastAPI HTTP request lifetimes.

    Args:
        repository_id: Primary key of the Repository to index.
    """
    with SessionLocal() as db:
        repo = db.get(Repository, repository_id)
        if repo is None:
            logger.error("index_repository called with unknown repository_id=%d", repository_id)
            return

        logger.info("Starting indexing pipeline for repo: %s (id=%d)", repo.full_name, repo.id)

        try:
            # 1. Mark as 'indexing'
            _set_status(db, repo, "indexing")

            # 2. Delete any previously indexed chunks (re-index support)
            db.execute(delete(CodeChunk).where(CodeChunk.repository_id == repository_id))
            db.commit()
            logger.info("Cleared old chunks for repo %d", repository_id)

            # 3. Parse owner/repo from the stored URL and fetch file tree
            owner, repo_name = parse_github_url(repo.url)
            file_entries = await fetch_repo_tree(owner, repo_name)
            logger.info("Found %d indexable files in %s/%s", len(file_entries), owner, repo_name)

            if not file_entries:
                logger.warning("No supported files found in repo %d. Marking completed.", repository_id)
                _set_status(db, repo, "completed")
                return

            # 4. Process files: fetch → chunk → batch embed → insert
            total_chunks_inserted = 0
            pending_chunks: list[Chunk] = []

            async def flush_batch(force: bool = False) -> None:
                nonlocal total_chunks_inserted, pending_chunks

                if not pending_chunks:
                    return
                if not force and len(pending_chunks) < BATCH_SIZE:
                    return

                # Run CPU-bound embedding in a thread so event loop stays unblocked
                vectors = await asyncio.to_thread(embed_chunks, pending_chunks)

                db_chunks = [
                    CodeChunk(
                        repository_id=repository_id,
                        file_path=chunk.file_path,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        content=chunk.content,
                        language=chunk.language,
                        embedding=vector,
                    )
                    for chunk, vector in zip(pending_chunks, vectors)
                ]

                db.add_all(db_chunks)
                db.commit()

                total_chunks_inserted += len(db_chunks)
                logger.info("Inserted batch of %d chunks (total so far: %d)", len(db_chunks), total_chunks_inserted)
                pending_chunks = []

            for i, file_entry in enumerate(file_entries, start=1):
                file_path: str = file_entry["path"]
                logger.debug("[%d/%d] Fetching %s", i, len(file_entries), file_path)

                content = await fetch_file_content(owner, repo_name, file_path)
                if not content:
                    continue

                file_chunks = chunk_file(file_path, content)
                if not file_chunks:
                    continue

                pending_chunks.extend(file_chunks)
                await flush_batch(force=False)

            # Final flush of leftover chunks
            await flush_batch(force=True)

            # 5. Mark as 'completed'
            logger.info(
                "Indexing complete for repo %d (%s): %d chunks stored.",
                repository_id,
                repo.full_name,
                total_chunks_inserted,
            )
            _set_status(db, repo, "completed")

        except Exception as exc:
            logger.exception(
                "Indexing failed for repo %d (%s): %s",
                repository_id,
                repo.full_name if repo else "unknown",
                exc,
            )
            try:
                _set_status(db, repo, "failed")
            except Exception:
                pass
