"""Hybrid retrieval engine for RepoPilot Phase 3.

Combines semantic search (pgvector cosine similarity) and full-text search
(PostgreSQL tsvector/tsquery) via Reciprocal Rank Fusion (RRF).

Pipeline:
  1. Embed the query -> 384-dim vector via the same BERT model used at index time
  2. Semantic search  -> top-N chunks by cosine distance  (embedding <=> query_vec)
  3. Full-text search -> top-N chunks by ts_rank          (tsvector @@ tsquery)
  4. RRF fusion       -> single ranked list without needing score normalization

Reference: Cormack, Clarke & Buettcher (2009) - "Reciprocal Rank Fusion
outperforms Condorcet and individual Rank Learning Methods"
"""

import logging
from dataclasses import dataclass, field

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.embedder import embed_query

logger = logging.getLogger(__name__)

# Standard RRF constant — empirically optimal at k=60
RRF_K: int = 60


@dataclass
class SearchResult:
    """A single retrieved code chunk with source attribution and relevance score.

    Attributes:
        chunk_id:      Primary key of the CodeChunk row.
        repository_id: Parent repository ID.
        file_path:     Relative file path within the repository.
        start_line:    First line of the chunk (1-indexed).
        end_line:      Last line of the chunk (1-indexed).
        content:       Raw source text of the chunk.
        language:      Detected programming language (e.g. 'python').
        score:         Relevance score — RRF score before reranking,
                       cross-encoder logit score after reranking.
    """

    chunk_id: int
    repository_id: int
    file_path: str
    start_line: int
    end_line: int
    content: str
    language: str | None
    score: float = field(default=0.0)


def _semantic_search(
    db: Session,
    repository_id: int,
    query_vector: list[float],
    limit: int,
) -> list[dict]:
    """Run cosine similarity search using pgvector's <=> distance operator.

    Requires: pgvector extension + HNSW index on code_chunks.embedding
    (created by the lifespan handler in main.py on startup).

    Args:
        db:            Active SQLAlchemy session.
        repository_id: Restrict search to this repository.
        query_vector:  384-dim embedding of the user query.
        limit:         Max rows to return.

    Returns:
        List of row dicts ordered by ascending cosine distance (most similar first).
    """
    # Cast the Python list to a pgvector literal so SQLAlchemy sends it correctly
    vec_literal = f"[{','.join(str(v) for v in query_vector)}]"

    sql = text("""
        SELECT
            id,
            repository_id,
            file_path,
            start_line,
            end_line,
            content,
            language,
            (embedding <=> CAST(:query_vec AS vector)) AS distance
        FROM code_chunks
        WHERE repository_id = :repo_id
        ORDER BY distance ASC
        LIMIT :limit
    """)

    rows = db.execute(sql, {
        "query_vec": vec_literal,
        "repo_id":   repository_id,
        "limit":     limit,
    }).fetchall()

    return [dict(row._mapping) for row in rows]


def _keyword_search(
    db: Session,
    repository_id: int,
    query: str,
    limit: int,
) -> list[dict]:
    """Run full-text search using PostgreSQL tsvector/tsquery.

    Uses plainto_tsquery() for robust handling of code-related queries
    (strips punctuation, handles single terms gracefully).

    Requires: GIN index on to_tsvector('english', content)
    (created by the lifespan handler in main.py on startup).

    Args:
        db:            Active SQLAlchemy session.
        repository_id: Restrict search to this repository.
        query:         Natural language or keyword query.
        limit:         Max rows to return.

    Returns:
        List of row dicts ordered by descending ts_rank.
    """
    sql = text("""
        SELECT
            id,
            repository_id,
            file_path,
            start_line,
            end_line,
            content,
            language,
            ts_rank(
                to_tsvector('english', content),
                plainto_tsquery('english', :query)
            ) AS fts_rank
        FROM code_chunks
        WHERE
            repository_id = :repo_id
            AND to_tsvector('english', content) @@ plainto_tsquery('english', :query)
        ORDER BY fts_rank DESC
        LIMIT :limit
    """)

    rows = db.execute(sql, {
        "query":   query,
        "repo_id": repository_id,
        "limit":   limit,
    }).fetchall()

    return [dict(row._mapping) for row in rows]


def _rrf_fusion(
    semantic_results: list[dict],
    keyword_results: list[dict],
    k: int = RRF_K,
) -> list[SearchResult]:
    """Fuse semantic and keyword rankings via Reciprocal Rank Fusion.

    RRF is rank-based: it does not require semantic distances and FTS scores
    to be on the same scale. Each document's score is the sum of its
    reciprocal rank positions across both result lists:

        score(d) = Σ_i  1 / (k + rank_i(d))

    Documents absent from a list receive no contribution from that list.

    Args:
        semantic_results: Rows from _semantic_search() (ordered by similarity).
        keyword_results:  Rows from _keyword_search() (ordered by ts_rank).
        k:                RRF constant (standard default: 60).

    Returns:
        Merged, RRF-ranked list of SearchResult objects.
    """
    # Ordered chunk ID lists (list position = zero-based rank)
    semantic_ids: list[int] = [r["id"] for r in semantic_results]
    keyword_ids:  list[int] = [r["id"] for r in keyword_results]

    # Build a row data lookup keyed by chunk ID (semantic takes precedence for ties)
    chunk_data: dict[int, dict] = {}
    for row in semantic_results + keyword_results:
        chunk_data.setdefault(row["id"], row)

    # Score every unique chunk
    results: list[SearchResult] = []
    for chunk_id, row in chunk_data.items():
        rrf_score = 0.0
        if chunk_id in semantic_ids:
            rrf_score += 1.0 / (k + semantic_ids.index(chunk_id) + 1)
        if chunk_id in keyword_ids:
            rrf_score += 1.0 / (k + keyword_ids.index(chunk_id) + 1)

        results.append(SearchResult(
            chunk_id=chunk_id,
            repository_id=row["repository_id"],
            file_path=row["file_path"],
            start_line=row["start_line"],
            end_line=row["end_line"],
            content=row["content"],
            language=row.get("language"),
            score=rrf_score,
        ))

    return sorted(results, key=lambda r: r.score, reverse=True)


def hybrid_search(
    db: Session,
    repository_id: int,
    query: str,
    limit: int = 20,
) -> list[SearchResult]:
    """Run hybrid retrieval: semantic + full-text search, fused via RRF.

    This is the primary retrieval function for the Phase 3 RAG pipeline.
    It embeds the query once, runs both retrievers, and returns a single
    fused ranked list. Downstream steps (reranking, LLM generation) refine it.

    Args:
        db:            Active SQLAlchemy session.
        repository_id: Repository to search within.
        query:         Natural language question or keyword string.
        limit:         Candidate count per retriever (before fusion).
                       Higher values -> better recall; reranker handles precision.

    Returns:
        RRF-ranked list of SearchResult objects (up to 2*limit, deduplicated).
    """
    logger.info(
        "Hybrid search: repo=%d  query=%r  limit=%d",
        repository_id, query, limit,
    )

    # 1. Embed the query using the same BERT model used during indexing
    query_vector = embed_query(query)

    # 2. Run both retrievers independently
    semantic = _semantic_search(db, repository_id, query_vector, limit=limit)
    keyword  = _keyword_search(db, repository_id, query,         limit=limit)

    logger.info(
        "Retrieval: %d semantic  %d keyword  results",
        len(semantic), len(keyword),
    )

    # 3. Fuse rankings and return
    return _rrf_fusion(semantic, keyword)
