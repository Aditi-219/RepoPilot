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
import re
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.embedder import embed_query

logger = logging.getLogger(__name__)

# Standard RRF constant — empirically optimal at k=60
RRF_K: int = 60

# Domain and natural language stopwords excluded from FTS OR queries
STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "do", "does", "did",
    "how", "what", "where", "which", "why", "when", "who", "can", "could", "would",
    "should", "might", "to", "of", "in", "on", "for", "from", "and", "or", "by", "with",
    "it", "its", "this", "that", "these", "those", "as", "at", "into", "about", "if",
    "there", "their", "user", "code", "codebase", "repopilot",
}


def build_or_tsquery(question: str) -> str | None:
    """Build an OR-chained PostgreSQL tsquery string from a natural language or code query.

    Splits camelCase identifiers, removes punctuation, ignores stopwords and single-char tokens,
    and deduplicates terms to prevent syntax errors in to_tsquery().
    """
    # split camelCase, then keep only alphanumeric words
    text_val = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", question)
    words = re.findall(r"[A-Za-z0-9]+", text_val.lower())
    terms = []
    for w in words:
        if w in STOPWORDS or len(w) < 2:
            continue
        if w not in terms:
            terms.append(w)
    return " | ".join(terms) if terms else None


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

    Uses an OR-based tsquery (to_tsquery) built by build_or_tsquery()
    for high-recall code retrieval across identifiers and natural language terms.

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
    tsquery_str = build_or_tsquery(query)
    if not tsquery_str:
        return []

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
                to_tsquery('english', :tsquery)
            ) AS fts_rank
        FROM code_chunks
        WHERE
            repository_id = :repo_id
            AND to_tsvector('english', content) @@ to_tsquery('english', :tsquery)
        ORDER BY fts_rank DESC
        LIMIT :limit
    """)

    try:
        rows = db.execute(sql, {
            "tsquery": tsquery_str,
            "repo_id": repository_id,
            "limit":   limit,
        }).fetchall()
        return [dict(row._mapping) for row in rows]
    except Exception as exc:
        logger.warning("FTS query failed for tsquery '%s': %s", tsquery_str, exc)
        return []


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


def retrieve_all_stages(
    query: str,
    k_pool: int = 50,
    repository_id: int | None = None,
    db: Session | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Retrieve search results across all stages of the retrieval pipeline.

    Designed specifically for component-level evaluation of:
      1. Dense retrieval (pgvector cosine similarity)
      2. Sparse full-text search (PostgreSQL tsvector / ts_rank)
      3. Rank fusion (Reciprocal Rank Fusion)
      4. Reranking (Heuristic deduplication or Neural Cross-Encoder)

    Calls existing functions in sequence, preserving exact runtime retrieval behavior.

    Args:
        query:         Natural language or code identifier search query.
        k_pool:        Maximum candidates to retrieve/retain at each stage (default 50).
        repository_id: Target repository ID. If None, auto-resolves to the local 'repopilot'
                       record or first completed repository.
        db:            Active SQLAlchemy session. If None, uses an isolated SessionLocal().

    Returns:
        dict[str, list[dict[str, Any]]]: Dictionary mapping stage names to candidate lists.
        Each candidate dictionary contains:
          - 'file_path':  str
          - 'start_line': int
          - 'end_line':   int
          - 'chunk_id':   int
          - 'score':      float
          - 'content':    str
    """
    from app.database import SessionLocal
    from app.models import Repository
    from app.reranker import rerank

    def _execute(session: Session) -> dict[str, list[dict[str, Any]]]:
        nonlocal repository_id
        if repository_id is None:
            # First, prioritize the 'repopilot' evaluation repository
            repo = (
                session.query(Repository)
                .filter(Repository.full_name.ilike("%repopilot%"))
                .first()
            )
            # Fall back to any completed repository
            if repo is None:
                repo = (
                    session.query(Repository)
                    .filter(Repository.status == "completed")
                    .first()
                )
            if repo is None:
                raise ValueError("No indexed repository found in database. Please index a repository first.")
            resolved_repo_id = repo.id
        else:
            resolved_repo_id = repository_id

        # 1. Dense retrieval (pgvector cosine similarity)
        query_vector = embed_query(query)
        semantic_rows = _semantic_search(session, resolved_repo_id, query_vector, limit=k_pool)
        dense_results = [
            {
                "file_path": row["file_path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "chunk_id": row["id"],
                "score": round(1.0 - float(row["distance"]), 4),
                "content": row["content"],
            }
            for row in semantic_rows
        ]

        # 2. Sparse full-text search (tsvector / ts_rank)
        keyword_rows = _keyword_search(session, resolved_repo_id, query, limit=k_pool)
        fts_results = [
            {
                "file_path": row["file_path"],
                "start_line": row["start_line"],
                "end_line": row["end_line"],
                "chunk_id": row["id"],
                "score": round(float(row["fts_rank"]), 4),
                "content": row["content"],
            }
            for row in keyword_rows
        ]

        # 3. Reciprocal Rank Fusion (RRF)
        fused_candidates = _rrf_fusion(semantic_rows, keyword_rows, k=RRF_K)
        fused_results = [
            {
                "file_path": cand.file_path,
                "start_line": cand.start_line,
                "end_line": cand.end_line,
                "chunk_id": cand.chunk_id,
                "score": round(cand.score, 6),
                "content": cand.content,
            }
            for cand in fused_candidates[:k_pool]
        ]

        # 4. Reranking (Heuristic or Cross-Encoder depending on ENABLE_NEURAL_RERANKER)
        reranked_candidates = rerank(query, fused_candidates, top_k=k_pool)
        reranked_results = [
            {
                "file_path": cand.file_path,
                "start_line": cand.start_line,
                "end_line": cand.end_line,
                "chunk_id": cand.chunk_id,
                "score": round(cand.score, 6),
                "content": cand.content,
            }
            for cand in reranked_candidates
        ]

        return {
            "dense": dense_results,
            "fts": fts_results,
            "fused": fused_results,
            "reranked": reranked_results,
        }

    if db is not None:
        return _execute(db)
    with SessionLocal() as session:
        return _execute(session)

