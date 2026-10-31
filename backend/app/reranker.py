"""Reranker module for RepoPilot Phase 3 RAG pipeline.

Orders retrieved candidates for precision. Supports two modes:

- **Heuristic mode** (default): fast deduplication + RRF rank order preserved.
  Suitable for local CPU-only setups with no extra dependencies.

- **Neural mode** (opt-in via ENABLE_NEURAL_RERANKER=true): scores each
  (query, chunk) pair through a cross-encoder model
  (cross-encoder/ms-marco-MiniLM-L-6-v2) and re-sorts by the raw logit score.
  Requires sentence-transformers >= 2.2 and sufficient RAM/CPU (or GPU).
  Adds ~200–500 ms latency on CPU for a set of 20 candidates.

Set ENABLE_NEURAL_RERANKER=true in your .env to activate neural reranking.
"""

import logging
import os

from app.search import SearchResult

logger = logging.getLogger(__name__)

# Read once at module load; avoids re-reading the env var on every call.
_NEURAL_RERANKER_ENABLED: bool = os.getenv("ENABLE_NEURAL_RERANKER", "false").lower() == "true"

# Model name for the cross-encoder. This is a small, fast model that works well
# for code retrieval tasks (trained on MS MARCO passage ranking).
_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Lazy-loaded cross-encoder — only initialised when neural mode is first used.
_cross_encoder = None


def _get_cross_encoder():
    """Lazily load the cross-encoder model on first use.

    Returns:
        A sentence_transformers.CrossEncoder instance, or None on import error.
    """
    global _cross_encoder  # noqa: PLW0603
    if _cross_encoder is not None:
        return _cross_encoder

    try:
        from sentence_transformers import CrossEncoder  # noqa: PLC0415

        logger.info("Loading neural cross-encoder: %s", _CROSS_ENCODER_MODEL)
        _cross_encoder = CrossEncoder(_CROSS_ENCODER_MODEL)
        logger.info("Neural cross-encoder loaded successfully.")
    except Exception as exc:
        logger.warning(
            "Could not load cross-encoder model '%s' (falling back to heuristic): %s",
            _CROSS_ENCODER_MODEL,
            exc,
        )
        _cross_encoder = None

    return _cross_encoder


def _neural_rerank(
    query: str,
    candidates: list[SearchResult],
    top_k: int,
) -> list[SearchResult]:
    """Score candidates with a cross-encoder and return the top-k by score.

    Args:
        query:      The user's search query.
        candidates: Pre-filtered list of SearchResult objects to score.
        top_k:      Maximum number of results to return.

    Returns:
        Candidates sorted by descending cross-encoder logit score, truncated to top_k.
        Falls back to heuristic ordering if the model is unavailable.
    """
    encoder = _get_cross_encoder()
    if encoder is None:
        logger.warning("Cross-encoder unavailable; falling back to heuristic reranking.")
        return candidates[:top_k]

    pairs = [(query, r.content) for r in candidates]
    try:
        scores = encoder.predict(pairs)
    except Exception as exc:
        logger.warning("Cross-encoder prediction failed (falling back): %s", exc)
        return candidates[:top_k]

    for result, score in zip(candidates, scores, strict=False):
        result.score = float(score)

    ranked = sorted(candidates, key=lambda r: r.score, reverse=True)
    logger.info(
        "Neural reranker scored %d candidates; returning top %d.",
        len(ranked),
        top_k,
    )
    return ranked[:top_k]


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 8,
) -> list[SearchResult]:
    """Rerank retrieval candidates and return the top-k results.

    Uses neural cross-encoder scoring when ENABLE_NEURAL_RERANKER=true;
    otherwise applies fast heuristic deduplication preserving RRF order.

    Args:
        query:   The user's search query (used only by the neural path).
        results: RRF-ranked list of SearchResult objects from hybrid_search().
        top_k:   Maximum number of results to return.

    Returns:
        Up to top_k SearchResult objects ordered by relevance.
    """
    if not results:
        return []

    # Deduplicate by (file_path, start_line, end_line) to remove exact duplicates
    # that can appear when semantic and keyword searches retrieve the same chunk.
    seen: set[tuple[str, int, int]] = set()
    unique_results: list[SearchResult] = []
    for r in results:
        key = (r.file_path, r.start_line, r.end_line)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    if _NEURAL_RERANKER_ENABLED:
        return _neural_rerank(query, unique_results, top_k)

    # Hybrid RRF rank order is already high quality; just truncate.
    return unique_results[:top_k]
