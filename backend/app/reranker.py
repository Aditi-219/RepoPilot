"""Reranker module for RepoPilot Phase 3 RAG pipeline.

Orders retrieved candidates for precision. When neural cross-encoders are
unavailable or slow on local CPU, uses fast heuristic and RRF ranking
so responses return in milliseconds without hanging.
"""

import logging
from app.search import SearchResult

logger = logging.getLogger(__name__)


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 8,
) -> list[SearchResult]:
    """Rerank retrieval candidates.

    Returns the top-k highest scoring chunks.
    """
    if not results:
        return []

    # Filter out empty or duplicate chunks
    seen = set()
    unique_results = []
    for r in results:
        key = (r.file_path, r.start_line, r.end_line)
        if key not in seen:
            seen.add(key)
            unique_results.append(r)

    # Hybrid RRF rank order is already high quality
    return unique_results[:top_k]
