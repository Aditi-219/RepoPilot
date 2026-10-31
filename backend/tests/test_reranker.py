"""Unit tests for RepoPilot reranker module."""

from unittest.mock import MagicMock, patch

from app.reranker import _neural_rerank, rerank
from app.search import SearchResult


def _make_result(chunk_id: int, file_path: str, start: int, end: int, score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        repository_id=1,
        file_path=file_path,
        start_line=start,
        end_line=end,
        content=f"content from {file_path}",
        language="python",
        score=score,
    )


def test_rerank_empty_results():
    """Reranking an empty list returns empty list."""
    assert rerank("query", []) == []


def test_rerank_deduplicates_identical_chunks():
    """Identical file_path and line range should be deduplicated."""
    chunk1 = _make_result(1, "main.py", 1, 10, score=0.9)
    chunk2 = _make_result(2, "main.py", 1, 10, score=0.8)  # duplicate bounds
    chunk3 = _make_result(3, "main.py", 11, 20, score=0.7)

    results = rerank("test", [chunk1, chunk2, chunk3], top_k=5)
    assert len(results) == 2
    assert results[0].start_line == 1
    assert results[1].start_line == 11


def test_rerank_top_k_truncation():
    """Rerank should truncate results to requested top_k."""
    chunks = [_make_result(i, f"file_{i}.py", 1, 10, score=float(i)) for i in range(10)]
    results = rerank("test", chunks, top_k=4)
    assert len(results) == 4


def test_neural_rerank_with_mocked_cross_encoder():
    """Neural reranker should sort candidates by predicted cross-encoder scores."""
    chunks = [
        _make_result(1, "low.py", 1, 10, score=0.1),
        _make_result(2, "high.py", 1, 10, score=0.2),
    ]

    mock_encoder = MagicMock()
    # Predict higher score for low.py than high.py
    mock_encoder.predict.return_value = [0.95, 0.15]

    with patch("app.reranker._get_cross_encoder", return_value=mock_encoder):
        ranked = _neural_rerank("test query", chunks, top_k=2)
        assert len(ranked) == 2
        assert ranked[0].file_path == "low.py"
        assert ranked[0].score == 0.95
        assert ranked[1].file_path == "high.py"
        assert ranked[1].score == 0.15
