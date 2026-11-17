"""Unit tests for RepoPilot search and retrieval multi-stage evaluation."""

from unittest.mock import MagicMock, patch

from app.search import (
    _rrf_fusion,
    build_or_tsquery,
    retrieve_all_stages,
)


def test_build_or_tsquery():
    # Standard query with stopwords and casing
    result = build_or_tsquery("How does the RRF function work?")
    assert result == "rrf | function | work"

    # Query with special characters and punctuation that would break to_tsquery
    special_query = "What's the (fastest) way to compute 'mrr' score: [x > 0]?"
    result_special = build_or_tsquery(special_query)
    assert result_special is not None
    assert "fastest" in result_special
    assert "compute" in result_special
    assert "mrr" in result_special
    assert "score" in result_special
    # Verify no special characters remain
    assert all(c.isalnum() or c in (" ", "|") for c in result_special)

    # All stopwords returns None
    assert build_or_tsquery("what is the do are") is None


def test_rrf_fusion_ordering():
    semantic_rows = [
        {"id": 1, "repository_id": 1, "file_path": "a.py", "start_line": 1, "end_line": 10, "content": "foo"},
        {"id": 2, "repository_id": 1, "file_path": "b.py", "start_line": 1, "end_line": 10, "content": "bar"},
    ]
    keyword_rows = [
        {"id": 2, "repository_id": 1, "file_path": "b.py", "start_line": 1, "end_line": 10, "content": "bar"},
        {"id": 3, "repository_id": 1, "file_path": "c.py", "start_line": 1, "end_line": 10, "content": "baz"},
    ]

    fused = _rrf_fusion(semantic_rows, keyword_rows, k=60)
    assert len(fused) == 3
    # Chunk 2 appears in both semantic and keyword results, so it should rank highest
    assert fused[0].chunk_id == 2


def test_retrieve_all_stages_structure():
    mock_db = MagicMock()

    semantic_return = [
        {"id": 10, "repository_id": 1, "file_path": "backend/app/search.py", "start_line": 1, "end_line": 30, "distance": 0.25, "content": "def hybrid_search(): pass"},
    ]
    keyword_return = [
        {"id": 10, "repository_id": 1, "file_path": "backend/app/search.py", "start_line": 1, "end_line": 30, "fts_rank": 0.85, "content": "def hybrid_search(): pass"},
        {"id": 11, "repository_id": 1, "file_path": "backend/app/indexer.py", "start_line": 1, "end_line": 20, "fts_rank": 0.40, "content": "def index(): pass"},
    ]

    with (
        patch("app.search.embed_query", return_value=[0.1] * 384),
        patch("app.search._semantic_search", return_value=semantic_return),
        patch("app.search._keyword_search", return_value=keyword_return),
    ):
        stages = retrieve_all_stages(
            query="hybrid search",
            k_pool=10,
            repository_id=1,
            db=mock_db,
        )

    assert "dense" in stages
    assert "fts" in stages
    assert "fused" in stages
    assert "reranked" in stages

    for _stage_name, items in stages.items():
        assert isinstance(items, list)
        for item in items:
            assert "file_path" in item
            assert "start_line" in item
            assert "end_line" in item
            assert "chunk_id" in item
            assert "score" in item

    assert stages["dense"][0]["chunk_id"] == 10
    assert stages["dense"][0]["score"] == 0.75  # 1.0 - 0.25
    assert stages["fts"][0]["chunk_id"] == 10
    assert stages["fts"][0]["score"] == 0.85
