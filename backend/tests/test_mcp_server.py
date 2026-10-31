"""Unit tests for RepoPilot MCP Server."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.mcp_server import (
    ask_codebase,
    fetch_repo_metadata,
    inspect_file_tree,
    mcp_app,
    read_source_file,
    search_codebase,
)


@pytest.mark.asyncio
async def test_mcp_server_registers_all_tools():
    """Verify MCP Server lists all 5 tools."""
    tools = await mcp_app.list_tools()
    tool_names = {t.name for t in tools}
    expected = {
        "fetch_repo_metadata",
        "inspect_file_tree",
        "read_source_file",
        "search_codebase",
        "ask_codebase",
    }
    assert expected.issubset(tool_names)


@pytest.mark.asyncio
async def test_mcp_fetch_repo_metadata():
    """Verify MCP metadata tool invokes fetch_repository."""
    mock_data = {
        "full_name": "tiangolo/fastapi",
        "stars": 75000,
        "language": "Python",
        "description": "FastAPI framework",
        "url": "https://github.com/tiangolo/fastapi",
    }
    with patch("app.mcp_server.fetch_repository", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_data
        result = await fetch_repo_metadata("tiangolo", "fastapi")
        assert result["full_name"] == "tiangolo/fastapi"
        assert result["stars"] == 75000


@pytest.mark.asyncio
async def test_mcp_inspect_file_tree_filters():
    """Verify MCP file tree tool filters by path prefix."""
    sample_files = [
        {"path": "app/main.py", "size": 1000},
        {"path": "app/utils.py", "size": 500},
        {"path": "frontend/index.html", "size": 200},
    ]
    with patch("app.mcp_server.fetch_repo_tree", new_callable=AsyncMock) as mock_tree:
        mock_tree.return_value = sample_files

        filtered = await inspect_file_tree("owner", "repo", prefix="app")
        assert len(filtered) == 2
        assert all(f["path"].startswith("app/") for f in filtered)


@pytest.mark.asyncio
async def test_mcp_read_source_file_formatting():
    """Verify MCP file reader numbers lines and limits bounds."""
    content = "import os\nimport sys\n\ndef main():\n    pass\n"
    with patch("app.mcp_server.fetch_file_content", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = content
        result = await read_source_file("owner", "repo", "main.py", start_line=1, end_line=3)
        assert "File: main.py (Lines 1-3 of 5)" in result
        assert "   1 | import os" in result
        assert "   2 | import sys" in result


def test_mcp_search_codebase_unindexed_repo():
    """Searching an unindexed repository returns an informative error object."""
    mock_db = MagicMock()
    mock_repo = MagicMock()
    mock_repo.status = "indexing"
    mock_repo.full_name = "test/repo"
    mock_db.get.return_value = mock_repo

    with patch("app.mcp_server.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        res = search_codebase(repository_id=999, query="test")
        assert len(res) == 1
        assert "not indexed yet" in res[0]["error"]


@pytest.mark.asyncio
async def test_mcp_ask_codebase_unindexed_repo():
    """Asking questions on an unindexed repository returns a safe refusal."""
    mock_db = MagicMock()
    mock_repo = MagicMock()
    mock_repo.status = "pending"
    mock_db.get.return_value = mock_repo

    with patch("app.mcp_server.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        answer = await ask_codebase(repository_id=999, query="How does it work?")
        assert "No indexed code found" in answer
