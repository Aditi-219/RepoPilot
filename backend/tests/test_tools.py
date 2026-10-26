"""Unit tests for LangGraph GitHub tools."""

from unittest.mock import AsyncMock, patch

import pytest
from app.tools.github_tools import fetch_repo_metadata, inspect_file_tree, read_source_file


@pytest.mark.asyncio
async def test_fetch_repo_metadata_invokes_service():
    """Verify fetch_repo_metadata constructs URL and delegates to fetch_repository."""
    mock_metadata = {
        "github_id": 12345,
        "full_name": "tiangolo/fastapi",
        "url": "https://github.com/tiangolo/fastapi",
        "description": "FastAPI framework",
        "stars": 75000,
        "language": "Python",
    }
    with patch("app.tools.github_tools.fetch_repository", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_metadata
        result = await fetch_repo_metadata.ainvoke({"owner": "tiangolo", "repo": "fastapi"})

        mock_fetch.assert_awaited_once_with("https://github.com/tiangolo/fastapi")
        assert result["github_id"] == 12345
        assert result["full_name"] == "tiangolo/fastapi"


@pytest.mark.asyncio
async def test_inspect_file_tree_filters_by_prefix():
    """Verify inspect_file_tree correctly filters paths by prefix."""
    sample_tree = [
        {"path": "backend/app/main.py", "size": 1024},
        {"path": "backend/app/routers/rag.py", "size": 2048},
        {"path": "frontend/src/App.jsx", "size": 512},
        {"path": "README.md", "size": 256},
    ]
    with patch("app.tools.github_tools.fetch_repo_tree", new_callable=AsyncMock) as mock_tree:
        mock_tree.return_value = sample_tree

        # Without prefix: returns all entries
        all_files = await inspect_file_tree.ainvoke({"owner": "owner", "repo": "repo", "prefix": ""})
        assert len(all_files) == 4

        # With prefix 'backend/app': returns only backend app entries
        backend_files = await inspect_file_tree.ainvoke({
            "owner": "owner",
            "repo": "repo",
            "prefix": "backend/app",
        })
        assert len(backend_files) == 2
        assert all(f["path"].startswith("backend/app/") for f in backend_files)

        # With prefix 'frontend': returns only frontend entries
        frontend_files = await inspect_file_tree.ainvoke({
            "owner": "owner",
            "repo": "repo",
            "prefix": "frontend",
        })
        assert len(frontend_files) == 1
        assert frontend_files[0]["path"] == "frontend/src/App.jsx"


@pytest.mark.asyncio
async def test_read_source_file_success_and_line_slicing():
    """Verify read_source_file slices content and adds line numbers."""
    fake_code = "\n".join([f"line_{i}" for i in range(1, 21)])

    with patch("app.tools.github_tools.fetch_file_content", new_callable=AsyncMock) as mock_content:
        mock_content.return_value = fake_code

        # Request lines 5 to 10
        result = await read_source_file.ainvoke({
            "owner": "owner",
            "repo": "repo",
            "file_path": "main.py",
            "start_line": 5,
            "end_line": 10,
        })

        assert "Lines 5-10 of 20" in result
        assert "5 | line_5" in result
        assert "10 | line_10" in result
        assert "1 | line_1" not in result
        assert "11 | line_11" not in result


@pytest.mark.asyncio
async def test_read_source_file_empty_and_missing():
    """Verify handling when a file is empty or cannot be fetched."""
    with patch("app.tools.github_tools.fetch_file_content", new_callable=AsyncMock) as mock_content:
        # None (binary / missing / error)
        mock_content.return_value = None
        error_result = await read_source_file.ainvoke({
            "owner": "owner",
            "repo": "repo",
            "file_path": "missing.py",
        })
        assert "Error: Could not read or decode file" in error_result

        # Empty content
        mock_content.return_value = ""
        empty_result = await read_source_file.ainvoke({
            "owner": "owner",
            "repo": "repo",
            "file_path": "empty.py",
        })
        assert "empty.py' is empty" in empty_result
