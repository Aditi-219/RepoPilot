"""Unit tests for RepoPilot indexer pipeline with concurrent downloading."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.indexer import index_repository
from app.models import Repository


@pytest.mark.asyncio
async def test_index_repository_unknown_id():
    """index_repository should exit cleanly when repo_id does not exist."""
    mock_db = MagicMock()
    mock_db.get.return_value = None

    with patch("app.indexer.SessionLocal", return_value=mock_db):
        mock_db.__enter__.return_value = mock_db
        # Should not raise exception
        await index_repository(9999)
        mock_db.commit.assert_not_called()


@pytest.mark.asyncio
async def test_index_repository_empty_repo_marks_completed():
    """A repository with no indexable files should be marked completed."""
    mock_db = MagicMock()
    mock_repo = MagicMock(spec=Repository)
    mock_repo.id = 1
    mock_repo.full_name = "test/empty"
    mock_repo.url = "https://github.com/test/empty"
    mock_repo.status = "pending"
    mock_db.get.return_value = mock_repo

    with (
        patch("app.indexer.SessionLocal", return_value=mock_db),
        patch("app.indexer.parse_github_url", return_value=("test", "empty")),
        patch("app.indexer.fetch_repo_tree", new_callable=AsyncMock, return_value=[]),
    ):
        mock_db.__enter__.return_value = mock_db
        await index_repository(1)
        assert mock_repo.status == "completed"


@pytest.mark.asyncio
async def test_index_repository_concurrent_fetch_and_batch_insert():
    """Verify concurrent file fetching, chunking, and batch DB insertion."""
    mock_db = MagicMock()
    mock_repo = MagicMock(spec=Repository)
    mock_repo.id = 42
    mock_repo.full_name = "test/project"
    mock_repo.url = "https://github.com/test/project"
    mock_repo.status = "pending"
    mock_db.get.return_value = mock_repo

    tree = [
        {"path": f"module_{i}.py", "size": 100}
        for i in range(5)
    ]

    async def fake_fetch(owner, repo, path):
        return f"# Code for {path}\ndef func_{path.replace('.', '_')}():\n    return 1\n"

    with (
        patch("app.indexer.SessionLocal", return_value=mock_db),
        patch("app.indexer.parse_github_url", return_value=("test", "project")),
        patch("app.indexer.fetch_repo_tree", new_callable=AsyncMock, return_value=tree),
        patch("app.indexer.fetch_file_content", side_effect=fake_fetch),
        patch("app.indexer.embed_chunks", return_value=[[0.1] * 384] * 5),
    ):
        mock_db.__enter__.return_value = mock_db
        await index_repository(42)

        # Assert status is completed
        assert mock_repo.status == "completed"
        # Assert records were added to db
        assert mock_db.add_all.called
