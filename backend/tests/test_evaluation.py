"""Unit tests for LangSmith evaluation, dataset synchronization, and tool hardening."""

from unittest.mock import AsyncMock, patch

import pytest
from app.agent.evaluation import (
    evaluate_citation_presence,
    evaluate_keyword_grounding,
    evaluate_tool_selection,
    sync_evaluation_dataset,
)
from app.tools.github_tools import fetch_repo_metadata, inspect_file_tree, read_source_file
from fastapi import HTTPException


class DummyRun:
    def __init__(self, outputs: dict):
        self.outputs = outputs


class DummyExample:
    def __init__(self, outputs: dict):
        self.outputs = outputs


def test_evaluate_tool_selection_full_match():
    run = DummyRun({"messages": [{"role": "assistant", "tool_calls": [{"name": "inspect_file_tree"}]}]})
    example = DummyExample({"expected_tools": ["inspect_file_tree"]})
    res = evaluate_tool_selection(run, example)
    assert res["score"] == 1.0


def test_evaluate_tool_selection_mismatch():
    run = DummyRun({"messages": [{"role": "assistant", "tool_calls": [{"name": "fetch_repo_metadata"}]}]})
    example = DummyExample({"expected_tools": ["inspect_file_tree"]})
    res = evaluate_tool_selection(run, example)
    assert res["score"] == 0.0


def test_evaluate_keyword_grounding_matches():
    run = DummyRun({"messages": [{"role": "assistant", "content": "FastAPI is a modern Python web framework"}]})
    example = DummyExample({"expected_keywords": ["python", "fastapi"]})
    res = evaluate_keyword_grounding(run, example)
    assert res["score"] == 1.0


def test_evaluate_keyword_grounding_partial():
    run = DummyRun({"messages": [{"role": "assistant", "content": "FastAPI is awesome"}]})
    example = DummyExample({"expected_keywords": ["fastapi", "uvicorn"]})
    res = evaluate_keyword_grounding(run, example)
    assert res["score"] == 0.5


def test_evaluate_citation_presence_detected():
    run = DummyRun({"messages": [{"role": "assistant", "content": "Defined in app/main.py at Lines 1-20."}]})
    example = DummyExample({"requires_citation": True})
    res = evaluate_citation_presence(run, example)
    assert res["score"] == 1.0


def test_evaluate_citation_presence_missing():
    run = DummyRun({"messages": [{"role": "assistant", "content": "I could not locate any files."}]})
    example = DummyExample({"requires_citation": True})
    res = evaluate_citation_presence(run, example)
    assert res["score"] == 0.0


def test_sync_evaluation_dataset_without_api_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    result = sync_evaluation_dataset("test-dataset")
    assert result is None


@pytest.mark.asyncio
async def test_fetch_repo_metadata_catches_http_error():
    with patch("app.tools.github_tools.fetch_repository", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.side_effect = HTTPException(status_code=404, detail="Repo not found")
        res = await fetch_repo_metadata.ainvoke({"owner": "nonexistent", "repo": "repo"})
        assert "error" in res
        assert res["status_code"] == 404


@pytest.mark.asyncio
async def test_inspect_file_tree_catches_rate_limit():
    with patch("app.tools.github_tools.fetch_repo_tree", new_callable=AsyncMock) as mock_tree:
        mock_tree.side_effect = HTTPException(status_code=403, detail="Rate limit exceeded")
        res = await inspect_file_tree.ainvoke({"owner": "owner", "repo": "repo"})
        assert isinstance(res, list)
        assert len(res) == 1
        assert "error" in res[0]
        assert res[0]["status_code"] == 403


@pytest.mark.asyncio
async def test_read_source_file_catches_timeout():
    with patch("app.tools.github_tools.fetch_file_content", new_callable=AsyncMock) as mock_content:
        mock_content.side_effect = HTTPException(status_code=504, detail="GitHub timed out")
        res = await read_source_file.ainvoke({"owner": "owner", "repo": "repo", "file_path": "test.py"})
        assert "Error reading file 'test.py': GitHub timed out" in res
