"""Unit tests for LangSmith evaluation, dataset synchronization, and evaluators."""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

# Ensure evals/scripts is in sys.path
EVALS_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if EVALS_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, EVALS_SCRIPTS_DIR)

from eval_agent import (
    evaluate_citation_presence,
    evaluate_keyword_grounding,
    evaluate_tool_selection,
    sync_evaluation_dataset,
)


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
