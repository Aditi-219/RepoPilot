"""Unit tests for Phase 1 application-level generation harness (no network)."""

from __future__ import annotations

import json
import os
import sys

import pytest

# Make evals/scripts importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

import run_application_generate as rag  # noqa: E402


def test_extract_file_references_deduplicates_and_parses_paths():
    text = (
        "Check `backend/app/search.py` and [backend/app/search.py:L14-L58] "
        "as well as backend/app/reranker.py:L10. Ignore plain words or v1.2."
    )
    refs = rag._extract_file_references(text)
    assert refs == ["backend/app/search.py", "backend/app/reranker.py"]


def test_load_completed_ids_handles_missing_and_corrupt_lines(tmp_path):
    missing = tmp_path / "does_not_exist.jsonl"
    assert rag.load_completed_ids(str(missing)) == set()

    cache_file = tmp_path / "run.jsonl"
    cache_file.write_text(
        json.dumps({"question_id": "rp-001", "final_answer": "ok"}) + "\n"
        + "{not valid json\n"
        + json.dumps({"question_id": "rp-002", "final_answer": "ok2"}) + "\n",
        encoding="utf-8",
    )
    assert rag.load_completed_ids(str(cache_file)) == {"rp-001", "rp-002"}


def test_make_error_record_includes_required_fields():
    ex = {
        "id": "rp-099",
        "question": "How does X work?",
        "category": "architecture",
        "answerable": True,
        "gt_files": ["backend/app/main.py"],
        "gt_symbols": ["lifespan"],
        "ideal_answer": "It runs on startup.",
    }
    rec = rag._make_error_record(
        ex,
        sha="abc123",
        timestamp="20261126_120000",
        error_msg="timeout",
        model_name="qwen2.5-coder:1.5b",
        hardware_note="Windows 11 (AMD64, 8 CPUs)",
    )
    assert rec["question_id"] == "rp-099"
    assert rec["error"] is True
    assert rec["timeout"] is False
    assert rec["error_message"] == "timeout"
    assert rec["model_name"] == "qwen2.5-coder:1.5b"
    assert rec["hardware_note"] == "Windows 11 (AMD64, 8 CPUs)"
    assert rec["input_tokens"] is None
    assert rec["output_tokens"] is None
    assert rec["generator_config"] == {"model": "qwen2.5-coder:1.5b", "temperature": 0.2}
    assert rec["tool_durations"] == []
    assert rec["time_to_first_tool_s"] is None
    assert rec["llm_calls"] == 0


def test_stream_question_parses_usage_tool_durations_and_llm_calls():
    sse_lines = [
        'data: {"type": "tool_start", "tool": "inspect_file_tree", "args": {}}',
        'data: {"type": "tool_end", "tool": "inspect_file_tree", "output": "backend/app/search.py"}',
        'data: {"type": "tool_start", "tool": "read_source_file", "args": {"file_path": "backend/app/search.py"}}',
        'data: {"type": "tool_end", "tool": "read_source_file", "output": "def _rrf_fusion(): pass"}',
        'data: {"type": "token", "delta": "See `backend/app/search.py`."}',
        'data: {"type": "usage", "input_tokens": 1200, "output_tokens": 85}',
        'data: {"type": "done"}',
    ]

    class FakeResp:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def raise_for_status(self):
            return None

        def iter_lines(self):
            return iter(sse_lines)

    class FakeClient:
        def stream(self, *args, **kwargs):
            return FakeResp()

    res = rag._stream_question(FakeClient(), "http://localhost:8000", 8, "sess-1", "Where is RRF?")
    assert res["final_answer"] == "See `backend/app/search.py`."
    assert res["citations"] == ["backend/app/search.py"]
    assert len(res["retrieved_context"]) == 2
    assert len(res["tool_durations"]) == 2
    assert res["time_to_first_tool_s"] is not None
    assert res["llm_calls"] == 3  # 1 initial + 2 sequential tool rounds
    assert res["input_tokens"] == 1200
    assert res["output_tokens"] == 85
    assert res["error"] is False


def test_summarize_run_and_quota_ledger(tmp_path):
    records = [
        {
            "question_id": "rp-001", "category": "locate", "answerable": True,
            "error": False, "total_latency_s": 10.0, "time_to_first_tool_s": 2.0,
            "input_tokens": 1000, "output_tokens": 100, "llm_calls": 4,
        },
        {
            "question_id": "rp-019", "category": "unanswerable", "answerable": False,
            "error": False, "total_latency_s": 5.0, "time_to_first_tool_s": 1.0,
            "input_tokens": 500, "output_tokens": 50, "llm_calls": 2,
        },
        {
            "question_id": "rp-021", "category": "out_of_scope", "answerable": False,
            "error": True, "total_latency_s": None, "time_to_first_tool_s": None,
            "input_tokens": None, "output_tokens": None, "llm_calls": 1,
        },
    ]
    s = rag.summarize_run(records)
    assert s["total"] == 3
    assert s["success"] == 2
    assert s["errors"] == 1
    assert s["failed_ids"] == ["rp-021"]
    assert s["total_requests"] == 7
    assert s["p50_latency_s"] == 7.5
    assert s["mean_input_tokens"] == 750.0
    assert s["max_input_tokens"] == 1000
    assert s["slices"]["answerable"]["success"] == 1
    assert s["slices"]["unanswerable"]["success"] == 1
    assert s["slices"]["out_of_scope"]["errors"] == 1

    ledger = tmp_path / "quota_ledger.jsonl"
    rag.append_quota_ledger("run_application_generate.py", "gemini-3.5-flash-lite", 7, "out.jsonl", str(ledger))
    entry = json.loads(ledger.read_text(encoding="utf-8").strip())
    assert entry["requests_used"] == 7
    assert entry["script"] == "run_application_generate.py"


def test_dry_run_does_not_write_file_or_call_network(tmp_path, capsys):
    dataset = tmp_path / "mini.jsonl"
    dataset.write_text(
        json.dumps({"id": "rp-001", "question": "Q1?", "answerable": True}) + "\n"
        + json.dumps({"id": "rp-002", "question": "Q2?", "answerable": False}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "out.jsonl"
    returned = rag.run_generation(
        dataset_path=str(dataset),
        output_path=str(out),
        limit=1,
        dry_run=True,
        model_name="test-model",
        hardware_note="test-hw",
    )
    assert returned == str(out)
    assert not out.exists()
    captured = capsys.readouterr().out
    assert "DRY RUN" in captured
    assert "rp-001" in captured
    assert "rp-002" not in captured
