"""Unit tests for Phase 2 application-level quality judge & aggregation (0 network calls)."""

from __future__ import annotations

import datetime
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

import run_application_judge as raj  # noqa: E402


def _sample_thresholds():
    return raj.load_thresholds()


def test_deduplicate_keeps_latest_record():
    records = [
        {"question_id": "rp-001", "error": True, "error_message": "dns"},
        {"question_id": "rp-001", "error": False, "retried": True, "total_latency_s": 20.0},
    ]
    deduped = raj.deduplicate_records(records)
    assert len(deduped) == 1
    assert deduped[0]["error"] is False
    assert deduped[0]["retried"] is True


def test_error_row_tracking_in_ops_and_quality():
    records = [
        {"question_id": "rp-001", "error": False, "total_latency_s": 10.0},
        {"question_id": "rp-002", "error": True, "error_message": "503"},
    ]
    succ, errs = raj.split_success_and_errors(records)
    assert [r["question_id"] for r in succ] == ["rp-001"]
    assert [r["question_id"] for r in errs] == ["rp-002"]

    ops = raj.compute_ops_summary(records)
    assert ops["total_questions"] == 2
    assert ops["success_count"] == 1
    assert ops["error_count"] == 1
    assert ops["failed_ids"] == ["rp-002"]
    assert ops["availability_rate"] == 0.5
    assert ops["error_rate"] == 0.5

    agg = raj.aggregate_judge_results([], errs, _sample_thresholds())
    assert agg["error_rows_excluded_from_judge"] == [
        {"question_id": "rp-002", "error_message": "503"}
    ]


def test_latency_exclusion_of_retried_rows():
    records = [
        {
            "question_id": "rp-001", "error": False, "retried": True,
            "total_latency_s": 99.0, "time_to_first_tool_s": 9.0,
            "tool_durations": [{"tool": "read", "duration_s": 1.0}],
            "input_tokens": 1000, "output_tokens": 100,
        },
        {
            "question_id": "rp-002", "error": False, "retried": False,
            "total_latency_s": 10.0, "time_to_first_tool_s": 2.0,
            "tool_durations": [{"tool": "read", "duration_s": 3.0}],
            "input_tokens": 2000, "output_tokens": 200,
        },
    ]
    ops = raj.compute_ops_summary(records)
    assert ops["retried_excluded_from_latency"] == 1
    assert ops["p50_latency_s"] == 10.0
    assert ops["mean_tool_time_s"] == 3.0
    assert ops["mean_llm_time_s"] == 7.0
    assert ops["mean_input_tokens"] == 1500.0
    assert ops["max_input_tokens"] == 2000


def test_slice_aggregation():
    judged = [
        {
            "question_id": "rp-001", "category": "locate", "answerable": True,
            "final_answer": "In `backend/app/search.py`.",
            "metrics": {"correctness": {"score": 1.0}, "completeness": {"score": 0.9}, "style": {"score": 0.9}},
        },
        {
            "question_id": "rp-019", "category": "unanswerable", "answerable": False,
            "final_answer": "Redis is not used.",
            "metrics": {"correctness": {"score": 1.0}, "completeness": {"score": 1.0}, "style": {"score": 0.8}},
        },
        {
            "question_id": "rp-022", "category": "out_of_scope", "answerable": False,
            "final_answer": "I only answer codebase questions.",
            "metrics": {"correctness": {"score": 1.0}, "completeness": {"score": 1.0}, "style": {"score": 1.0}},
        },
    ]
    agg = raj.aggregate_judge_results(judged, [], _sample_thresholds())
    assert agg["slices"]["answerable"]["count"] == 1
    assert agg["slices"]["answerable"]["correctness"]["mean"] == 1.0
    assert agg["slices"]["unanswerable"]["count"] == 1
    assert agg["slices"]["out_of_scope"]["count"] == 1


def test_refusal_failure_detection():
    judged = [
        {
            "question_id": "rp-019", "category": "unanswerable", "answerable": False,
            "final_answer": "Redis is not used.",
            "metrics": {"correctness": {"score": 1.0}, "completeness": {"score": 1.0}, "style": {"score": 0.8}},
        },
        {
            "question_id": "rp-021", "category": "out_of_scope", "answerable": False,
            "final_answer": "Upon the surge of midnight's churning deep...",
            "metrics": {"correctness": {"score": 0.0}, "completeness": {"score": 0.0}, "style": {"score": 0.0}},
        },
    ]
    agg = raj.aggregate_judge_results(judged, [], _sample_thresholds())
    assert agg["refusal_pass_rate"] == 0.5
    assert len(agg["failing_items"]) == 1
    assert agg["failing_items"][0]["question_id"] == "rp-021"


def test_score_distribution_and_ceiling_flag():
    judged = [
        {"question_id": f"rp-{i}", "category": "locate", "answerable": True,
         "metrics": {"correctness": {"score": 1.0}, "completeness": {"score": 0.7}, "style": {"score": 0.5}}}
        for i in range(8)
    ] + [
        {"question_id": "rp-9", "category": "locate", "answerable": True,
         "metrics": {"correctness": {"score": 0.0}, "completeness": {"score": 0.3}, "style": {"score": 0.9}}}
    ]
    dist = raj.compute_score_distribution(judged, "answerable")
    assert dist["correctness"]["exact_1_0"] == 8
    assert dist["correctness"]["exact_0_0"] == 1
    assert dist["correctness"]["ceiling_effect_flag"] is True
    assert dist["completeness"]["ceiling_effect_flag"] is False
    assert dist["completeness"]["histogram"]["[0.6,0.8)"] == 8
    assert dist["completeness"]["histogram"]["[0.2,0.4)"] == 1


def test_noise_math():
    pass1 = [{"question_id": "rp-001", "metrics": {"correctness": {"score": 0.8}, "completeness": {"score": 0.8}, "style": {"score": 0.9}}}]
    pass2 = [{"question_id": "rp-001", "metrics": {"correctness": {"score": 1.0}, "completeness": {"score": 0.8}, "style": {"score": 0.7}}}]
    noise = raj.compute_judge_noise([pass1, pass2], noise_multiplier=2.0)
    assert noise["completeness"]["avg_std_dev"] == 0.0
    assert noise["correctness"]["avg_std_dev"] > 0.0
    assert noise["correctness"]["regression_threshold"] == round(noise["correctness"]["avg_std_dev"] * 2.0, 4)


def test_request_estimation():
    assert raj.estimate_judge_requests(32, passes=1, metrics_per_row=3) == 96
    assert raj.estimate_judge_requests(32, passes=2, metrics_per_row=3) == 192


def test_select_pilot_rows_and_sum_today_quota(tmp_path):
    rows = [
        {"question_id": "rp-002", "category": "locate", "answerable": True},
        {"question_id": "rp-003", "category": "locate", "answerable": True},
        {"question_id": "rp-019", "category": "unanswerable", "answerable": False},
        {"question_id": "rp-021", "category": "out_of_scope", "answerable": False},
    ]
    pilot = raj.select_pilot_rows(rows)
    assert [r["question_id"] for r in pilot] == ["rp-002", "rp-019", "rp-021"]

    ledger = tmp_path / "quota_ledger.jsonl"
    today = datetime.datetime.now(raj.PACIFIC_TZ).date().isoformat()
    ledger.write_text(
        json.dumps({"date": today, "quota_tz": "America/Los_Angeles", "requests_used": 10}) + "\n"
        + json.dumps({"date": "2020-01-01", "quota_tz": "America/Los_Angeles", "requests_used": 999}) + "\n"
        + json.dumps({"date": today, "quota_tz": "America/Los_Angeles", "requests_used": 5}) + "\n",
        encoding="utf-8",
    )
    assert raj.sum_today_quota(str(ledger)) == 15


def test_deterministic_style_checks_and_citation_formats():
    # Accepted formats
    assert raj.has_citation_with_lines("See `backend/app/search.py:L193-L245`.") is True
    assert raj.has_citation_with_lines("See [`backend/app/agent/state.py:L11-L25`](backend/app/agent/state.py).") is True

    # Must FAIL on bare path, bare line number, or code block without citation
    assert raj.has_citation_with_lines("See backend/app/search.py.") is False
    assert raj.has_citation_with_lines("See lines L14-L58 for details.") is False
    assert raj.has_citation_with_lines("```python\ndef foo():\n    return 1\n```") is False

    # Prose vs fenced code word split
    sample = "Here is how it works:\n```python\n" + ("x = 1\n" * 50) + "```\nShort summary."
    prose_w, code_w, total_w = raj.split_prose_and_code_words(sample)
    assert prose_w == 7
    assert code_w > 100
    assert total_w > 100

    res_good = raj.evaluate_deterministic_style(
        "Implemented in `backend/app/search.py:L193-L245` using `_rrf_fusion`.\n```python\n"
        + ("a = 1\n" * 350)
        + "```",
        word_budget=300,
    )
    assert res_good["has_citation_with_lines"] is True
    assert res_good["identifiers_in_backticks"] is True
    assert res_good["within_word_budget"] is True  # prose is < 300 even though total > 300
    assert res_good["no_banned_filler"] is True

    res_bad = raj.evaluate_deterministic_style("Great question! " + ("word " * 305), word_budget=300)
    assert res_bad["within_word_budget"] is False
    assert res_bad["no_banned_filler"] is False
    assert "great question" in res_bad["matched_filler"]


def test_completeness_from_reason_and_mismatch_detection():
    reason = (
        "1. `backend/app/search.py` [COVERED]\n"
        "2. `_rrf_fusion` function [COVERED]\n"
        "3. `RRF_K = 60` constant [MISSING]\n"
        "Score: 2/3."
    )
    score, cov, tot = raj.compute_completeness_from_reason(reason)
    assert (cov, tot) == (2, 3)
    assert score == round(2 / 3, 4)
    assert raj.check_completeness_mismatch(0.6667, score) is False
    assert raj.check_completeness_mismatch(1.0, score) is True
    assert raj.compute_completeness_from_reason("No flags here") == (None, 0, 0)


def test_literal_key_points_and_basename_and_wrong_path():
    kps = [
        {"text": "File `backend/app/search.py`", "kind": "literal", "match": ["backend/app/search.py"]},
        {"text": "Function `_rrf_fusion`", "kind": "literal", "match": ["_rrf_fusion"]},
        {"text": "Function `rerank`", "kind": "literal", "match": ["rerank"]},
        {"text": "Function `chunk_file`", "kind": "literal", "match": ["chunk_file"]},
        {"text": "Constant `RRF_K = 60`", "kind": "literal", "match": ["RRF_K = 60"], "aliases": ["k = 60"]},
        {"text": "HTTP `404`", "kind": "literal", "match": ["404"]},
        {"text": "Table `repositories`", "kind": "literal", "match": [], "raw_match": ["`repositories`", "'repositories'"]},
        {"text": "Explains rank fusion", "kind": "semantic", "match": []},
    ]

    # 1. Prefix collision: rerank must NOT match reranker.py/_neural_rerank; chunk_file must NOT match chunk_file_ast
    lit_prefix, _ = raj.check_literal_key_points("See backend/app/reranker.py and _neural_rerank and chunk_file_ast.", kps)
    by_txt = {x["text"]: x["covered"] for x in lit_prefix}
    assert by_txt["Function `rerank`"] is False
    assert by_txt["Function `chunk_file`"] is False

    # 2. Number inside a line citation: 404 must NOT match :L404 or L400-L404
    lit_num, _ = raj.check_literal_key_points("See backend/app/helpers.py:L400-L404 and L404.", kps)
    assert {x["text"]: x["covered"] for x in lit_num}["HTTP `404`"] is False

    # 3. Whitespace & backtick variants: `RRF_K` = 60 and RRF_K=60 both match
    for variant in ("`RRF_K` = 60", "RRF_K=60"):
        lit_ws, _ = raj.check_literal_key_points(f"Uses {variant}.", kps)
        assert {x["text"]: x["covered"] for x in lit_ws}["Constant `RRF_K = 60`"] is True

    # 4. Alias match: k = 60 matches
    lit_alias, _ = raj.check_literal_key_points("Uses k = 60.", kps)
    assert {x["text"]: x["covered"] for x in lit_alias}["Constant `RRF_K = 60`"] is True

    # 5. Basename-only ("search.py") MUST FAIL backend/app/search.py
    lit_base, sem_base = raj.check_literal_key_points("Implemented in search.py via _rrf_fusion.", kps)
    assert {x["text"]: x["covered"] for x in lit_base}["File `backend/app/search.py`"] is False
    assert len(sem_base) == 1

    # 6. Wrong path MUST FAIL
    lit_wrong, _ = raj.check_literal_key_points("In backend/app/nonexistent_fusion.py.", kps)
    assert {x["text"]: x["covered"] for x in lit_wrong}["File `backend/app/search.py`"] is False

    # 7. Generic raw_match requires backticks/quotes
    lit_raw_fail, _ = raj.check_literal_key_points("The repositories are indexed.", kps)
    assert {x["text"]: x["covered"] for x in lit_raw_fail}["Table `repositories`"] is False
    lit_raw_ok, _ = raj.check_literal_key_points("Maps to `repositories`.", kps)
    assert {x["text"]: x["covered"] for x in lit_raw_ok}["Table `repositories`"] is True


def test_is_strict_pass():
    assert raj.is_strict_pass({"metrics": {"correctness": {"score": 0.8}, "completeness": {"score": 1.0}}}) is True
    assert raj.is_strict_pass({"metrics": {"correctness": {"score": 1.0}, "completeness": {"score": 0.6667}}}) is False
    assert raj.is_strict_pass({"metrics": {"correctness": {"score": 0.79}, "completeness": {"score": 1.0}}}) is False
    assert raj.is_strict_pass({"metrics": {"correctness": {"score": None}, "completeness": {"score": 1.0}}}) is False
