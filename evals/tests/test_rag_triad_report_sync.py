"""Tests ensuring evals/RAG_TRIAD_REPORT.md and report_data_bundle.json match raw run JSON."""

from __future__ import annotations

import json
import os
import statistics
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

from generate_rag_triad_report import (
    BUNDLE_PATH,
    REMAINING_PATH,
    REPORT_PATH,
    RETRIEVER_FULL_PATH,
    RUN_PATH,
    build_bundle,
    render_report,
)


def test_report_matches_generated_output_exactly():
    """Fail if any number or section in evals/RAG_TRIAD_REPORT.md diverges from JSON data."""
    bundle = build_bundle()
    expected_md = render_report(bundle)

    assert os.path.exists(REPORT_PATH), "evals/RAG_TRIAD_REPORT.md is missing."
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        actual_md = f.read()

    assert actual_md == expected_md, (
        "evals/RAG_TRIAD_REPORT.md does not match the JSON-generated output. "
        "Re-run python evals/scripts/generate_rag_triad_report.py."
    )


def test_recompute_all_summary_numbers_from_raw_run_json():
    """Independently recompute all summary numbers from raw run JSON and compare with report_data_bundle.json."""
    with open(RUN_PATH, "r", encoding="utf-8") as f:
        raw_run = json.load(f)
    with open(REMAINING_PATH, "r", encoding="utf-8") as f:
        raw_rem = json.load(f)
    with open(RETRIEVER_FULL_PATH, "r", encoding="utf-8") as f:
        raw_ret = json.load(f)
    with open(BUNDLE_PATH, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    rows = raw_run["results"]
    assert len(rows) == bundle["dataset_stats"]["evaluated_count"] == 26

    # 1. Recompute RAG Triad + chunk_file_precision stats directly from raw results
    metric_map = {
        "context_relevancy": "context_relevancy_score",
        "chunk_file_precision": "chunk_file_precision",
        "faithfulness": "faithfulness_score",
        "answer_relevancy": "answer_relevancy_score",
    }
    for bundle_key, row_key in metric_map.items():
        vals = [float(r[row_key]) for r in rows]
        ts = bundle["triad_summary"][bundle_key]
        assert ts["n"] == len(vals)
        assert abs(ts["mean"] - round(statistics.mean(vals), 4)) <= 1e-4
        assert abs(ts["p50"] - round(statistics.median(vals), 4)) <= 1e-4
        assert abs(ts["std_dev"] - round(statistics.stdev(vals), 4)) <= 1e-4
        assert abs(ts["min"] - round(min(vals), 4)) <= 1e-4
        assert abs(ts["max"] - round(max(vals), 4)) <= 1e-4
        assert abs(ts["pass_rate"] - round(sum(1 for v in vals if v >= 0.7) / len(vals), 4)) <= 1e-4

    # 2. Recompute faithfulness strict-1.0 pass rate and unsupported-claim counts
    faith_vals = [float(r["faithfulness_score"]) for r in rows]
    strict_pass = sum(1 for v in faith_vals if v == 1.0)
    unsupported = sum(1 for v in faith_vals if v < 1.0)
    fa = bundle["faithfulness_audit"]
    assert fa["total"] == len(faith_vals)
    assert fa["threshold_1_0_pass_count"] == strict_pass
    assert fa["threshold_1_0_pass_rate"] == round(strict_pass / len(faith_vals), 4)
    assert fa["any_unsupported_claim_count"] == unsupported
    assert fa["any_unsupported_claim_pct"] == round(unsupported / len(faith_vals) * 100, 1)

    # 3. Recompute Contextual Precision & Recall excluding parse errors
    prec_vals = [
        d["contextual_precision"]
        for d in raw_rem["contextual_precision_recall"]["details"]
        if "invalid JSON" not in d.get("contextual_precision_reason", "")
    ]
    rec_vals = [
        d["contextual_recall"]
        for d in raw_rem["contextual_precision_recall"]["details"]
        if "invalid JSON" not in d.get("contextual_recall_reason", "")
    ]
    cp = bundle["contextual_metrics_clean"]["contextual_precision"]
    cr = bundle["contextual_metrics_clean"]["contextual_recall"]
    assert cp["n_valid"] == len(prec_vals)
    assert cp["mean"] == round(statistics.mean(prec_vals), 4)
    assert cp["p50"] == round(statistics.median(prec_vals), 4)
    assert cp["pass_rate"] == round(sum(1 for v in prec_vals if v >= 0.7) / len(prec_vals), 4)

    assert cr["n_valid"] == len(rec_vals)
    assert cr["mean"] == round(statistics.mean(rec_vals), 4)
    assert cr["p50"] == round(statistics.median(rec_vals), 4)
    assert cr["pass_rate"] == round(sum(1 for v in rec_vals if v >= 0.7) / len(rec_vals), 4)

    # 4. Recompute retriever k_pool=50 stats
    assert bundle["retriever_k50_reranked"]["hit_at_5"] == raw_ret["summary"]["reranked"]["Hit@5"]
    assert bundle["retriever_k50_reranked"]["recall_at_5"] == raw_ret["summary"]["reranked"]["Recall@5"]


def test_report_status_and_disclaimers():
    """Verify Limitations, collapsed tracks, Next Steps, and pool=15 vs pool=50 note."""
    with open(REPORT_PATH, "r", encoding="utf-8") as f:
        md = f.read()

    assert "judge-flagged, pending human review" in md
    assert "Pipeline retrieval (pool 15) and the component retriever baseline (pool 50) are different systems." in md
    assert "Retrieval recall favors `k=5`; `k=4` not yet tested on the triad." in md
    assert "## 2. Limitations & Incomplete Tracks" in md
    assert "## 6. Next Steps" in md
    assert "generations_baseline_k5.json` used `groq/openai/gpt-oss-120b`" in md
    assert "not_logged_in_130722" in md
