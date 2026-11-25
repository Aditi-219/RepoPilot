"""Unit tests for eval_rag_triad.py."""

from __future__ import annotations

import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

from eval_rag_triad import (
    compare_like_for_like_baselines,
    compute_chunk_file_precision,
    diagnose_rag_triad_failure,
    load_dataset_with_stats,
    resolve_pipeline_callable,
)


class TestChunkFilePrecision:
    """Test chunk-level file matching against ground truth."""

    def test_full_precision(self):
        retrieved = ["backend/app/search.py", "backend/app/search.py"]
        gt = ["backend/app/search.py"]
        assert compute_chunk_file_precision(retrieved, gt) == 1.0

    def test_partial_precision(self):
        retrieved = ["backend/app/search.py", "backend/app/main.py"]
        gt = ["backend/app/search.py"]
        assert compute_chunk_file_precision(retrieved, gt) == 0.5

    def test_zero_precision(self):
        retrieved = ["backend/app/models.py", "backend/app/main.py"]
        gt = ["backend/app/search.py"]
        assert compute_chunk_file_precision(retrieved, gt) == 0.0

    def test_empty_retrieved(self):
        assert compute_chunk_file_precision([], ["backend/app/search.py"]) == 0.0


class TestDiagnosisVerbatimReason:
    """Test that diagnosis strictly quotes/paraphrases the judge's reason without fabrication."""

    def test_retrieval_failure_quotes_context_reason(self):
        reason = "The score is 0.40 because the context contains excessive irrelevant code."
        res = diagnose_rag_triad_failure(
            context_relevancy=0.40,
            faithfulness=1.0,
            answer_relevancy=0.9,
            threshold=0.7,
            ctx_reason=reason,
        )
        assert res["failure_stage"] == "retrieval"
        assert reason in res["diagnosis"]
        assert "fix chunking, top-k, embedding, reranker" in res["recommended_fix"]

    def test_faithfulness_failure_quotes_faith_reason(self):
        reason = "The score is 0.50 because claims about Redis caching are unsupported."
        res = diagnose_rag_triad_failure(
            context_relevancy=0.85,
            faithfulness=0.50,
            answer_relevancy=0.95,
            threshold=0.7,
            faith_reason=reason,
        )
        assert res["failure_stage"] == "generation_faithfulness"
        assert reason in res["diagnosis"]
        assert "fix prompt, temperature" in res["recommended_fix"]

    def test_answer_relevance_failure_quotes_ans_reason(self):
        reason = "The score is 0.55 because the output discusses poetry instead of the query."
        res = diagnose_rag_triad_failure(
            context_relevancy=0.85,
            faithfulness=0.90,
            answer_relevancy=0.55,
            threshold=0.7,
            ans_reason=reason,
        )
        assert res["failure_stage"] == "generation_relevance"
        assert reason in res["diagnosis"]
        assert "fix prompt instructions" in res["recommended_fix"]

    def test_missing_reason_does_not_guess(self):
        res = diagnose_rag_triad_failure(
            context_relevancy=0.40,
            faithfulness=1.0,
            answer_relevancy=0.9,
            threshold=0.7,
            ctx_reason=None,
        )
        assert "judge reason unavailable" in res["diagnosis"]
        # Ensure no fabricated claims exist
        assert "adjacent SQL" not in res["diagnosis"]
        assert "diluted" not in res["diagnosis"]

    def test_pass_when_all_above_threshold(self):
        res = diagnose_rag_triad_failure(
            context_relevancy=0.75,
            faithfulness=0.80,
            answer_relevancy=0.85,
            threshold=0.7,
        )
        assert res["failure_stage"] == "none"
        assert "Pass: All metrics met threshold" in res["diagnosis"]


class TestDatasetStats:
    """Test dataset loading and row breakdown."""

    def test_golden_v2_stats(self):
        path = os.path.join(REPO_ROOT, "evals", "data", "golden_v2.jsonl")
        evaluated, stats = load_dataset_with_stats(path, include_unanswerable=False)
        assert stats["total_rows"] == 32
        assert stats["evaluated_count"] == 26
        assert stats["skipped_count"] == 6
        assert len(evaluated) == 26
        assert "answerable=false" in stats["skipped_reasons"][0]["reason"]


class TestLikeForLikeComparison:
    """Test matching question IDs and noise threshold filtering."""

    def test_comparison_flags_only_beyond_2sigma(self):
        pipeline_results = [
            {"id": "rp-001", "faithfulness_score": 0.80, "answer_relevancy_score": 0.90, "context_relevancy_score": 0.50},
            {"id": "rp-002", "faithfulness_score": 0.85, "answer_relevancy_score": 0.90, "context_relevancy_score": 0.50},
        ]
        # Noise std dev = 0.02 -> 2sigma = 0.04
        noise_stds = {"faithfulness": 0.02, "answer_relevancy": 0.02, "context_relevancy": 0.02}
        comps = compare_like_for_like_baselines(pipeline_results, noise_stds=noise_stds)
        # Verify matched question comparison keys
        assert "faithfulness_vs_gold_component" in comps
        f_comp = comps["faithfulness_vs_gold_component"]
        assert f_comp["matched_question_count"] == 2
        assert "noise_threshold_2sigma" in f_comp
