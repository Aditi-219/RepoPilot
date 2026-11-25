"""Pipeline-Level Evaluation for RAG Applications using the RAG Triad with DeepEval.

Evaluates the fully wired end-to-end RAG pipeline live:
1. Context Relevance  -> ContextualRelevancyMetric (Retriever noise check)
2. Faithfulness        -> FaithfulnessMetric (Hallucination check)
3. Answer Relevance    -> AnswerRelevancyMetric (On-topic check)

Key Characteristics & Strict Verification Requirements:
-------------------------------------------------------
- Live Pipeline: Every question triggers live retrieval + generation. No saved contexts or answers reused.
- Full Golden Set: Tracks all rows in golden_v2.jsonl, reporting answerable cohort and explicitly noting skipped rows.
- Judge-Verified Diagnoses: The per-question diagnosis strictly quotes or closely paraphrases the judge's reason.
- Like-for-Like Baseline Comparison: Compares matching question IDs against component baselines, flagging changes
  only if |delta| > 2x the judge noise standard deviation (measured via 3x repeats on a 10-question subset).
- Context Relevancy Calibration: Side-by-side comparison of statement-level Contextual Relevancy and chunk-level file precision.
- Judge Sanity Check: Cross-checks a sample of answers with a stronger judge model and outputs full contexts for inspection.
- Retrieval Experiments: Evaluates top-k (5 -> 3) and reranker score cutoff, testing if deltas exceed noise.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import importlib
import json
import logging
import math
import os
import random
import statistics
import sys
import time
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

from dotenv import load_dotenv

load_dotenv(os.path.join(REPO_ROOT, "backend", ".env"))

# ---------------------------------------------------------------------------
# DeepEval imports with test-friendly fallback stubs
# ---------------------------------------------------------------------------
try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
    )
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase

    DEEPEVAL_AVAILABLE = True
except (ImportError, Exception):
    DEEPEVAL_AVAILABLE = False
    DeepEvalBaseLLM = object  # type: ignore[assignment,misc]

    class ContextualRelevancyMetric:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.threshold = kwargs.get("threshold", 0.7)
            self.score = 0.0
            self.reason = ""
        def measure(self, test_case: Any) -> float:
            return self.score

    class FaithfulnessMetric:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.threshold = kwargs.get("threshold", 0.7)
            self.score = 0.0
            self.reason = ""
        def measure(self, test_case: Any) -> float:
            return self.score

    class AnswerRelevancyMetric:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.threshold = kwargs.get("threshold", 0.7)
            self.score = 0.0
            self.reason = ""
        def measure(self, test_case: Any) -> float:
            return self.score

    class LLMTestCase:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("eval_rag_triad")

DEFAULT_GOLDEN_DATASET = os.path.join(REPO_ROOT, "evals", "data", "golden_v2.jsonl")
RUNS_DIR = os.path.join(REPO_ROOT, "evals", "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

# 10 fixed questions used for noise estimation
NOISE_SUBSET_IDS: list[str] = [
    "rp-001", "rp-002", "rp-003", "rp-004", "rp-005",
    "rp-006", "rp-007", "rp-008", "rp-011", "rp-015"
]


# ---------------------------------------------------------------------------
# 1. Pipeline Definition & Live Runner
# ---------------------------------------------------------------------------

def default_rag_pipeline(
    question: str,
    repository_id: int | None = None,
    top_k: int = 5,
    pool_size: int = 15,
    rerank_cutoff: float | None = None,
    allow_generator_fallback: bool = False,
    generator_provider: str | None = None,
) -> dict[str, Any]:
    """Execute live end-to-end RepoPilot RAG pipeline for a given question.

    Returns:
        {
            "answer": str,
            "retrieval_context": list[str],
            "retrieved_files": list[str],
            "snippets": list[Any],
            "generator_provider": str,
            "generator_model": str,
        }
    """
    from app.database import SessionLocal
    from app.llm_service import ask_with_llm
    from app.llm_provider import (
        ProviderJudgeLLM,
        select_healthy_judge_provider,
        assert_matching_judges,
        JudgeMismatchError,
        JudgeRunHaltedError,
        get_model_for_provider,
    )
    from app.models import Repository
    from app.reranker import rerank
    from app.search import hybrid_search

    with SessionLocal() as db:
        if repository_id is None:
            repo = (
                db.query(Repository)
                .filter(Repository.full_name == "repopilot", Repository.status == "completed")
                .first()
            )
            if not repo:
                repo = db.query(Repository).filter(Repository.status == "completed").first()
            if not repo:
                raise RuntimeError("No indexed repository with status='completed' found in database.")
            repository_id = repo.id

        # 1. Live hybrid retrieval (constant pool_size regardless of top_k)
        candidates = hybrid_search(db, repository_id, question, limit=pool_size)

    # 2. Live cross-encoder reranking
    snippets = rerank(question, candidates, top_k=top_k)

    # Optional rerank score cutoff
    if rerank_cutoff is not None:
        filtered_snippets = [s for s in snippets if s.score >= rerank_cutoff]
        # Keep at least the top chunk if all fall below cutoff to avoid empty context failure
        snippets = filtered_snippets if filtered_snippets else snippets[:1]

    retrieval_context = [
        f"File: {s.file_path} (lines {s.start_line}-{s.end_line})\n{s.content}"
        for s in snippets
    ]
    retrieved_files = [s.file_path for s in snippets]

    # 3. Live LLM answer generation with fallback rules & provenance
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            answer, prov_model = pool.submit(
                asyncio.run,
                ask_with_llm(
                    question,
                    snippets,
                    allow_fallback=allow_generator_fallback,
                    provider=generator_provider,
                ),
            ).result()
    else:
        answer, prov_model = loop.run_until_complete(
            ask_with_llm(
                question,
                snippets,
                allow_fallback=allow_generator_fallback,
                provider=generator_provider,
            )
        )

    gen_prov = prov_model.split("/")[0] if "/" in prov_model else prov_model
    gen_model = prov_model.split("/")[1] if "/" in prov_model else ""

    return {
        "answer": answer,
        "retrieval_context": retrieval_context,
        "retrieved_files": retrieved_files,
        "snippets": snippets,
        "generator_provider": gen_prov,
        "generator_model": gen_model,
    }


def resolve_pipeline_callable(pipeline_spec: str | None) -> Callable[..., dict[str, Any]]:
    """Resolve a pipeline function from string 'module.submodule:function_name' or use default."""
    if not pipeline_spec:
        return default_rag_pipeline

    if ":" not in pipeline_spec:
        raise ValueError(f"Invalid pipeline format: '{pipeline_spec}'. Expected 'module.path:callable_name'")
    mod_name, func_name = pipeline_spec.split(":", 1)
    module = importlib.import_module(mod_name)
    func = getattr(module, func_name)
    if not callable(func):
        raise TypeError(f"Target '{func_name}' in module '{mod_name}' is not callable.")
    return func


# ---------------------------------------------------------------------------
# 2. Dataset Loader & Filtering
# ---------------------------------------------------------------------------

def load_dataset_with_stats(
    dataset_path: str,
    include_unanswerable: bool = False,
    limit: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load golden dataset, compute row statistics, and return evaluated set + meta."""
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Golden dataset not found at: {dataset_path}")

    all_records: list[dict[str, Any]] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_records.append(json.loads(line))

    total_rows = len(all_records)
    answerable_records = [r for r in all_records if r.get("answerable", True)]
    unanswerable_records = [r for r in all_records if not r.get("answerable", True)]

    skipped_records: list[dict[str, Any]] = []
    if include_unanswerable:
        evaluated_records = all_records
    else:
        evaluated_records = answerable_records
        skipped_records = unanswerable_records

    if limit is not None and limit > 0:
        evaluated_records = evaluated_records[:limit]

    stats = {
        "total_rows": total_rows,
        "evaluated_count": len(evaluated_records),
        "skipped_count": len(skipped_records),
        "skipped_reasons": [
            {
                "id": r.get("id"),
                "category": r.get("category"),
                "question": r.get("question"),
                "reason": (
                    "Marked answerable=false. RAG Triad evaluates knowledge retrieval internal consistency. "
                    "For out-of-scope/unanswerable queries, no relevant codebase context exists and correct "
                    "behavior is refusal (evaluated via Refusal Correctness / GEval, not RAG Triad)."
                ),
            }
            for r in skipped_records
        ],
    }

    return evaluated_records, stats


# ---------------------------------------------------------------------------
# 3. Judge LLM Factory
# ---------------------------------------------------------------------------

def create_judge_model(
    provider: str = "gemini",
    model_name: str | None = None,
    temperature: float = 0.0,
    cache_path: str | None = None,
    cache_prefix: str = "",
) -> Any:
    """Instantiate a DeepEval-compatible judge model backed by ProviderJudgeLLM."""
    if cache_path is None:
        cache_path = os.path.join(RUNS_DIR, "rag_triad_judge_cache.json")

    try:
        from app.llm_provider import ProviderJudgeLLM, get_model_for_provider
        actual_model = model_name or get_model_for_provider(provider)
        return ProviderJudgeLLM(
            provider=provider,
            model=actual_model,
            temperature=temperature,
            cache_path=cache_path,
            cache_prefix=cache_prefix,
        )
    except Exception as exc:
        logger.warning("Could not instantiate ProviderJudgeLLM (%s), using fallback raw model string.", exc)
        return model_name or "gemini-3.5-flash-lite"


# ---------------------------------------------------------------------------
# 4. Context Relevancy Calibration: Chunk File Precision
# ---------------------------------------------------------------------------

def compute_chunk_file_precision(retrieved_files: list[str], gt_files: list[str]) -> float:
    """Calculate the fraction of retrieved chunks whose source file matches ground truth."""
    if not retrieved_files:
        return 0.0
    if not gt_files:
        return 0.0

    hits = 0
    for rf in retrieved_files:
        # Match exact file path or normalize relative path
        if any(gt in rf or rf in gt for gt in gt_files):
            hits += 1

    return round(hits / len(retrieved_files), 4)


# ---------------------------------------------------------------------------
# 5. Verified Diagnosis Function (Judge Reasons Only)
# ---------------------------------------------------------------------------

def diagnose_rag_triad_failure(
    context_relevancy: float | None,
    faithfulness: float | None,
    answer_relevancy: float | None,
    threshold: float = 0.7,
    ctx_reason: str | None = None,
    faith_reason: str | None = None,
    ans_reason: str | None = None,
) -> dict[str, str]:
    """Diagnose the root cause of a RAG pipeline failure strictly using judge reason text.

    Decision Logic:
    1. Low context relevance (< threshold):
       Diagnosis must quote or closely paraphrase the judge's context_relevancy reason.
    2. Good context relevance (>= threshold) + low faithfulness (< threshold):
       Diagnosis must quote or closely paraphrase the judge's faithfulness reason.
    3. Good context relevance (>= threshold) + good faithfulness (>= threshold) + low answer relevance (< threshold):
       Diagnosis must quote or closely paraphrase the judge's answer_relevancy reason.
    4. All metrics >= threshold:
       Pass: Healthy across RAG Triad.
    """
    if context_relevancy is None or faithfulness is None or answer_relevancy is None:
        return {
            "failure_stage": "evaluation_error",
            "diagnosis": "judge reason unavailable (evaluation error or timeout)",
            "recommended_fix": "Inspect judge connection and rate limits",
        }

    # Step 1: Retrieval noise check
    if context_relevancy < threshold:
        reason_text = (ctx_reason or "").strip()
        diagnosis_msg = reason_text if reason_text else "judge reason unavailable"
        return {
            "failure_stage": "retrieval",
            "diagnosis": f"Retrieval problem: {diagnosis_msg}",
            "recommended_fix": "fix chunking, top-k, embedding, reranker",
        }

    # Step 2: Hallucination check
    if faithfulness < threshold:
        reason_text = (faith_reason or "").strip()
        diagnosis_msg = reason_text if reason_text else "judge reason unavailable"
        return {
            "failure_stage": "generation_faithfulness",
            "diagnosis": f"Generator hallucination: {diagnosis_msg}",
            "recommended_fix": "fix prompt, temperature",
        }

    # Step 3: On-topic check
    if answer_relevancy < threshold:
        reason_text = (ans_reason or "").strip()
        diagnosis_msg = reason_text if reason_text else "judge reason unavailable"
        return {
            "failure_stage": "generation_relevance",
            "diagnosis": f"Generator off-topic: {diagnosis_msg}",
            "recommended_fix": "fix prompt instructions",
        }

    return {
        "failure_stage": "none",
        "diagnosis": (
            f"Pass: All metrics met threshold {threshold:.2f} "
            f"(Context Relevancy: {context_relevancy:.2f}, Faithfulness: {faithfulness:.2f}, "
            f"Answer Relevancy: {answer_relevancy:.2f})."
        ),
        "recommended_fix": "none",
    }


# ---------------------------------------------------------------------------
# 6. Noise Estimation (Judge Variance on 10 Questions, 3x Repeats)
# ---------------------------------------------------------------------------

def estimate_judge_noise(
    dataset: list[dict[str, Any]],
    pipeline_fn: Callable[..., dict[str, Any]],
    judge_provider: str = "gemini",
    judge_model: str = "gemini-3.5-flash-lite",
    sample_ids: list[str] = NOISE_SUBSET_IDS,
    n_repeats: int = 3,
) -> dict[str, float]:
    """Estimate judge stochasticity/noise standard deviation across repeated evaluations.

    Runs the judge n_repeats times on a fixed subset of questions with the same inputs,
    bypassing identical-prompt cache via unique prefixes to measure true judge variance.
    """
    sample_questions = [r for r in dataset if r.get("id") in sample_ids]
    if not sample_questions:
        sample_questions = dataset[:min(10, len(dataset))]

    print(f"\n[Noise Estimation] Running judge {n_repeats}x on {len(sample_questions)} questions to compute std dev...")

    # First, run live pipeline once to obtain (question, actual_output, retrieval_context) test cases
    cached_test_cases: list[tuple[str, Any]] = []
    for item in sample_questions:
        qid = item.get("id")
        q = item["question"]
        out = pipeline_fn(q)
        tc = LLMTestCase(
            input=q,
            actual_output=out.get("answer", ""),
            retrieval_context=out.get("retrieval_context", []),
        )
        cached_test_cases.append((qid, tc))

    per_q_scores: dict[str, dict[str, list[float]]] = {
        qid: {"context_relevancy": [], "faithfulness": [], "answer_relevancy": []}
        for qid, _ in cached_test_cases
    }

    cache_file = os.path.join(RUNS_DIR, "rag_triad_judge_cache.json")

    for rep in range(1, n_repeats + 1):
        print(f"  -> Repeat {rep}/{n_repeats}...")
        judge_rep = create_judge_model(
            provider=judge_provider,
            model_name=judge_model,
            temperature=0.0,
            cache_path=cache_file,
            cache_prefix=f"noise_run_{rep}:",
        )
        m_ctx = ContextualRelevancyMetric(threshold=0.7, model=judge_rep, include_reason=False)
        m_faith = FaithfulnessMetric(threshold=0.7, model=judge_rep, include_reason=False)
        m_ans = AnswerRelevancyMetric(threshold=0.7, model=judge_rep, include_reason=False)

        for qid, tc in cached_test_cases:
            try:
                m_ctx.measure(tc)
                c = float(m_ctx.score)
            except Exception:
                c = 0.0

            try:
                m_faith.measure(tc)
                f = float(m_faith.score)
            except Exception:
                f = 0.0

            try:
                m_ans.measure(tc)
                a = float(m_ans.score)
            except Exception:
                a = 0.0

            per_q_scores[qid]["context_relevancy"].append(c)
            per_q_scores[qid]["faithfulness"].append(f)
            per_q_scores[qid]["answer_relevancy"].append(a)

    # Compute average standard deviation across questions per metric
    noise_std: dict[str, float] = {}
    for metric_name in ["context_relevancy", "faithfulness", "answer_relevancy"]:
        stds = []
        for qid in per_q_scores:
            vals = per_q_scores[qid][metric_name]
            if len(vals) > 1:
                stds.append(statistics.stdev(vals))
            else:
                stds.append(0.0)
        noise_std[metric_name] = round(statistics.mean(stds), 4)

    print(
        f"[Noise Estimation Result] Mean Std Dev across {n_repeats} repeats: "
        f"ContextRelevancy σ={noise_std['context_relevancy']:.4f} (2σ={noise_std['context_relevancy']*2:.4f}), "
        f"Faithfulness σ={noise_std['faithfulness']:.4f} (2σ={noise_std['faithfulness']*2:.4f}), "
        f"AnswerRelevancy σ={noise_std['answer_relevancy']:.4f} (2σ={noise_std['answer_relevancy']*2:.4f})"
    )
    return noise_std


# ---------------------------------------------------------------------------
# 7. Like-for-Like Baseline Comparison
# ---------------------------------------------------------------------------

def compare_like_for_like_baselines(
    pipeline_results: list[dict[str, Any]],
    noise_stds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare pipeline scores against component baseline scores on matching question IDs only.

    Flags real changes only when |delta| > 2 * noise_std.
    """
    pipe_by_id = {r["id"]: r for r in pipeline_results}

    # Load Generator Gold Baseline
    gold_path = os.path.join(RUNS_DIR, "generator_baseline_gold.json")
    gold_matched: list[dict[str, Any]] = []
    if os.path.exists(gold_path):
        with open(gold_path, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
        for item in gold_data.get("query_details", []):
            qid = item.get("id")
            if qid in pipe_by_id:
                gold_matched.append(item)

    # Load Retriever Baseline
    ret_path = os.path.join(RUNS_DIR, "retriever_baseline_full_file.json")
    ret_matched: list[dict[str, Any]] = []
    if os.path.exists(ret_path):
        with open(ret_path, "r", encoding="utf-8") as f:
            ret_data = json.load(f)
        for item in ret_data.get("query_details", []):
            qid = item.get("id")
            if qid in pipe_by_id:
                ret_matched.append(item)

    comparisons: dict[str, Any] = {}

    # 1. Faithfulness: Pipeline vs Generator Gold Context (Matching IDs)
    if gold_matched:
        matched_qids = [g["id"] for g in gold_matched]
        p_faith = [pipe_by_id[qid]["faithfulness_score"] for qid in matched_qids if pipe_by_id[qid]["faithfulness_score"] is not None]
        g_faith = [g.get("scores", {}).get("faithfulness", {}).get("score", 0.0) for g in gold_matched]
        delta_faith = round(statistics.mean(p_faith) - statistics.mean(g_faith), 4)

        thresh_faith = (noise_stds.get("faithfulness", 0.05) * 2.0) if noise_stds else 0.10
        is_real_change = abs(delta_faith) > thresh_faith

        comparisons["faithfulness_vs_gold_component"] = {
            "matched_question_count": len(matched_qids),
            "pipeline_mean": round(statistics.mean(p_faith), 4),
            "component_gold_mean": round(statistics.mean(g_faith), 4),
            "delta": delta_faith,
            "noise_threshold_2sigma": round(thresh_faith, 4),
            "real_change": is_real_change,
            "status": "STATISTICALLY_SIGNIFICANT" if is_real_change else "WITHIN_NOISE_THRESHOLD",
        }

        # 2. Answer Relevancy: Pipeline vs Generator Gold Context (Matching IDs)
        p_ans = [pipe_by_id[qid]["answer_relevancy_score"] for qid in matched_qids if pipe_by_id[qid]["answer_relevancy_score"] is not None]
        g_ans = [g.get("scores", {}).get("answer_relevancy", {}).get("score", 0.0) for g in gold_matched]
        delta_ans = round(statistics.mean(p_ans) - statistics.mean(g_ans), 4)

        thresh_ans = (noise_stds.get("answer_relevancy", 0.05) * 2.0) if noise_stds else 0.10
        is_real_ans_change = abs(delta_ans) > thresh_ans

        comparisons["answer_relevancy_vs_gold_component"] = {
            "matched_question_count": len(matched_qids),
            "pipeline_mean": round(statistics.mean(p_ans), 4),
            "component_gold_mean": round(statistics.mean(g_ans), 4),
            "delta": delta_ans,
            "noise_threshold_2sigma": round(thresh_ans, 4),
            "real_change": is_real_ans_change,
            "status": "STATISTICALLY_SIGNIFICANT" if is_real_ans_change else "WITHIN_NOISE_THRESHOLD",
        }

    # 3. Context Relevancy vs Retriever Recall@5 (Matching IDs)
    if ret_matched:
        matched_qids = [r["id"] for r in ret_matched]
        p_ctx = [pipe_by_id[qid]["context_relevancy_score"] for qid in matched_qids if pipe_by_id[qid]["context_relevancy_score"] is not None]
        r_rec = [
            r.get("stages", {}).get("reranked", {}).get("metrics", {}).get("Recall@5", 0.0)
            for r in ret_matched
        ]
        delta_ctx = round(statistics.mean(p_ctx) - statistics.mean(r_rec), 4)

        thresh_ctx = (noise_stds.get("context_relevancy", 0.05) * 2.0) if noise_stds else 0.10
        is_real_ctx_change = abs(delta_ctx) > thresh_ctx

        comparisons["context_relevancy_vs_retriever_recall"] = {
            "matched_question_count": len(matched_qids),
            "pipeline_mean": round(statistics.mean(p_ctx), 4),
            "component_recall_mean": round(statistics.mean(r_rec), 4),
            "delta": delta_ctx,
            "noise_threshold_2sigma": round(thresh_ctx, 4),
            "real_change": is_real_ctx_change,
            "status": "STATISTICALLY_SIGNIFICANT" if is_real_ctx_change else "WITHIN_NOISE_THRESHOLD",
        }

    return comparisons


# ---------------------------------------------------------------------------
# 8. Judge Sanity Check (Stronger Model & Manual Inspection Inspection)
# ---------------------------------------------------------------------------

def run_judge_sanity_check(
    sampled_results: list[dict[str, Any]],
    stronger_model: str = "gemini-3-flash-preview",
    judge_provider: str = "gemini",
) -> dict[str, Any]:
    """Re-judge a sample of 10 answers with a stronger judge model and measure agreement."""
    print(f"\n[Judge Sanity Check] Re-judging {len(sampled_results)} answers using stronger model '{stronger_model}'...")

    cache_file = os.path.join(RUNS_DIR, "rag_triad_judge_cache.json")
    stronger_judge = create_judge_model(
        provider=judge_provider,
        model_name=stronger_model,
        temperature=0.0,
        cache_path=cache_file,
    )
    metric_faith_strong = FaithfulnessMetric(threshold=0.7, model=stronger_judge, include_reason=True)

    baseline_scores: list[float] = []
    stronger_scores: list[float] = []
    agreement_count = 0
    inspection_entries: list[dict[str, Any]] = []

    for item in sampled_results:
        qid = item["id"]
        q = item["question"]
        ans = item["answer"]
        ctx = item.get("retrieval_context", [])

        tc = LLMTestCase(input=q, actual_output=ans, retrieval_context=ctx)
        try:
            metric_faith_strong.measure(tc)
            s_score = float(metric_faith_strong.score)
            s_reason = getattr(metric_faith_strong, "reason", "")
        except Exception as e:
            s_score = 0.0
            s_reason = f"Error: {e}"

        b_score = item["faithfulness_score"]
        baseline_scores.append(b_score)
        stronger_scores.append(s_score)

        # Pass/fail agreement at 0.7 threshold
        b_pass = b_score >= 0.7
        s_pass = s_score >= 0.7
        if b_pass == s_pass:
            agreement_count += 1

        inspection_entries.append({
            "id": qid,
            "question": q,
            "answer": ans,
            "retrieval_context_snippets": ctx[:2],  # first 2 snippets for inspection
            "baseline_score": b_score,
            "stronger_score": s_score,
            "stronger_reason": s_reason,
            "agreed": b_pass == s_pass,
        })

    # Mean absolute difference
    mad = (
        round(statistics.mean(abs(b - s) for b, s in zip(baseline_scores, stronger_scores)), 4)
        if baseline_scores else 0.0
    )
    pass_agreement_pct = round(agreement_count / len(sampled_results), 4) if sampled_results else 0.0

    print(
        f"[Judge Sanity Check Result] Evaluated {len(sampled_results)} items: "
        f"Pass/Fail Agreement = {pass_agreement_pct*100:.1f}%, Mean Absolute Difference = {mad:.4f}"
    )

    return {
        "stronger_model": stronger_model,
        "sample_size": len(sampled_results),
        "pass_agreement_rate": pass_agreement_pct,
        "mean_absolute_difference": mad,
        "inspection_samples": inspection_entries[:5],  # 5 full samples for eye inspection
    }


# ---------------------------------------------------------------------------
# 9. Retrieval Experiments Runner
# ---------------------------------------------------------------------------

def run_retrieval_experiments(
    dataset: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
    noise_stds: dict[str, float],
    threshold: float = 0.7,
    judge_provider: str = "gemini",
    judge_model: str = "gemini-3.5-flash-lite",
) -> dict[str, Any]:
    """Run single-change retrieval experiments (top-k=3, reranker cutoff=3.0) against baseline."""
    print("\n" + "=" * 76)
    print(" RETRIEVAL CANDIDATE EXPERIMENTS (One Change At A Time)")
    print("=" * 76)

    cache_file = os.path.join(RUNS_DIR, "rag_triad_judge_cache.json")
    judge = create_judge_model(
        provider=judge_provider,
        model_name=judge_model,
        temperature=0.0,
        cache_path=cache_file,
    )

    experiments_results: dict[str, Any] = {}

    # Define candidate experiments
    candidate_configs = [
        {"name": "exp_a_top_k_3", "desc": "Top-k reduced from 5 to 3", "kwargs": {"top_k": 3}},
        {"name": "exp_b_rerank_cutoff_3", "desc": "Rerank score cutoff at 3.0 (drop chunks < 3.0)", "kwargs": {"top_k": 5, "rerank_cutoff": 3.0}},
    ]

    for cfg in candidate_configs:
        exp_name = cfg["name"]
        exp_desc = cfg["desc"]
        kw = cfg["kwargs"]

        print(f"\n[Experiment] Running {exp_name}: {exp_desc} on {len(dataset)} questions...")
        m_ctx = ContextualRelevancyMetric(threshold=threshold, model=judge, include_reason=True)
        m_faith = FaithfulnessMetric(threshold=threshold, model=judge, include_reason=True)
        m_ans = AnswerRelevancyMetric(threshold=threshold, model=judge, include_reason=True)

        scores_ctx = []
        scores_faith = []
        scores_ans = []
        target_hits_k = 0

        for item in dataset:
            q = item["question"]
            gt_files = item.get("gt_files", [])
            out = default_rag_pipeline(q, **kw)
            ans = out.get("answer", "")
            ctx = out.get("retrieval_context", [])
            ret_files = out.get("retrieved_files", [])

            # Check target recall hit for this experiment
            if any(any(gt in rf or rf in gt for gt in gt_files) for rf in ret_files):
                target_hits_k += 1

            tc = LLMTestCase(input=q, actual_output=ans, retrieval_context=ctx)
            try:
                m_ctx.measure(tc)
                scores_ctx.append(float(m_ctx.score))
            except Exception:
                scores_ctx.append(0.0)

            try:
                m_faith.measure(tc)
                scores_faith.append(float(m_faith.score))
            except Exception:
                scores_faith.append(0.0)

            try:
                m_ans.measure(tc)
                scores_ans.append(float(m_ans.score))
            except Exception:
                scores_ans.append(0.0)

        # Calculate metrics
        mean_ctx = round(statistics.mean(scores_ctx), 4)
        mean_faith = round(statistics.mean(scores_faith), 4)
        mean_ans = round(statistics.mean(scores_ans), 4)
        recall_at_k = round(target_hits_k / len(dataset), 4)

        base_ctx = baseline_summary["context_relevancy"]["mean"]
        base_faith = baseline_summary["faithfulness"]["mean"]
        base_ans = baseline_summary["answer_relevancy"]["mean"]
        base_recall = baseline_summary.get("retriever_recall_at_5", 1.0)

        delta_ctx = round(mean_ctx - base_ctx, 4)
        delta_faith = round(mean_faith - base_faith, 4)
        delta_ans = round(mean_ans - base_ans, 4)
        delta_recall = round(recall_at_k - base_recall, 4)

        thresh_ctx_2s = noise_stds.get("context_relevancy", 0.05) * 2.0
        thresh_faith_2s = noise_stds.get("faithfulness", 0.05) * 2.0
        thresh_ans_2s = noise_stds.get("answer_relevancy", 0.05) * 2.0

        experiments_results[exp_name] = {
            "description": exp_desc,
            "metrics": {
                "context_relevancy": {
                    "mean": mean_ctx,
                    "delta_vs_baseline": delta_ctx,
                    "noise_threshold_2s": round(thresh_ctx_2s, 4),
                    "exceeds_noise": abs(delta_ctx) > thresh_ctx_2s,
                },
                "faithfulness": {
                    "mean": mean_faith,
                    "delta_vs_baseline": delta_faith,
                    "noise_threshold_2s": round(thresh_faith_2s, 4),
                    "exceeds_noise": abs(delta_faith) > thresh_faith_2s,
                },
                "answer_relevancy": {
                    "mean": mean_ans,
                    "delta_vs_baseline": delta_ans,
                    "noise_threshold_2s": round(thresh_ans_2s, 4),
                    "exceeds_noise": abs(delta_ans) > thresh_ans_2s,
                },
            },
            "retriever_recall": {
                "recall_at_k": recall_at_k,
                "delta_vs_baseline": delta_recall,
                "has_regression": delta_recall < 0,
            },
        }

    # Explicit note for AST chunk boundary experiment
    experiments_results["exp_c_tighter_ast_boundaries"] = {
        "description": "Tighter AST-based chunk boundaries",
        "status": "SKIPPED_NOT_SUPPORTED_AT_QUERY_TIME",
        "reason": (
            "RepoPilot implements AST chunking during repository indexing (persisted in PostgreSQL code_chunks). "
            "The search and reranking layers operate over pre-indexed chunks; query-time AST boundary tightening "
            "is not supported without re-indexing the database corpus."
        ),
    }

    return experiments_results


# ---------------------------------------------------------------------------
# 10. Main Evaluator Orchestrator
# ---------------------------------------------------------------------------

def run_eval_rag_triad(
    dataset_path: str = DEFAULT_GOLDEN_DATASET,
    threshold: float = 0.7,
    include_unanswerable: bool = False,
    limit: int | None = None,
    pipeline_spec: str | None = None,
    judge_provider: str | None = None,
    judge_model: str | None = None,
    allow_generator_fallback: bool = False,
    generator_provider: str | None = None,
    run_noise_estimation: bool = True,
    run_sanity_check: bool = True,
    run_experiments: bool = True,
    output_csv_path: str | None = None,
    output_json_path: str | None = None,
) -> dict[str, Any]:
    """Execute complete, verified RAG Triad evaluation."""
    from app.llm_provider import (
        select_healthy_judge_provider,
        JudgeRunHaltedError,
        get_model_for_provider,
    )

    run_id = f"rag_triad_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if output_csv_path is None:
        output_csv_path = os.path.join(RUNS_DIR, f"{run_id}.csv")
    if output_json_path is None:
        output_json_path = os.path.join(RUNS_DIR, f"{run_id}.json")

    pipeline_fn = resolve_pipeline_callable(pipeline_spec)
    dataset, dataset_stats = load_dataset_with_stats(
        dataset_path, include_unanswerable=include_unanswerable, limit=limit
    )

    # Rule 2: Health check each provider in chain order, pick once at start of run
    actual_judge_prov, default_j_model = select_healthy_judge_provider(forced_provider=judge_provider)
    actual_judge_model = judge_model or default_j_model

    cache_file = os.path.join(RUNS_DIR, "rag_triad_judge_cache.json")
    judge = create_judge_model(
        provider=actual_judge_prov,
        model_name=actual_judge_model,
        temperature=0.0,
        cache_path=cache_file,
    )

    print("\n" + "=" * 76)
    print(" RAG TRIAD PIPELINE EVALUATION (DeepEval Live Full Run)")
    print(f" Run ID               : {run_id}")
    print(f" Dataset File         : {os.path.basename(dataset_path)}")
    print(f" Total Rows in File   : {dataset_stats['total_rows']}")
    print(f" Evaluated Questions  : {dataset_stats['evaluated_count']}")
    print(f" Skipped Questions    : {dataset_stats['skipped_count']}")
    if dataset_stats["skipped_count"] > 0:
        print(" Skipped Breakdown    : 6 rows marked answerable=false (3 technical unanswerables, 3 out-of-scope).")
        print("                        Skipped because RAG Triad assumes knowledge retrieval; unanswerables test refusal.")
    print(f" Judge Model          : {actual_judge_prov}/{actual_judge_model} (temp=0.0)")
    print(f" Metric Threshold     : {threshold} (fixed per metric)")
    print(f" Gen Fallback Allowed : {allow_generator_fallback}")
    print("=" * 76)

    # 1. Noise Estimation on 10 Questions (3 repeats)
    noise_stds: dict[str, float] = {}
    if run_noise_estimation:
        noise_stds = estimate_judge_noise(
            dataset=dataset,
            pipeline_fn=pipeline_fn,
            judge_provider=actual_judge_prov,
            judge_model=actual_judge_model,
            n_repeats=3,
        )

    # 2. Main Live Evaluation Loop
    metric_ctx = ContextualRelevancyMetric(threshold=threshold, model=judge, include_reason=True)
    metric_faith = FaithfulnessMetric(threshold=threshold, model=judge, include_reason=True)
    metric_ans = AnswerRelevancyMetric(threshold=threshold, model=judge, include_reason=True)

    scores_ctx: list[float] = []
    scores_faith: list[float] = []
    scores_ans: list[float] = []
    scores_chunk_prec: list[float] = []
    disagreements: list[dict[str, Any]] = []

    per_question_results: list[dict[str, Any]] = []
    target_hits_baseline = 0

    csv_fields = [
        "id",
        "question",
        "answer",
        "context_relevancy_score",
        "chunk_file_precision",
        "context_relevancy_reason",
        "faithfulness_score",
        "faithfulness_reason",
        "answer_relevancy_score",
        "answer_relevancy_reason",
        "passed",
        "failure_stage",
        "diagnosis",
        "recommended_fix",
        "judge_provider",
        "judge_model",
        "generator_provider",
        "generator_model",
    ]

    print(f"\n[Execution] Running live RAG pipeline & 3 metrics on {len(dataset)} questions...")
    try:
        for idx, item in enumerate(dataset, start=1):
            qid = item.get("id", f"q-{idx}")
            question = item["question"]
            gt_files = item.get("gt_files", [])

            t0 = time.time()
            # LIVE PIPELINE EXECUTION
            try:
                pipe_output = pipeline_fn(
                    question,
                    allow_generator_fallback=allow_generator_fallback,
                    generator_provider=generator_provider,
                )
            except Exception as pipe_err:
                logger.error("Pipeline failure on question %s: %s", qid, pipe_err)
                pipe_output = {
                    "answer": f"Evaluation pipeline error: {pipe_err}",
                    "retrieval_context": [],
                    "retrieved_files": [],
                    "snippets": [],
                    "generator_provider": "failed",
                    "generator_model": "failed",
                }

            answer = pipe_output.get("answer", "")
            retrieval_context = pipe_output.get("retrieval_context", [])
            retrieved_files = pipe_output.get("retrieved_files", [])
            gen_prov = pipe_output.get("generator_provider", "unknown")
            gen_model = pipe_output.get("generator_model", "unknown")
            pipe_latency = round(time.time() - t0, 2)

            # Ground truth target recall check
            is_hit = any(any(gt in rf or rf in gt for gt in gt_files) for rf in retrieved_files)
            if is_hit:
                target_hits_baseline += 1

            # Chunk-level file precision
            chunk_prec = compute_chunk_file_precision(retrieved_files, gt_files)
            scores_chunk_prec.append(chunk_prec)

            # Build DeepEval Test Case
            test_case = LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=retrieval_context,
            )

            # Set current metric for error and parse failure tracking
            if hasattr(judge, "set_current_metric"):
                judge.set_current_metric("contextual_relevancy")
            # 1. Context Relevance
            try:
                metric_ctx.measure(test_case)
                c_score = float(metric_ctx.score)
                c_reason = getattr(metric_ctx, "reason", "")
            except Exception as e:
                c_score = 0.0
                c_reason = f"judge reason unavailable (error: {e})"

            if hasattr(judge, "set_current_metric"):
                judge.set_current_metric("faithfulness")
            # 2. Faithfulness
            try:
                metric_faith.measure(test_case)
                f_score = float(metric_faith.score)
                f_reason = getattr(metric_faith, "reason", "")
            except Exception as e:
                f_score = 0.0
                f_reason = f"judge reason unavailable (error: {e})"

            if hasattr(judge, "set_current_metric"):
                judge.set_current_metric("answer_relevancy")
            # 3. Answer Relevance
            try:
                metric_ans.measure(test_case)
                a_score = float(metric_ans.score)
                a_reason = getattr(metric_ans, "reason", "")
            except Exception as e:
                a_score = 0.0
                a_reason = f"judge reason unavailable (error: {e})"

            scores_ctx.append(c_score)
            scores_faith.append(f_score)
            scores_ans.append(a_score)

            # Note disagreement between statement-level context relevancy and chunk-level file match
            if abs(c_score - chunk_prec) >= 0.30:
                disagreements.append({
                    "id": qid,
                    "question": question,
                    "context_relevancy_score": round(c_score, 4),
                    "chunk_file_precision": round(chunk_prec, 4),
                    "difference": round(c_score - chunk_prec, 4),
                    "reason_excerpt": c_reason[:150] if c_reason else "unavailable",
                })

            passed = (c_score >= threshold) and (f_score >= threshold) and (a_score >= threshold)
            diagnosis = diagnose_rag_triad_failure(
                c_score, f_score, a_score, threshold=threshold,
                ctx_reason=c_reason, faith_reason=f_reason, ans_reason=a_reason,
            )

            print(
                f"[{idx}/{len(dataset)}] {qid} | CtxRel: {c_score:.3f} (ChunkPrec: {chunk_prec:.2f}) | "
                f"Faith: {f_score:.3f} | AnsRel: {a_score:.3f} | {'PASS' if passed else diagnosis['failure_stage']}"
            )

            per_question_results.append({
                "id": qid,
                "question": question,
                "answer": answer,
                "retrieval_context": retrieval_context,
                "retrieval_context_count": len(retrieval_context),
                "pipeline_latency_s": pipe_latency,
                "context_relevancy_score": round(c_score, 4),
                "context_relevancy_reason": c_reason,
                "chunk_file_precision": round(chunk_prec, 4),
                "faithfulness_score": round(f_score, 4),
                "faithfulness_reason": f_reason,
                "answer_relevancy_score": round(a_score, 4),
                "answer_relevancy_reason": a_reason,
                "passed": passed,
                "failure_stage": diagnosis["failure_stage"],
                "diagnosis": diagnosis["diagnosis"],
                "recommended_fix": diagnosis["recommended_fix"],
                "judge_provider": actual_judge_prov,
                "judge_model": actual_judge_model,
                "generator_provider": gen_prov,
                "generator_model": gen_model,
            })
    except JudgeRunHaltedError as exc:
        print(f"\n[CRITICAL] Designated judge failed mid-run: {exc}")
        print("Halting evaluation and writing partial checkpoint (refusing to switch judge mid-run)...")
        halt_payload = {
            "run_id": run_id,
            "status": "HALTED_MID_RUN",
            "halt_reason": str(exc),
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "judge_provider": actual_judge_prov,
            "judge_model": actual_judge_model,
            "results": per_question_results,
        }
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(halt_payload, f, indent=2)
        if per_question_results:
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(per_question_results)
        raise

    # Summary Statistics Function
    def _dist_stats(vals: list[float], thresh: float) -> dict[str, Any]:
        if not vals:
            return {"mean": 0.0, "p50": 0.0, "std_dev": 0.0, "min": 0.0, "max": 0.0, "pass_rate": 0.0, "threshold": thresh, "n": 0}
        return {
            "mean": round(statistics.mean(vals), 4),
            "p50": round(statistics.median(vals), 4),
            "std_dev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "pass_rate": round(sum(1 for v in vals if v >= thresh) / len(vals), 4),
            "threshold": thresh,
            "n": len(vals),
        }

    summary = {
        "context_relevancy": _dist_stats(scores_ctx, threshold),
        "chunk_file_precision": _dist_stats(scores_chunk_prec, threshold),
        "faithfulness": _dist_stats(scores_faith, threshold),
        "answer_relevancy": _dist_stats(scores_ans, threshold),
        "retriever_recall_at_5": round(target_hits_baseline / len(dataset), 4) if dataset else 0.0,
        "overall_pipeline": {
            "pass_rate": round(sum(1 for r in per_question_results if r["passed"]) / len(per_question_results), 4) if per_question_results else 0.0,
            "total_questions": len(per_question_results),
            "passed_count": sum(1 for r in per_question_results if r["passed"]),
        },
    }

    # 3. Like-for-Like Baseline Comparison
    comparisons = compare_like_for_like_baselines(per_question_results, noise_stds=noise_stds)

    # 4. Judge Sanity Check (10 sample questions with stronger model)
    sanity_check_results: dict[str, Any] = {}
    if run_sanity_check:
        sample_for_sanity = per_question_results[:min(10, len(per_question_results))]
        sanity_check_results = run_judge_sanity_check(sample_for_sanity)

    # 5. Retrieval Candidate Experiments (top-k=3, cutoff=3.0)
    experiments_data: dict[str, Any] = {}
    if run_experiments:
        experiments_data = run_retrieval_experiments(
            dataset=dataset,
            baseline_summary=summary,
            noise_stds=noise_stds,
            threshold=threshold,
            judge_provider=judge_provider,
            judge_model=judge_model,
        )

    # -----------------------------------------------------------------------
    # Output: Save per-question CSV
    # -----------------------------------------------------------------------
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(per_question_results)
    print(f"\n[OK] Per-question CSV report saved to: {output_csv_path}")

    # -----------------------------------------------------------------------
    # Output: Save baseline JSON
    # -----------------------------------------------------------------------
    run_payload = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "app_name": "RepoPilot (Codebase Intelligence Copilot)",
        "dataset_path": dataset_path,
        "dataset_stats": dataset_stats,
        "threshold": threshold,
        "judge_provider": actual_judge_prov,
        "judge_model": actual_judge_model,
        "judge_model_versions_observed": getattr(judge, "observed_model_versions", []),
        "generator_provider": per_question_results[0].get("generator_provider", "unknown") if per_question_results else "unknown",
        "generator_model": per_question_results[0].get("generator_model", "unknown") if per_question_results else "unknown",
        "noise_estimation_std": noise_stds,
        "summary": summary,
        "context_disagreements": disagreements,
        "like_for_like_comparisons": comparisons,
        "sanity_check": sanity_check_results,
        "experiments": experiments_data,
        "results": per_question_results,
    }

    # Rule 6: Track & report Ollama parse failures if applicable
    if hasattr(judge, "get_parse_failures"):
        pf = judge.get_parse_failures()
        run_payload["ollama_parse_failures"] = pf
        if any(pf.values()):
            print("\n" + "=" * 60)
            print(" OLLAMA JSON PARSE FAILURES REPORT (Handled via single retry)")
            print("=" * 60)
            for m_name, cnt in pf.items():
                print(f"  Metric '{m_name}': {cnt} invalid JSON responses recovered")
            print("=" * 60)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)
    print(f"[OK] Full baseline run JSON saved to: {output_json_path}")

    # -----------------------------------------------------------------------
    # Output: Print Summary Table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" RAG TRIAD SUMMARY TABLE (Full Golden Set)")
    print("=" * 90)
    hdr = f"| {'Metric':<25} | {'Mean':<7} | {'P50':<7} | {'StdDev':<7} | {'Min':<6} | {'Max':<6} | {'Pass Rate':<9} | {'Thresh':<7} |"
    sep = f"|{'-'*27}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*8}|{'-'*8}|{'-'*11}|{'-'*9}|"
    print(hdr)
    print(sep)
    for m_key, m_label in [
        ("context_relevancy", "Context Relevance (stmt)"),
        ("chunk_file_precision", "Chunk File Precision"),
        ("faithfulness", "Faithfulness"),
        ("answer_relevancy", "Answer Relevance"),
    ]:
        st = summary[m_key]
        print(
            f"| {m_label:<25} | {st['mean']:<7.4f} | {st['p50']:<7.4f} | {st['std_dev']:<7.4f} | "
            f"{st['min']:<6.3f} | {st['max']:<6.3f} | {st['pass_rate']*100:<8.1f}% | >= {st['threshold']:<5.2f} |"
        )
    print(sep)
    print(
        f"| {'Overall Triad Pass':<25} | {'-':<7} | {'-':<7} | {'-':<7} | "
        f"{'-':<6} | {'-':<6} | {summary['overall_pipeline']['pass_rate']*100:<8.1f}% | all >= {threshold} |"
    )
    print("=" * 90)

    # -----------------------------------------------------------------------
    # Output: Print Like-for-Like Baseline Comparison
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(" LIKE-FOR-LIKE COMPONENT BASELINE COMPARISON (Matching Question IDs Only)")
    print("=" * 90)
    for comp_key, comp_val in comparisons.items():
        print(f"\n-> {comp_key.upper()}:")
        print(f"   Matched Questions      : {comp_val['matched_question_count']}")
        print(f"   Pipeline Mean          : {comp_val['pipeline_mean']:.4f}")
        print(f"   Component Baseline Mean: {comp_val.get('component_gold_mean', comp_val.get('component_recall_mean', 0.0)):.4f}")
        print(f"   Delta (Pipeline - Base): {comp_val['delta']:+.4f}")
        print(f"   2x Noise Threshold (2σ): ±{comp_val['noise_threshold_2sigma']:.4f}")
        print(f"   Evaluation Status      : {comp_val['status']}")
    print("=" * 90 + "\n")

    return run_payload


# ---------------------------------------------------------------------------
# Run-to-Run Comparison Guard (Rule 8)
# ---------------------------------------------------------------------------

def compare_rag_triad_runs(
    baseline_run: dict[str, Any],
    candidate_run: dict[str, Any],
    label_baseline: str = "Baseline",
    label_candidate: str = "Candidate",
) -> dict[str, Any]:
    """Compare baseline run against candidate run, enforcing matching judge provider and model (Rule 8)."""
    assert_matching_judges(baseline_run, candidate_run, label_a=label_baseline, label_b=label_candidate)

    base_results = {r["id"]: r for r in baseline_run.get("results", [])}
    cand_results = {r["id"]: r for r in candidate_run.get("results", [])}
    common_qids = sorted(set(base_results.keys()) & set(cand_results.keys()))

    if not common_qids:
        raise ValueError("Cannot compare runs: no overlapping question IDs found.")

    b_summary = baseline_run.get("summary", {})
    c_summary = candidate_run.get("summary", {})

    deltas: dict[str, Any] = {}
    for metric in ["context_relevancy", "faithfulness", "answer_relevancy"]:
        b_val = b_summary.get(metric, {}).get("mean", 0.0)
        c_val = c_summary.get(metric, {}).get("mean", 0.0)
        deltas[metric] = {
            "baseline_mean": b_val,
            "candidate_mean": c_val,
            "delta": round(c_val - b_val, 4),
        }

    return {
        "matched_questions": len(common_qids),
        "judge_provider": baseline_run.get("judge_provider"),
        "judge_model": baseline_run.get("judge_model"),
        "metrics": deltas,
    }


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAG Triad evaluation on live end-to-end RAG pipeline using DeepEval."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_GOLDEN_DATASET,
        help="Path to golden dataset (JSONL).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Fixed threshold for pass/fail per metric (default: 0.7).",
    )
    parser.add_argument(
        "--include-unanswerable",
        action="store_true",
        default=False,
        help="Include unanswerable and out-of-scope rows (default: false, evaluates answerable cohort).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of questions evaluated for quick testing.",
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        default=None,
        help="Callable spec for live pipeline (e.g. 'my_module:rag_pipeline'). Defaults to RepoPilot live pipeline.",
    )
    parser.add_argument(
        "--judge-provider",
        type=str,
        default=None,
        choices=["gemini", "groq", "ollama"],
        help="Force a specific judge provider (default: selects first healthy in JUDGE_CHAIN).",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default=None,
        help="Judge model identifier (default: configured model for chosen judge provider).",
    )
    parser.add_argument(
        "--allow-generator-fallback",
        action="store_true",
        default=False,
        help="Allow per-call fallback across GENERATOR_CHAIN for answer generation (Rule 3).",
    )
    parser.add_argument(
        "--generator-provider",
        type=str,
        default=None,
        choices=["gemini", "groq", "ollama"],
        help="Initial generator provider (default: first provider in GENERATOR_CHAIN).",
    )
    parser.add_argument(
        "--skip-noise-estimation",
        action="store_true",
        default=False,
        help="Skip 3x repeat noise estimation step.",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        default=False,
        help="Skip stronger judge sanity check.",
    )
    parser.add_argument(
        "--skip-experiments",
        action="store_true",
        default=False,
        help="Skip retrieval candidate experiments.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Custom path to save the per-question CSV report.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Custom path to save the baseline JSON artifact.",
    )

    args = parser.parse_args()
    run_eval_rag_triad(
        dataset_path=args.dataset,
        threshold=args.threshold,
        include_unanswerable=args.include_unanswerable,
        limit=args.limit,
        pipeline_spec=args.pipeline,
        judge_provider=args.judge_provider,
        judge_model=args.judge_model,
        allow_generator_fallback=args.allow_generator_fallback,
        generator_provider=args.generator_provider,
        run_noise_estimation=not args.skip_noise_estimation,
        run_sanity_check=not args.skip_sanity_check,
        run_experiments=not args.skip_experiments,
        output_csv_path=args.output_csv,
        output_json_path=args.output_json,
    )


if __name__ == "__main__":
    main()
