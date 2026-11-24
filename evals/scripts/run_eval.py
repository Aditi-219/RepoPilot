"""Evaluation runner for RepoPilot retrieval across all pipeline stages.

Computes deterministic IR metrics (Hit@k, Recall@k, Precision@k, MRR, nDCG@k)
across all 4 stages ('dense', 'fts', 'fused', 'reranked') and generates a category-level
breakdown table. Saves the full run results to evals/runs/.

Usage:
    python evals/scripts/run_eval.py [--level file|symbol] [--k-pool 50]
"""

import argparse
import datetime
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Any

# Add project roots to sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

from app.database import SessionLocal
from app.models import Repository
from app.reranker import rerank
from app.search import SearchResult, retrieve_all_stages
from metrics import compute_all_metrics, hit_at_k, recall_at_k

GOLDEN_V2_PATH = os.path.join(REPO_ROOT, "evals", "data", "golden_v2.jsonl")
DEFAULT_DATASET = GOLDEN_V2_PATH if os.path.exists(GOLDEN_V2_PATH) else os.path.join(REPO_ROOT, "evals", "data", "golden_v1.jsonl")
DEFAULT_SYMBOLS = os.path.join(REPO_ROOT, "evals", "data", "symbol_ranges.json")
RUNS_DIR = os.path.join(REPO_ROOT, "evals", "runs")


def load_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_symbol_ranges(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format an ASCII / Markdown table with aligned columns."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))

    header_str = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * widths[i] for i in range(len(headers)))
    row_strs = [" | ".join(str(val).ljust(widths[i]) for i, val in enumerate(r)) for r in rows]

    return f"| {header_str} |\n| {sep_str} |\n" + "\n".join(f"| {r} |" for r in row_strs)


def run_evaluation(
    dataset_path: str = DEFAULT_DATASET,
    symbols_path: str = DEFAULT_SYMBOLS,
    level: str = "file",
    corpus: str = "full",
    k_pool: int = 50,
    output_path: str | None = None,
    save_baseline: bool = False,
) -> dict[str, Any]:
    examples = load_dataset(dataset_path)
    symbol_ranges = load_symbol_ranges(symbols_path)

    # Resolve target repository from database for the specified corpus configuration
    db = SessionLocal()
    try:
        target_repo_name = "repopilot-code" if corpus == "code" else "repopilot"
        target_repo = db.query(Repository).filter(Repository.full_name == target_repo_name).first()
        if target_repo is None:
            raise ValueError(
                f"Repository '{target_repo_name}' for corpus '{corpus}' not found in database. "
                f"Run 'python evals/scripts/index_local_snapshot.py --corpus {corpus}' first."
            )
        target_repo_id = target_repo.id
    finally:
        db.close()

    # Filter to answerable questions with ground-truth files for retrieval evaluation
    eval_examples = [ex for ex in examples if ex.get("answerable", True) and ex.get("gt_files")]
    print("\n========================================================")
    print(f"RepoPilot Retrieval Benchmark v1.1 ({corpus.upper()} CORPUS | {level.upper()}-LEVEL)")
    print(f"Target Repository: {target_repo_name} (ID={target_repo_id})")
    print(f"Total benchmark questions: {len(examples)} (Answerable: {len(eval_examples)})")
    print(f"Candidate pool depth: k_pool = {k_pool}")
    print("========================================================\n")

    stages = ["dense", "fts", "fused", "reranked"]
    all_query_results = []

    # Aggregators: stage -> category -> metric -> list of values
    stage_cat_metrics = {
        st: defaultdict(lambda: defaultdict(list))
        for st in stages
    }

    pool_recalls = []
    pool_hits = []
    reranker_latencies = []
    improved_count = 0
    worse_count = 0
    same_count = 0
    failed_queries = []

    for idx, ex in enumerate(eval_examples, start=1):
        qid = ex["id"]
        cat = ex.get("category", "general")
        q = ex["question"]
        gt_files = ex["gt_files"]
        gt_symbols = ex.get("gt_symbols", [])

        print(f"[{idx:02d}/{len(eval_examples):02d}] ({cat.upper()}) {qid}: {q[:65]}...")

        stage_candidates = retrieve_all_stages(query=q, k_pool=k_pool, repository_id=target_repo_id)

        # 1. Candidate Pool Ceiling Evaluation (Depth k_pool)
        fused_pool = stage_candidates.get("fused", [])
        p_rec = recall_at_k(
            ranked_chunks=fused_pool,
            gt_files=gt_files,
            k=k_pool,
            gt_symbols=gt_symbols,
            symbol_ranges=symbol_ranges,
            level=level,
        )
        p_hit = hit_at_k(
            ranked_chunks=fused_pool,
            gt_files=gt_files,
            k=k_pool,
            gt_symbols=gt_symbols,
            symbol_ranges=symbol_ranges,
            level=level,
        )
        pool_recalls.append(p_rec)
        pool_hits.append(p_hit)

        # 2. Benchmark isolated reranker step latency
        fused_sr = [
            SearchResult(
                chunk_id=c.get("chunk_id", 0),
                repository_id=1,
                file_path=c.get("file_path", ""),
                start_line=c.get("start_line", 0),
                end_line=c.get("end_line", 0),
                content=c.get("content", ""),
                language=c.get("language"),
                score=c.get("score", 0.0),
            )
            for c in fused_pool
        ]
        t0 = time.perf_counter()
        rerank(q, fused_sr, top_k=k_pool)
        r_lat_ms = (time.perf_counter() - t0) * 1000.0
        reranker_latencies.append(r_lat_ms)

        query_record = {
            "id": qid,
            "category": cat,
            "question": q,
            "gt_files": gt_files,
            "gt_symbols": gt_symbols,
            "stages": {},
            "rerank_latency_ms": round(r_lat_ms, 2),
        }

        for st in stages:
            cand_list = stage_candidates.get(st, [])
            metrics = compute_all_metrics(
                ranked_chunks=cand_list,
                gt_files=gt_files,
                gt_symbols=gt_symbols,
                symbol_ranges=symbol_ranges,
                level=level,
            )
            query_record["stages"][st] = {
                "metrics": metrics,
                "top_5_retrieved": [
                    f"{c.get('file_path')}:L{c.get('start_line')}-L{c.get('end_line')} (score={c.get('score', 0):.4f})"
                    for c in cand_list[:5]
                ],
            }

            for m_name, m_val in metrics.items():
                stage_cat_metrics[st][cat][m_name].append(m_val)
                stage_cat_metrics[st]["ALL"][m_name].append(m_val)

        # Compare FUSED vs RERANKED MRR
        fused_mrr = query_record["stages"]["fused"]["metrics"]["MRR"]
        reranked_mrr = query_record["stages"]["reranked"]["metrics"]["MRR"]
        if reranked_mrr > fused_mrr:
            improved_count += 1
        elif reranked_mrr < fused_mrr:
            worse_count += 1
        else:
            same_count += 1

        # Track failed queries (Hit@5 == 0.0 or Hit@1 == 0.0)
        reranked_hit5 = query_record["stages"]["reranked"]["metrics"]["Hit@5"]
        if reranked_hit5 == 0.0:
            failed_queries.append({
                "id": qid,
                "category": cat,
                "question": q,
                "gt_files": gt_files,
                "gt_symbols": gt_symbols,
                "in_pool_50": p_hit > 0.0,
                "pool_recall_50": p_rec,
                "top_5": query_record["stages"]["reranked"]["top_5_retrieved"],
            })

        all_query_results.append(query_record)

    # Compute summary tables
    key_metrics = ["Hit@1", "Hit@3", "Hit@5", "ChunkMention@5", "ChunkHit@5", "Recall@5", "Precision@5", "MRR", "nDCG@5"]

    print("\n" + "=" * 80)
    print(f"BENCHMARK SUMMARY BY STAGE ({level.upper()}-LEVEL, MACRO-AVERAGE ACROSS ALL QUERIES)")
    print("=" * 80)

    summary_headers = ["Stage"] + key_metrics
    summary_rows = []
    summary_data = {}

    for st in stages:
        row = [st.upper()]
        stage_summary = {}
        for m in key_metrics:
            vals = stage_cat_metrics[st]["ALL"][m]
            avg = sum(vals) / len(vals) if vals else 0.0
            row.append(f"{avg:.3f}")
            stage_summary[m] = round(avg, 4)
        summary_rows.append(row)
        summary_data[st] = stage_summary

    print(format_table(summary_headers, summary_rows))

    # Candidate Pool Ceiling & Reranker Diagnostics
    avg_pool_recall = sum(pool_recalls) / len(pool_recalls) if pool_recalls else 0.0
    avg_pool_hit = sum(pool_hits) / len(pool_hits) if pool_hits else 0.0
    sorted_lats = sorted(reranker_latencies) if reranker_latencies else [0.0]
    p50_lat = statistics.median(sorted_lats)
    p95_idx = min(int(round(0.95 * len(sorted_lats))) - 1, len(sorted_lats) - 1)
    p95_lat = sorted_lats[max(0, p95_idx)]

    print("\n" + "=" * 80)
    print("CANDIDATE POOL CEILING & RERANKER ANALYSIS")
    print("=" * 80)
    print(f"  * Candidate Pool Ceiling (Depth k={k_pool}):")
    print(f"      - Pool Hit@{k_pool}:    {avg_pool_hit * 100:.1f}%")
    print(f"      - Pool Recall@{k_pool}: {avg_pool_recall * 100:.1f}%")
    print(f"  * Reranker Impact (FUSED -> RERANKED MRR):")
    print(f"      - Improved (Win):  {improved_count} queries")
    print(f"      - Worse (Loss):    {worse_count} queries")
    print(f"      - Tied (Same):     {same_count} queries")
    print(f"  * Reranker Latency per Query:")
    print(f"      - P50: {p50_lat:.2f} ms")
    print(f"      - P95: {p95_lat:.2f} ms")

    # Breakdown by category for the final reranked stage
    print("\n" + "=" * 80)
    print(f"RERANKED STAGE BREAKDOWN BY QUESTION CATEGORY ({level.upper()}-LEVEL)")
    print("=" * 80)

    cat_headers = ["Category", "Count"] + key_metrics
    cat_rows = []
    cat_data = {}

    categories = sorted({ex.get("category", "general") for ex in eval_examples})
    for cat in categories:
        cnt = len(stage_cat_metrics["reranked"][cat]["MRR"])
        row = [cat, str(cnt)]
        cat_metrics = {}
        for m in key_metrics:
            vals = stage_cat_metrics["reranked"][cat][m]
            avg = sum(vals) / len(vals) if vals else 0.0
            row.append(f"{avg:.3f}")
            cat_metrics[m] = round(avg, 4)
        cat_rows.append(row)
        cat_data[cat] = cat_metrics

    print(format_table(cat_headers, cat_rows))

    # Print failed query diagnostics
    if failed_queries:
        print("\n" + "=" * 80)
        print(f"FAILED QUERIES (Hit@5 == 0.0) — DIAGNOSTIC BREAKDOWN ({len(failed_queries)} queries)")
        print("=" * 80)
        for fq in failed_queries:
            status_desc = "Ceiling Miss (Not in Top-50 Pool)" if not fq["in_pool_50"] else "Ranking Miss (In Pool, not Top-5)"
            print(f"\n[{fq['id']}] ({fq['category'].upper()}) Status: {status_desc}")
            print(f"  Question:  {fq['question']}")
            print(f"  GT Files:   {fq['gt_files']}")
            if fq["gt_symbols"]:
                print(f"  GT Symbols: {fq['gt_symbols']}")
            print("  Top 5 Retrieved Chunks:")
            for rank_i, chunk_desc in enumerate(fq["top_5"], 1):
                print(f"    {rank_i}. {chunk_desc}")
    print("\n" + "=" * 80 + "\n")

    # Save run artifact
    os.makedirs(RUNS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file = output_path or os.path.join(RUNS_DIR, f"run_{corpus}_{level}_{timestamp}.json")

    # Stage comparison analysis
    dense_mrr = summary_data.get("dense", {}).get("MRR", 0.0)
    fts_mrr = summary_data.get("fts", {}).get("MRR", 0.0)
    fused_mrr = summary_data.get("fused", {}).get("MRR", 0.0)
    reranked_mrr = summary_data.get("reranked", {}).get("MRR", 0.0)

    stage_comparison = {
        "fused_beats_dense": fused_mrr >= dense_mrr,
        "fused_mrr_gain_over_dense": round(fused_mrr - dense_mrr, 4),
        "fused_beats_fts": fused_mrr >= fts_mrr,
        "fused_mrr_gain_over_fts": round(fused_mrr - fts_mrr, 4),
        "reranked_gain_over_fused": round(reranked_mrr - fused_mrr, 4),
        "pool_ceiling": {
            "depth": k_pool,
            "hit_at_pool": round(avg_pool_hit, 4),
            "recall_at_pool": round(avg_pool_recall, 4),
        },
        "reranker_lift": {
            "improved": improved_count,
            "worse": worse_count,
            "same": same_count,
        },
        "reranker_latency_ms": {
            "p50": round(p50_lat, 2),
            "p95": round(p95_lat, 2),
        },
        "verdict": (
            "RRF fusion outperforms individual dense and sparse retrievers."
            if (fused_mrr >= dense_mrr and fused_mrr >= fts_mrr)
            else "RRF fusion underperformed best individual retriever."
        ),
    }

    run_payload = {
        "version": "v1.1",
        "timestamp": timestamp,
        "corpus": corpus,
        "corpus_policy": (
            "Full-index: All eligible repository files including documentation and markdown"
            if corpus == "full"
            else "Code-only: Source code files only; excluded docs/ directory, all .md files, and docs/20-resume-and-interview-preparation.md"
        ),
        "exclusions": (
            ["docs/", "*.md", "docs/20-resume-and-interview-preparation.md"]
            if corpus == "code"
            else []
        ),
        "dataset_version": os.path.basename(dataset_path),
        "commit": "a36d617658de5414f15879ed53338d6f83e090e5",
        "repository": target_repo_name,
        "repository_id": target_repo_id,
        "level": level,
        "k_pool": k_pool,
        "settings": {
            "ENABLE_NEURAL_RERANKER": os.getenv("ENABLE_NEURAL_RERANKER", "false"),
            "RRF_K": 60,
        },
        "total_examples": len(examples),
        "evaluated_examples": len(eval_examples),
        "summary": summary_data,
        "stage_comparison": stage_comparison,
        "category_breakdown_reranked": cat_data,
        "failed_queries": failed_queries,
        "query_details": all_query_results,
    }

    with open(run_file, "w", encoding="utf-8") as out:
        json.dump(run_payload, out, indent=2)
    print(f"Detailed run artifacts saved to: {run_file}")

    if save_baseline:
        baseline_file = os.path.join(RUNS_DIR, f"retriever_baseline_{corpus}_{level}.json")
        with open(baseline_file, "w", encoding="utf-8") as out:
            json.dump(run_payload, out, indent=2)
        print(f"Corpus baseline saved to: {baseline_file}")

        # If evaluating full index, also update retriever_baseline_{level}.json for v1.0 compatibility
        if corpus == "full":
            legacy_baseline = os.path.join(RUNS_DIR, f"retriever_baseline_{level}.json")
            with open(legacy_baseline, "w", encoding="utf-8") as out:
                json.dump(run_payload, out, indent=2)
            print(f"Legacy baseline updated at: {legacy_baseline}")
        print()

    return run_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RepoPilot Retrieval Evaluation Runner (v1.1 Dual-Corpus)")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Path to golden questions JSONL")
    parser.add_argument("--corpus", choices=["full", "code"], default="full", help="Corpus configuration: 'full' or 'code'")
    parser.add_argument("--level", choices=["file", "symbol"], default="file", help="Relevance match level")
    parser.add_argument("--k-pool", type=int, default=50, help="Candidate pool size per stage")
    parser.add_argument("--output", type=str, default=None, help="Custom output JSON path")
    parser.add_argument("--save-baseline", action="store_true", help="Save run as baseline for this corpus and level")
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        corpus=args.corpus,
        level=args.level,
        k_pool=args.k_pool,
        output_path=args.output,
        save_baseline=args.save_baseline,
    )


