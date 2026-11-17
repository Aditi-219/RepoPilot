"""Evaluation script for RepoPilot reranker modes (Step 6).

Evaluates:
1. Ceiling (Recall@pool & Hit@pool at depth 50 to see if target was ever in the candidate pool).
2. Lift (MRR, Hit@1, Hit@3, nDCG@5 comparison: Reranker Off vs Heuristic vs Neural Cross-Encoder).
   Reports average delta and Win / Loss / Tie query counts.
3. Cost (P50 and P95 latency in ms per query across all modes).

Saves results to evals/runs/reranker_evaluation.json.
"""

import json
import os
import statistics
import sys
import time
from typing import Any

# Ensure project paths are in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

from app.reranker import _neural_rerank
from app.search import SearchResult, retrieve_all_stages
from metrics import hit_at_k, mrr, ndcg_at_k

DATASET_PATH = os.path.join(REPO_ROOT, "evals", "data", "golden_v1.jsonl")
SYMBOLS_PATH = os.path.join(REPO_ROOT, "evals", "data", "symbol_ranges.json")
OUTPUT_PATH = os.path.join(REPO_ROOT, "evals", "runs", "reranker_evaluation.json")


def load_dataset() -> list[dict[str, Any]]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_symbol_ranges() -> dict[str, dict[str, Any]]:
    if not os.path.exists(SYMBOLS_PATH):
        return {}
    with open(SYMBOLS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def rerank_heuristic(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Heuristic mode: deduplicate by file_path and line range, preserving RRF order."""
    seen = set()
    unique = []
    for c in candidates:
        key = (c.get("file_path"), c.get("start_line"), c.get("end_line"))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:top_k]


def rerank_neural(query: str, candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Neural mode: re-score candidates using cross-encoder/ms-marco-MiniLM-L-6-v2."""
    sr_objects = [
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
        for c in candidates
    ]
    ranked_sr = _neural_rerank(query, sr_objects, top_k=top_k)
    return [
        {
            "chunk_id": r.chunk_id,
            "file_path": r.file_path,
            "start_line": r.start_line,
            "end_line": r.end_line,
            "content": r.content,
            "score": r.score,
        }
        for r in ranked_sr
    ]


def evaluate_reranker(k_pool: int = 50, top_k: int = 5) -> dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 6: RERANKER IN-DEPTH EVALUATION (CEILING, LIFT & COST)")
    print(f"Candidate Pool Depth: {k_pool} | Final Truncation: top_{top_k}")
    print("=" * 80 + "\n")

    examples = load_dataset()
    eval_examples = [ex for ex in examples if ex.get("answerable", True) and ex.get("gt_files")]

    # Data structures for tracking
    ceiling_hits = []
    ceiling_miss_queries = []

    modes = ["off", "heuristic", "neural"]
    mode_metrics = {m: {"Hit@1": [], "Hit@3": [], "Hit@5": [], "MRR": [], "nDCG@5": []} for m in modes}
    mode_latencies = {m: [] for m in modes}
    query_comparisons = []

    for idx, ex in enumerate(eval_examples, start=1):
        qid = ex["id"]
        q = ex["question"]
        gt_files = ex["gt_files"]

        # 1. Fetch Candidate Pool via multi-stage retrieval
        t0 = time.perf_counter()
        stages = retrieve_all_stages(query=q, k_pool=k_pool)
        pool = stages.get("fused", [])
        base_retrieval_ms = (time.perf_counter() - t0) * 1000

        # Check Ceiling at pool depth
        hit_at_pool = hit_at_k(pool, gt_files, k=k_pool, level="file")
        ceiling_hits.append(hit_at_pool)
        if hit_at_pool == 0.0:
            ceiling_miss_queries.append({
                "id": qid,
                "category": ex.get("category"),
                "question": q,
                "gt_files": gt_files,
            })

        # --- Mode A: Reranker Off (raw RRF pool slice) ---
        t_off_0 = time.perf_counter()
        res_off = pool[:top_k]
        lat_off_ms = (time.perf_counter() - t_off_0) * 1000
        mode_latencies["off"].append(lat_off_ms)

        # --- Mode B: Heuristic Reranker ---
        t_heu_0 = time.perf_counter()
        res_heu = rerank_heuristic(pool, top_k=top_k)
        lat_heu_ms = (time.perf_counter() - t_heu_0) * 1000
        mode_latencies["heuristic"].append(lat_heu_ms)

        # --- Mode C: Neural Cross-Encoder Reranker ---
        t_neu_0 = time.perf_counter()
        res_neu = rerank_neural(q, pool, top_k=top_k)
        lat_neu_ms = (time.perf_counter() - t_neu_0) * 1000
        mode_latencies["neural"].append(lat_neu_ms)

        # Compute metrics for each mode
        def compute_m(ranked_list: list[dict[str, Any]], target_files: list[str]) -> dict[str, float]:
            return {
                "Hit@1": hit_at_k(ranked_list, target_files, 1, level="file"),
                "Hit@3": hit_at_k(ranked_list, target_files, 3, level="file"),
                "Hit@5": hit_at_k(ranked_list, target_files, 5, level="file"),
                "MRR": mrr(ranked_list, target_files, level="file"),
                "nDCG@5": ndcg_at_k(ranked_list, target_files, 5, level="file"),
            }

        m_off = compute_m(res_off, gt_files)
        m_heu = compute_m(res_heu, gt_files)
        m_neu = compute_m(res_neu, gt_files)

        for k_m, v in m_off.items():
            mode_metrics["off"][k_m].append(v)
        for k_m, v in m_heu.items():
            mode_metrics["heuristic"][k_m].append(v)
        for k_m, v in m_neu.items():
            mode_metrics["neural"][k_m].append(v)

        query_comparisons.append({
            "id": qid,
            "category": ex.get("category"),
            "question": q,
            "hit_at_pool": hit_at_pool,
            "metrics": {"off": m_off, "heuristic": m_heu, "neural": m_neu},
            "latencies_ms": {"retrieval": round(base_retrieval_ms, 2), "heuristic": round(lat_heu_ms, 2), "neural": round(lat_neu_ms, 2)},
        })

    total_q = len(eval_examples)

    # -------------------------------------------------------------------------
    # 1. CEILING ANALYSIS
    # -------------------------------------------------------------------------
    pool_hit_rate = sum(ceiling_hits) / total_q if total_q else 0.0
    print("[1] CEILING: Recall & Hit at Candidate Pool Depth (k=50)")
    print(f"  -> Candidate Pool Hit Rate (Hit@{k_pool}): {pool_hit_rate * 100:.2f}% ({sum(ceiling_hits):.0f}/{total_q})")
    print(f"  -> Queries where target file was MISSING in pool: {len(ceiling_miss_queries)}")
    for miss in ceiling_miss_queries:
        print(f"     * [{miss['id']}] ({miss['category']}): {miss['question'][:60]}... Target: {miss['gt_files']}")

    # -------------------------------------------------------------------------
    # 2. LIFT COMPARISON & WIN / LOSS / TIE ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[2] LIFT COMPARISON ACROSS RERANKER MODES (TOP-5)")
    print("=" * 80)

    summary_table = []
    metric_names = ["Hit@1", "Hit@3", "Hit@5", "MRR", "nDCG@5"]

    for m in modes:
        row = [m.upper()]
        for met in metric_names:
            vals = mode_metrics[m][met]
            row.append(f"{sum(vals)/len(vals):.3f}")
        summary_table.append(row)

    headers = ["Mode", "Hit@1", "Hit@3", "Hit@5", "MRR", "nDCG@5"]
    col_w = [12, 8, 8, 8, 8, 8]
    print(" | ".join(h.ljust(col_w[i]) for i, h in enumerate(headers)))
    print("-+-".join("-" * col_w[i] for i in range(len(headers))))
    for r in summary_table:
        print(" | ".join(str(val).ljust(col_w[i]) for i, val in enumerate(r)))

    # Win / Loss / Tie analysis (Neural vs Fused / Heuristic)
    def compare_pairs(target_mode, baseline_mode, metric="MRR"):
        improved = 0
        worse = 0
        tied = 0
        for i in range(total_q):
            t_val = mode_metrics[target_mode][metric][i]
            b_val = mode_metrics[baseline_mode][metric][i]
            diff = t_val - b_val
            if diff > 1e-4:
                improved += 1
            elif diff < -1e-4:
                worse += 1
            else:
                tied += 1
        return improved, worse, tied

    heu_imp, heu_wor, heu_tie = compare_pairs("heuristic", "off", "MRR")
    neu_imp, neu_wor, neu_tie = compare_pairs("neural", "heuristic", "MRR")

    print("\n  Win / Loss / Tie Analysis on MRR:")
    print(f"  * Heuristic vs Off:      Improved: {heu_imp}, Regressed: {heu_wor}, Tied: {heu_tie}")
    print(f"  * Neural vs Heuristic:   Improved: {neu_imp}, Regressed: {neu_wor}, Tied: {neu_tie}")

    # -------------------------------------------------------------------------
    # 3. COST: LATENCY ANALYSIS (P50 & P95)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[3] COST: ADDED LATENCY PER QUERY (P50 & P95)")
    print("=" * 80)

    latency_stats = {}
    for m in modes:
        lats = mode_latencies[m]
        p50 = statistics.median(lats) if lats else 0.0
        p95 = statistics.quantiles(lats, n=20)[18] if len(lats) >= 20 else max(lats) if lats else 0.0
        mean_lat = statistics.mean(lats) if lats else 0.0
        latency_stats[m] = {"p50_ms": round(p50, 2), "p95_ms": round(p95, 2), "mean_ms": round(mean_lat, 2)}
        print(f"  * Mode: {m.upper():<10} -> P50: {p50:6.2f} ms | P95: {p95:6.2f} ms | Mean: {mean_lat:6.2f} ms")

    # Assemble comprehensive report
    report = {
        "ceiling_analysis": {
            "candidate_pool_depth": k_pool,
            "pool_hit_rate": round(pool_hit_rate, 4),
            "missing_queries_count": len(ceiling_miss_queries),
            "missing_queries": ceiling_miss_queries,
        },
        "lift_comparison": {
            "macro_averages": {
                m: {met: round(sum(mode_metrics[m][met]) / total_q, 4) for met in metric_names}
                for m in modes
            },
            "win_loss_tie_mrr": {
                "heuristic_vs_off": {"improved": heu_imp, "regressed": heu_wor, "tied": heu_tie},
                "neural_vs_heuristic": {"improved": neu_imp, "regressed": neu_wor, "tied": neu_tie},
            },
        },
        "cost_latencies": latency_stats,
        "query_details": query_comparisons,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        json.dump(report, out, indent=2)

    print(f"\nReranker evaluation report written to: {OUTPUT_PATH}\n")
    return report


if __name__ == "__main__":
    evaluate_reranker()
