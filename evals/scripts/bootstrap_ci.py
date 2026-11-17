"""Bootstrap confidence interval calculator for RepoPilot evaluation metrics.

Computes non-parametric bootstrap confidence intervals (95% CI) over per-query
results to generate statistically rigorous error bars.
"""

import argparse
import json
import os
import sys
import numpy as np

def compute_bootstrap_ci(
    run_file: str,
    stage: str = "reranked",
    n_bootstrap: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict:
    with open(run_file, "r", encoding="utf-8") as f:
        run_data = json.load(f)

    corpus = run_data.get("corpus", "unknown")
    level = run_data.get("level", "unknown")
    queries = run_data.get("query_details", [])

    if not queries:
        raise ValueError(f"No query_details found in {run_file}")

    metrics_list = ["Hit@1", "Hit@3", "Hit@5", "Recall@5", "Precision@5", "MRR", "nDCG@5"]

    # Gather per-query metric vectors
    query_metrics = {m: [] for m in metrics_list}
    for q in queries:
        stage_m = q.get("stages", {}).get(stage, {}).get("metrics", {})
        for m in metrics_list:
            if m in stage_m:
                query_metrics[m].append(stage_m[m])

    rng = np.random.default_rng(seed)
    n_queries = len(queries)
    alpha = 1.0 - confidence_level
    low_pct = (alpha / 2.0) * 100.0
    high_pct = (1.0 - alpha / 2.0) * 100.0

    ci_results = {}

    print("\n" + "=" * 80)
    print(f"BOOTSTRAP 95% CONFIDENCE INTERVALS ({corpus.upper()} | {level.upper()} | STAGE: {stage.upper()})")
    print(f"File: {os.path.basename(run_file)} | Queries: {n_queries} | Bootstrap Iterations: {n_bootstrap}")
    print("=" * 80)
    print(f"{'Metric':<14} | {'Mean':<8} | {'Std Error':<10} | {'95% CI [Low, High]':<22} | {'Formatted (Mean ± Margin)'}")
    print("-" * 80)

    for m in metrics_list:
        vals = np.array(query_metrics[m])
        if len(vals) == 0:
            continue

        sample_mean = float(np.mean(vals))

        # Bootstrap resampling
        boot_indices = rng.integers(0, n_queries, size=(n_bootstrap, n_queries))
        boot_means = np.mean(vals[boot_indices], axis=1)

        ci_low = float(np.percentile(boot_means, low_pct))
        ci_high = float(np.percentile(boot_means, high_pct))
        std_err = float(np.std(boot_means))
        margin = (ci_high - ci_low) / 2.0

        ci_results[m] = {
            "mean": round(sample_mean, 4),
            "std_error": round(std_err, 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "ci_margin": round(margin, 4),
        }

        print(f"{m:<14} | {sample_mean:<8.3f} | {std_err:<10.4f} | [{ci_low:.3f}, {ci_high:.3f}]              | {sample_mean:.3f} ± {margin:.3f}")

    print("=" * 80 + "\n")
    return {
        "run_file": run_file,
        "corpus": corpus,
        "level": level,
        "stage": stage,
        "n_queries": n_queries,
        "confidence_level": confidence_level,
        "ci_results": ci_results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute bootstrap confidence intervals")
    parser.add_argument("run_file", help="Path to evaluation run JSON file")
    parser.add_argument("--stage", default="reranked", choices=["dense", "fts", "fused", "reranked"])
    parser.add_argument("--bootstraps", type=int, default=2000)
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args()

    results = compute_bootstrap_ci(args.run_file, stage=args.stage, n_bootstrap=args.bootstraps)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Saved bootstrap CI to {args.out}")
