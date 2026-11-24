"""Compare two RepoPilot generator evaluation runs and show per-metric deltas.

Reads evals/runs/generator_noise.json (produced by --estimate-noise) to
suppress changes smaller than the judge's measurement noise floor.

Usage
-----
    # Compare current run against baseline
    python evals/scripts/compare_generator.py \\
        --baseline evals/runs/generator_baseline_gold.json \\
        --run      evals/runs/generator_gold_20261123_131308.json

    # Compare any two arbitrary runs
    python evals/scripts/compare_generator.py --baseline <path> --run <path>

Signal legend
-------------
  ✅ improved    — delta > 0 and outside judge noise floor
  ❌ regressed   — delta < 0 and outside judge noise floor
  within noise  — |delta| ≤ noise std dev; not a reliable change
  —             — no change (delta == 0)
  N/A           — metric absent in one or both runs
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS_DIR = os.path.join(REPO_ROOT, "evals", "runs")
NOISE_PATH = os.path.join(RUNS_DIR, "generator_noise.json")


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_noise() -> dict[str, float]:
    """Return per-metric noise std dev from the noise estimation file.

    Returns an empty dict when the file does not exist (noise floor unknown).
    """
    if not os.path.exists(NOISE_PATH):
        return {}
    data = _load_json(NOISE_PATH)
    return {
        metric: v.get("avg_std_dev", 0.0)
        for metric, v in data.get("per_metric_noise", {}).items()
    }


def _fmt(val: float | None, precision: int = 4) -> str:
    return f"{val:.{precision}f}" if val is not None else "N/A"


def _fmt_pct(val: float | None) -> str:
    return f"{val:.1%}" if val is not None else "N/A"


def _signal(delta: float, noise: float) -> str:
    if noise > 0 and abs(delta) <= noise:
        return "within noise"
    if delta > 0:
        return "✅ improved"
    if delta < 0:
        return "❌ regressed"
    return "—"


def _format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(str(val)))
    header_str = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    sep_str = "-+-".join("-" * widths[i] for i in range(len(headers)))
    row_strs = [
        " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r)) for r in rows
    ]
    return f"| {header_str} |\n| {sep_str} |\n" + "\n".join(
        f"| {r} |" for r in row_strs
    )


def compare(baseline_path: str, run_path: str) -> None:
    """Print a delta report between two generator evaluation runs."""
    baseline = _load_json(baseline_path)
    current = _load_json(run_path)
    noise = _load_noise()

    print("\n" + "=" * 72)
    print("RepoPilot Generator — Metric Delta Report")
    print("=" * 72)

    print(f"  Baseline : {os.path.basename(baseline_path)}")
    print(f"    timestamp  : {baseline.get('timestamp', 'unknown')}")
    print(f"    mode       : {baseline.get('contexts_mode', '?')}")
    print(f"    generator  : {baseline.get('generator_model', '?')}")
    print(f"    judge      : {baseline.get('judge_model', '?')}")
    print(f"    n_evaluated: {baseline.get('evaluated_answerable', '?')}")

    print(f"\n  Current  : {os.path.basename(run_path)}")
    print(f"    timestamp  : {current.get('timestamp', 'unknown')}")
    print(f"    mode       : {current.get('contexts_mode', '?')}")
    print(f"    generator  : {current.get('generator_model', '?')}")
    print(f"    judge      : {current.get('judge_model', '?')}")
    print(f"    n_evaluated: {current.get('evaluated_answerable', '?')}")

    if baseline.get("contexts_mode") != current.get("contexts_mode"):
        print(
            "\n  ⚠️  WARNING: runs use different contexts modes "
            f"({baseline.get('contexts_mode')} vs {current.get('contexts_mode')}). "
            "Comparison may not be meaningful.",
        )

    if noise:
        print(f"\n  Noise source : {os.path.relpath(NOISE_PATH)}")
        print(
            "  Changes ≤ noise std dev are marked 'within noise' "
            "and should not be treated as regressions."
        )
    else:
        print(
            "\n  ⚠️  No noise estimates found (generator_noise.json missing).\n"
            "     Run once with --estimate-noise to calibrate the noise floor.\n"
            "     Without it, all deltas are shown as-is with no significance filter."
        )

    # ---- Core metric table -----------------------------------------------
    print("\n" + "=" * 72)

    baseline_metrics: dict = baseline.get("metrics", {})
    current_metrics: dict = current.get("metrics", {})
    all_names = sorted(set(baseline_metrics) | set(current_metrics))

    headers = [
        "Metric", "Baseline Mean", "Current Mean", "Δ Mean", "Signal",
        "Baseline Pass%", "Current Pass%", "Δ Pass",
    ]
    rows: list[list[str]] = []

    for name in all_names:
        b = baseline_metrics.get(name, {})
        c = current_metrics.get(name, {})
        b_mean: float | None = b.get("mean")
        c_mean: float | None = c.get("mean")
        b_pass: float | None = b.get("pass_rate")
        c_pass: float | None = c.get("pass_rate")

        if b_mean is not None and c_mean is not None:
            delta = c_mean - b_mean
            noise_thr = noise.get(name, 0.0)
            sig = _signal(delta, noise_thr)
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "N/A"
            sig = "N/A"

        pass_delta_str = (
            f"{c_pass - b_pass:+.1%}" if b_pass is not None and c_pass is not None else "N/A"
        )

        rows.append([
            name,
            _fmt(b_mean),
            _fmt(c_mean),
            delta_str,
            sig,
            _fmt_pct(b_pass),
            _fmt_pct(c_pass),
            pass_delta_str,
        ])

    print(_format_table(headers, rows))

    # ---- Refusal / out-of-scope rows (not in averages) -------------------
    b_unans: float | None = baseline.get("unanswerable_pass_rate")
    c_unans: float | None = current.get("unanswerable_pass_rate")
    b_oos: float | None = baseline.get("out_of_scope_pass_rate")
    c_oos: float | None = current.get("out_of_scope_pass_rate")

    print("\n  Refusal metrics (excluded from averages above):")
    if b_unans is not None and c_unans is not None:
        d = c_unans - b_unans
        print(
            f"    unanswerable pass rate : "
            f"{b_unans:.1%} → {c_unans:.1%}  (Δ {d:+.1%})"
        )
    if b_oos is not None and c_oos is not None:
        d = c_oos - b_oos
        print(
            f"    out-of-scope pass rate : "
            f"{b_oos:.1%} → {c_oos:.1%}  (Δ {d:+.1%})"
        )

    # ---- Worst 5 from current run ----------------------------------------
    worst = current.get("worst_5", [])
    if worst:
        print("\n  Worst 5 questions in current run:")
        for entry in worst:
            print(
                f"    [{entry.get('id', '?')}] {entry.get('metric', '?'):22s}  "
                f"score={entry.get('score', 0):.3f}  "
                f"{str(entry.get('question', ''))[:55]}..."
            )

    print("\n" + "=" * 72 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare two RepoPilot generator evaluation runs",
    )
    parser.add_argument(
        "--baseline",
        required=True,
        metavar="PATH",
        help="Baseline JSON run file (e.g. evals/runs/generator_baseline_gold.json)",
    )
    parser.add_argument(
        "--run",
        required=True,
        metavar="PATH",
        help="Current run JSON file to compare against the baseline",
    )
    args = parser.parse_args()

    for label, path in [("--baseline", args.baseline), ("--run", args.run)]:
        if not os.path.exists(path):
            print(f"ERROR: {label} file not found: {path}", file=sys.stderr)
            sys.exit(1)

    compare(args.baseline, args.run)
