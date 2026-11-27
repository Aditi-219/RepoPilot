"""Application-level evaluation — Phase 2: Quality Judging via DeepEval G-Eval.

Reads a cached generation JSONL file produced by Phase 1
(`run_application_generate.py`) and scores every *successful* response using
three separate DeepEval G-Eval metrics backed by a single fixed judge
(`gemini-3.5-flash-lite`):

1. Correctness   (reference-based: final answer vs expected answer)
2. Completeness  (reference-based: key-point & multi-part coverage)
3. Style         (reference-free: code-copilot tone, citations, formatting, refusal)

Error rows (`error=True`) are NOT silently dropped: they are excluded from LLM
judging, counted in Operations availability / error_rate, and listed explicitly.

Usage
-----
    # Dry run — print estimated Gemini judge requests (0 network calls)
    python evals/scripts/run_application_judge.py --input evals/runs/application_<ts>.jsonl --dry-run

    # Judge Pass 1 (32 * 3 = 96 requests)
    python evals/scripts/run_application_judge.py --input evals/runs/application_<ts>.jsonl --baseline-save

    # Judge noise estimate (re-judges cached outputs N=2 times using distinct cache prefixes)
    python evals/scripts/run_application_judge.py --input evals/runs/application_<ts>.jsonl --estimate-noise --noise-passes 2
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import math
import os
import re
import statistics
import sys
from typing import Any

from dotenv import load_dotenv

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

load_dotenv(os.path.join(REPO_ROOT, "backend", ".env"))

from run_application_generate import (  # noqa: E402
    PACIFIC_TZ,
    QUOTA_LEDGER_PATH,
    _percentile,
    append_quota_ledger,
    classify_slice,
)

# DeepEval imports with graceful fallback so unit tests run without network/deepeval
try:
    from deepeval.metrics import GEval  # noqa: E402
    from deepeval.test_case import LLMTestCase  # noqa: E402

    try:
        from deepeval.test_case import SingleTurnParams as LLMTestCaseParams  # noqa: E402
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams  # type: ignore[no-redef]

    DEEPEVAL_AVAILABLE = True
except Exception:
    DEEPEVAL_AVAILABLE = False
    GEval = None  # type: ignore[assignment,misc]
    LLMTestCase = None  # type: ignore[assignment,misc]
    LLMTestCaseParams = None  # type: ignore[assignment,misc]

RUNS_DIR = os.path.join(REPO_ROOT, "evals", "runs")
THRESHOLDS_PATH = os.path.join(REPO_ROOT, "evals", "data", "application_thresholds.json")
BASELINE_PATH = os.path.join(RUNS_DIR, "application_baseline.json")
NOISE_PATH = os.path.join(RUNS_DIR, "application_noise.json")
JUDGE_CACHE_PATH = os.path.join(RUNS_DIR, "application_judge_cache.json")

FIXED_JUDGE_MODEL = "gemini-3.5-flash-lite"

SELF_JUDGING_NOTE = (
    "LIMITATION: Both the chat generator (gemini-3.5-flash-lite, temp 0.2) and the "
    "judge (gemini-3.5-flash-lite, temp 0.0) belong to the same model family, "
    "introducing self-judging bias. These scores are NOT comparable to earlier "
    "Ollama (qwen2.5-coder:1.5b) generator baselines."
)


# ---------------------------------------------------------------------------
# Pure loading, deduplication, and aggregation helpers (unit-tested)
# ---------------------------------------------------------------------------

def load_thresholds(path: str = THRESHOLDS_PATH) -> dict[str, Any]:
    """Load quality, safety, and operations thresholds from JSON config."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_cached_run(path: str) -> list[dict[str, Any]]:
    """Load JSONL records from a Phase 1 generation cache file."""
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def deduplicate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the latest record per question_id (preserving order of last seen)."""
    latest: dict[str, dict[str, Any]] = {}
    for rec in records:
        qid = rec.get("question_id")
        if qid:
            latest[qid] = rec
    return list(latest.values())


def split_success_and_errors(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Separate successful rows (to judge) from error rows (tracked in ops)."""
    deduped = deduplicate_records(records)
    successes = [r for r in deduped if not r.get("error")]
    errors = [r for r in deduped if r.get("error")]
    return successes, errors


def estimate_judge_requests(num_success_rows: int, passes: int = 1, metrics_per_row: int = 3) -> int:
    """Estimate Gemini judge API calls (1 call per metric per row per pass)."""
    return max(0, num_success_rows) * max(1, passes) * max(1, metrics_per_row)


def select_pilot_rows(success_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select 3 pilot rows: 1 answerable, 1 unanswerable, plus rp-021 (out_of_scope)."""
    by_id = {r.get("question_id"): r for r in success_rows}
    ans = next((r for r in success_rows if classify_slice(r) == "answerable"), None)
    unans = next((r for r in success_rows if classify_slice(r) == "unanswerable"), None)
    oos = by_id.get("rp-021") or next((r for r in success_rows if classify_slice(r) == "out_of_scope"), None)
    selected: list[dict[str, Any]] = []
    for item in (ans, unans, oos):
        if item is not None and item["question_id"] not in {x["question_id"] for x in selected}:
            selected.append(item)
    return selected


def sum_today_quota(ledger_path: str = QUOTA_LEDGER_PATH) -> int:
    """Return total requests logged for the current Gemini quota day (midnight Pacific)."""
    if not os.path.exists(ledger_path):
        return 0
    today_pacific = datetime.datetime.now(PACIFIC_TZ).date().isoformat()
    local_tz = datetime.datetime.now().astimezone().tzinfo
    total = 0
    with open(ledger_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if entry.get("quota_tz") == "America/Los_Angeles":
                    entry_date = entry.get("date")
                elif entry.get("timestamp"):
                    dt = datetime.datetime.fromisoformat(entry["timestamp"])
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=local_tz)
                    entry_date = dt.astimezone(PACIFIC_TZ).date().isoformat()
                else:
                    entry_date = entry.get("date")
                if entry_date == today_pacific:
                    total += int(entry.get("requests_used") or 0)
            except (json.JSONDecodeError, ValueError):
                pass
    return total


def compute_ops_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute availability, token usage, and latency breakdown from cached records.

    Retried rows (`retried=True`) are excluded from P50/P95/P99 latency stats so
    retry overhead does not skew latency, and the exclusion count is recorded.
    """
    deduped = deduplicate_records(records)
    total = len(deduped)
    successes = [r for r in deduped if not r.get("error")]
    errors = [r for r in deduped if r.get("error")]
    failed_ids = [r.get("question_id") for r in errors if r.get("question_id")]

    latency_rows = [r for r in successes if not r.get("retried")]
    retried_excluded = sum(1 for r in successes if r.get("retried"))

    latencies = [float(r["total_latency_s"]) for r in latency_rows if r.get("total_latency_s") is not None]
    ttft_tools = [float(r["time_to_first_tool_s"]) for r in latency_rows if r.get("time_to_first_tool_s") is not None]

    tool_times: list[float] = []
    llm_times: list[float] = []
    for r in latency_rows:
        tot = r.get("total_latency_s")
        durs = [float(d["duration_s"]) for d in (r.get("tool_durations") or []) if d.get("duration_s") is not None]
        if tot is not None:
            t_sum = round(sum(durs), 3)
            tool_times.append(t_sum)
            llm_times.append(round(max(0.0, float(tot) - t_sum), 3))

    in_toks = [int(r["input_tokens"]) for r in successes if r.get("input_tokens") is not None]
    out_toks = [int(r["output_tokens"]) for r in successes if r.get("output_tokens") is not None]

    return {
        "total_questions": total,
        "success_count": len(successes),
        "error_count": len(errors),
        "failed_ids": failed_ids,
        "availability_rate": round(len(successes) / total, 4) if total else 0.0,
        "error_rate": round(len(errors) / total, 4) if total else 0.0,
        "retried_excluded_from_latency": retried_excluded,
        "p50_latency_s": _percentile(latencies, 50),
        "p95_latency_s": _percentile(latencies, 95),
        "p99_latency_s": _percentile(latencies, 99),
        "p50_ttft_tool_s": _percentile(ttft_tools, 50),
        "p95_ttft_tool_s": _percentile(ttft_tools, 95),
        "mean_tool_time_s": round(sum(tool_times) / len(tool_times), 3) if tool_times else None,
        "mean_llm_time_s": round(sum(llm_times) / len(llm_times), 3) if llm_times else None,
        "mean_input_tokens": round(sum(in_toks) / len(in_toks), 1) if in_toks else None,
        "max_input_tokens": max(in_toks) if in_toks else None,
        "mean_output_tokens": round(sum(out_toks) / len(out_toks), 1) if out_toks else None,
        "max_output_tokens": max(out_toks) if out_toks else None,
    }


DEFAULT_WORD_BUDGET = 300  # Heuristic upper bound for concise code-copilot answers

BANNED_FILLER_PHRASES = (
    "great question",
    "sure!",
    "certainly!",
    "happy to help",
    "of course!",
    "as an ai",
    "i'd be happy to",
)

# Matches either `path/file.py:L193-L245` / path/file.py:L14 / path/file.py (lines 1-10)
# or markdown links like [path/file.py:L11-L25](path/file.py) / [label](path/file.py#L11-L25)
_CITATION_LINE_RE = re.compile(
    r"(?:[a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+"
    r"(?::L?\d+(?:[-–]L?\d+)?|#L\d+(?:[-–]L?\d+)?|\s*\(lines?\s+\d+(?:[-–]\d+)?\)))",
    re.IGNORECASE,
)


_FENCED_CODE_RE = re.compile(r"```[\s\S]*?```", re.MULTILINE)


def split_prose_and_code_words(text: str) -> tuple[int, int, int]:
    """Return (prose_words, code_words, total_words) separating fenced code blocks."""
    t = text or ""
    total_words = len(t.split())
    code_blocks = _FENCED_CODE_RE.findall(t)
    code_words = sum(len(blk.split()) for blk in code_blocks)
    prose_only = _FENCED_CODE_RE.sub(" ", t)
    prose_words = len(prose_only.split())
    return prose_words, code_words, total_words


def has_citation_with_lines(text: str) -> bool:
    """Return True if text cites at least one file path with a line or line range."""
    return bool(_CITATION_LINE_RE.search(text or ""))


def identifiers_in_backticks(text: str) -> bool:
    """Return True if text uses markdown backticks (inline or fenced code)."""
    t = text or ""
    return "`" in t


def within_word_budget(text: str, budget: int = DEFAULT_WORD_BUDGET) -> tuple[bool, int, int, int]:
    """Return (prose_words <= budget, prose_words, code_words, total_words)."""
    prose_w, code_w, total_w = split_prose_and_code_words(text)
    return prose_w <= budget, prose_w, code_w, total_w


def no_banned_filler(text: str) -> tuple[bool, list[str]]:
    """Return (True if no banned filler found, list of matched phrases)."""
    lower = (text or "").lower()
    found = [p for p in BANNED_FILLER_PHRASES if p in lower]
    return len(found) == 0, found


def evaluate_deterministic_style(
    text: str,
    word_budget: int = DEFAULT_WORD_BUDGET,
) -> dict[str, Any]:
    """Run all 4 deterministic Style v2 checks on one answer (0 LLM calls)."""
    cit_ok = has_citation_with_lines(text)
    bt_ok = identifiers_in_backticks(text)
    wb_ok, prose_w, code_w, total_w = within_word_budget(text, budget=word_budget)
    filler_ok, matched_filler = no_banned_filler(text)
    return {
        "has_citation_with_lines": cit_ok,
        "identifiers_in_backticks": bt_ok,
        "within_word_budget": wb_ok,
        "prose_words": prose_w,
        "code_words": code_w,
        "total_words": total_w,
        "word_budget_heuristic": word_budget,
        "no_banned_filler": filler_ok,
        "matched_filler": matched_filler,
    }


def summarize_deterministic_style(
    rows: list[dict[str, Any]],
    word_budget: int = DEFAULT_WORD_BUDGET,
) -> dict[str, Any]:
    """Run deterministic style checks across answerable rows and summarize pass rates."""
    ans_rows = [r for r in rows if classify_slice(r) == "answerable" and not r.get("error")]
    per_row: list[dict[str, Any]] = []
    checks = (
        "has_citation_with_lines",
        "identifiers_in_backticks",
        "within_word_budget",
        "no_banned_filler",
    )
    for r in ans_rows:
        res = evaluate_deterministic_style(r.get("final_answer", ""), word_budget=word_budget)
        failed = [c for c in checks if not res[c]]
        per_row.append({
            "question_id": r["question_id"],
            **res,
            "failed_checks": failed,
        })

    n = len(per_row)
    pass_rates = {
        c: round(sum(1 for x in per_row if x[c]) / n, 4) if n else 0.0
        for c in checks
    }
    pass_counts = {
        c: sum(1 for x in per_row if x[c])
        for c in checks
    }
    prose_wcs = sorted(x["prose_words"] for x in per_row)
    failures = [x for x in per_row if x["failed_checks"]]
    return {
        "n_answerable": n,
        "word_budget_heuristic": word_budget,
        "pass_counts": pass_counts,
        "pass_rates": pass_rates,
        "prose_word_count_stats": {
            "min": prose_wcs[0] if prose_wcs else 0,
            "p50": _percentile([float(w) for w in prose_wcs], 50),
            "p75": _percentile([float(w) for w in prose_wcs], 75),
            "p90": _percentile([float(w) for w in prose_wcs], 90),
            "max": prose_wcs[-1] if prose_wcs else 0,
            "over_budget_count": sum(1 for w in prose_wcs if w > word_budget),
        },
        "per_row": per_row,
        "failures": failures,
    }


_COVERED_RE = re.compile(r"\[COVERED\]", re.IGNORECASE)
_MISSING_RE = re.compile(r"\[MISSING\]", re.IGNORECASE)


def compute_completeness_from_reason(reason: str) -> tuple[float | None, int, int]:
    """Count [COVERED] and [MISSING] flags in Completeness v2 reason text.

    Returns (covered / total, covered_count, total_count), or (None, 0, 0) if
    no flags were found.
    """
    text = reason or ""
    covered = len(_COVERED_RE.findall(text))
    missing = len(_MISSING_RE.findall(text))
    total = covered + missing
    if total == 0:
        return None, 0, 0
    return round(covered / total, 4), covered, total


def check_completeness_mismatch(
    geval_score: float | None,
    code_score: float | None,
    tolerance: float = 0.05,
) -> bool:
    """Return True if G-Eval Completeness score diverges from covered/total in code."""
    if geval_score is None or code_score is None:
        return False
    return abs(float(geval_score) - float(code_score)) > tolerance


_LINE_CITATION_SPAN_RE = re.compile(
    r"(?::L?\d+(?:[-–]L?\d+)?|#L\d+(?:[-–]L?\d+)?|(?<![A-Za-z0-9_-])L\d+(?:[-–]L?\d+)?(?![A-Za-z0-9_-])|\(lines?\s+\d+(?:[-–]\d+)?\))",
    re.IGNORECASE,
)


def _normalize_for_literal(text: str, strip_line_citations: bool = False) -> str:
    """Normalize text for literal matching: optional line-span strip, strip backticks,
    collapse whitespace, normalize '=' spacing, and lowercase.
    """
    t = text or ""
    if strip_line_citations:
        t = _LINE_CITATION_SPAN_RE.sub(" ", t)
    t = t.replace("`", "")
    t = re.sub(r"\s*=\s*", "=", t)
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def _boundary_match(haystack: str, needle: str) -> bool:
    """Return True if needle appears in haystack with [A-Za-z0-9_] boundaries."""
    if not needle:
        return False
    pattern = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(needle) + r"(?![A-Za-z0-9_])")
    return bool(pattern.search(haystack))


def check_literal_key_points(
    answer: str,
    key_points: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Check literal key points with identifier boundaries, normalization, and aliases.

    - Identifier boundaries: match cannot be preceded or followed by [A-Za-z0-9_].
    - Normalizes backticks, whitespace, '=' spacing, and case.
    - Strips line-citation spans (:L12-L40, L193-L245) before matching numeric points
      (or any non-path literal).
    - Supports optional 'aliases' list and optional 'raw_match' list (checks raw
      lowercased answer before backtick stripping, e.g. for `repositories`).
    """
    raw_lower = (answer or "").lower()
    norm_with_lines = _normalize_for_literal(answer, strip_line_citations=False)
    norm_no_lines = _normalize_for_literal(answer, strip_line_citations=True)

    literal_results: list[dict[str, Any]] = []
    semantic_points: list[dict[str, Any]] = []
    for kp in key_points:
        if kp.get("kind") == "literal":
            hit: str | None = None
            raw_candidates = kp.get("raw_match") or []
            for rc in raw_candidates:
                if _boundary_match(raw_lower, rc.lower()):
                    hit = rc
                    break

            if hit is None:
                candidates = list(kp.get("match") or []) + list(kp.get("aliases") or [])
                for cand in candidates:
                    # Keep line numbers intact only if the candidate itself is a file path
                    is_path = "/" in cand or "\\" in cand
                    hay = norm_with_lines if is_path else norm_no_lines
                    norm_cand = _normalize_for_literal(cand, strip_line_citations=False)
                    if _boundary_match(hay, norm_cand):
                        hit = cand
                        break

            literal_results.append({
                "text": kp["text"],
                "kind": "literal",
                "covered": hit is not None,
                "matched": hit,
            })
        else:
            semantic_points.append(kp)
    return literal_results, semantic_points


def estimate_v2_rejudge_requests(
    golden_path: str,
    disputed_ids: tuple[str, ...] = ("rp-003", "rp-009", "rp-027"),
) -> dict[str, Any]:
    """Compute exact Rubric v2 re-judge request counts split by non-disputed vs disputed."""
    rows = load_cached_run(golden_path)
    disputed_set = set(disputed_ids)

    def _count_group(ans_subset: list[dict[str, Any]]) -> dict[str, int]:
        corr = len(ans_subset)
        tone = len(ans_subset)
        comp = sum(
            1 for r in ans_subset
            if any(isinstance(kp, dict) and kp.get("kind") == "semantic" for kp in (r.get("key_points") or []))
            or not r.get("key_points")
        )
        return {"rows": len(ans_subset), "correctness": corr, "style_tone": tone, "completeness_semantic": comp}

    ans_all = [r for r in rows if r.get("answerable", True)]
    ref_all = [r for r in rows if not r.get("answerable", True)]

    non_disputed_ans = [r for r in ans_all if r["id"] not in disputed_set]
    disputed_ans = [r for r in ans_all if r["id"] in disputed_set]

    g_a = _count_group(non_disputed_ans)
    g_a["refusal"] = len(ref_all)
    g_a["total"] = g_a["correctness"] + g_a["style_tone"] + g_a["completeness_semantic"] + g_a["refusal"]
    g_a["total_with_10pct_buffer"] = math.ceil(g_a["total"] * 1.10)

    g_b = _count_group(disputed_ans)
    g_b["total"] = g_b["correctness"] + g_b["style_tone"] + g_b["completeness_semantic"]
    g_b["total_with_10pct_buffer"] = math.ceil(g_b["total"] * 1.10)

    combined = g_a["total"] + g_b["total"]
    return {
        "non_disputed_and_refusal": g_a,
        "disputed_waiting_v3": g_b,
        "combined_total": combined,
        "combined_with_10pct_buffer": math.ceil(combined * 1.10),
    }


def is_strict_pass(row: dict[str, Any]) -> bool:
    """Row-level headline strict pass: Correctness >= 0.8 AND Completeness == 1.0."""
    metrics = row.get("metrics") or {}
    corr = metrics.get("correctness", {}).get("score")
    comp = metrics.get("completeness", {}).get("score")
    if corr is None or comp is None:
        return False
    return float(corr) >= 0.8 and float(comp) == 1.0


def _mean_or_none(vals: list[float]) -> float | None:
    return round(sum(vals) / len(vals), 4) if vals else None


def compute_score_distribution(
    judged_rows: list[dict[str, Any]],
    slice_name: str = "answerable",
) -> dict[str, Any]:
    """Compute exact 1.0/0.0 counts, 0.2-bucket histogram, and >70% ceiling flag."""
    rows = [r for r in judged_rows if classify_slice(r) == slice_name]
    dist: dict[str, Any] = {}
    for m in ("correctness", "completeness", "style"):
        scores = [
            float(r["metrics"][m]["score"])
            for r in rows
            if r.get("metrics", {}).get(m, {}).get("score") is not None
        ]
        n = len(scores)
        exact_1 = sum(1 for s in scores if s == 1.0)
        exact_0 = sum(1 for s in scores if s == 0.0)
        buckets = {
            "[0.0,0.2)": 0,
            "[0.2,0.4)": 0,
            "[0.4,0.6)": 0,
            "[0.6,0.8)": 0,
            "[0.8,1.0]": 0,
        }
        for s in scores:
            if s < 0.2:
                buckets["[0.0,0.2)"] += 1
            elif s < 0.4:
                buckets["[0.2,0.4)"] += 1
            elif s < 0.6:
                buckets["[0.4,0.6)"] += 1
            elif s < 0.8:
                buckets["[0.6,0.8)"] += 1
            else:
                buckets["[0.8,1.0]"] += 1
        dist[m] = {
            "n": n,
            "exact_1_0": exact_1,
            "exact_0_0": exact_0,
            "exact_1_0_pct": round(exact_1 / n, 4) if n else 0.0,
            "ceiling_effect_flag": (exact_1 / n > 0.70) if n else False,
            "histogram": buckets,
        }
    return dist


def aggregate_judge_results(
    judged_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate per-question G-Eval scores overall and by slice, plus failing items."""
    metric_names = ("correctness", "completeness", "style")

    def _agg_subset(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"count": len(rows)}
        for m in metric_names:
            scores = [
                float(r["metrics"][m]["score"])
                for r in rows
                if r.get("metrics", {}).get(m, {}).get("score") is not None
            ]
            out[m] = {
                "mean": _mean_or_none(scores),
                "min": round(min(scores), 4) if scores else None,
                "max": round(max(scores), 4) if scores else None,
                "n": len(scores),
            }
        return out

    overall = _agg_subset(judged_rows)
    slices: dict[str, Any] = {}
    for s_name in ("answerable", "unanswerable", "out_of_scope"):
        s_rows = [r for r in judged_rows if classify_slice(r) == s_name]
        slices[s_name] = _agg_subset(s_rows)

    # Refusal pass rate across unanswerable + out_of_scope (correctness >= 0.5)
    refusal_rows = [r for r in judged_rows if classify_slice(r) in ("unanswerable", "out_of_scope")]
    refusal_passes = [
        r for r in refusal_rows
        if (r.get("metrics", {}).get("correctness", {}).get("score") or 0.0) >= 0.5
    ]
    refusal_pass_rate = round(len(refusal_passes) / len(refusal_rows), 4) if refusal_rows else None

    # Flag failing items (any metric below 0.5, or any refusal failure)
    failing_items: list[dict[str, Any]] = []
    for r in judged_rows:
        m_scores = {
            m: r.get("metrics", {}).get(m, {}).get("score")
            for m in metric_names
        }
        low = [m for m, s in m_scores.items() if s is not None and s < 0.5]
        if low:
            failing_items.append({
                "question_id": r["question_id"],
                "slice": classify_slice(r),
                "failed_metrics": low,
                "scores": m_scores,
                "answer_preview": (r.get("final_answer") or "")[:120],
            })

    q_thresh = thresholds.get("quality", {})
    passed_thresholds = {
        m: (
            overall[m]["mean"] is not None
            and overall[m]["mean"] >= q_thresh.get(m, {}).get("min_mean", 0.0)
        )
        for m in metric_names
    }
    passed_thresholds["refusal_pass_rate"] = (
        refusal_pass_rate is not None
        and refusal_pass_rate >= q_thresh.get("refusal_pass_rate", {}).get("min_rate", 0.0)
    )

    return {
        "overall": overall,
        "slices": slices,
        "answerable_score_distribution": compute_score_distribution(judged_rows, "answerable"),
        "refusal_pass_rate": refusal_pass_rate,
        "failing_items": failing_items,
        "error_rows_excluded_from_judge": [
            {"question_id": e.get("question_id"), "error_message": e.get("error_message")}
            for e in error_rows
        ],
        "passed_thresholds": passed_thresholds,
    }


def compute_judge_noise(
    pass_results: list[list[dict[str, Any]]],
    noise_multiplier: float = 2.0,
) -> dict[str, Any]:
    """Compute per-metric judge noise (mean per-question std dev across N passes)."""
    metric_names = ("correctness", "completeness", "style")
    by_qid: dict[str, dict[str, list[float]]] = {}
    for p_rows in pass_results:
        for r in p_rows:
            qid = r["question_id"]
            by_qid.setdefault(qid, {m: [] for m in metric_names})
            for m in metric_names:
                s = r.get("metrics", {}).get(m, {}).get("score")
                if s is not None:
                    by_qid[qid][m].append(float(s))

    noise: dict[str, Any] = {}
    for m in metric_names:
        stds = [
            statistics.stdev(vals[m])
            for vals in by_qid.values()
            if len(vals[m]) >= 2
        ]
        avg_std = round(sum(stds) / len(stds), 4) if stds else 0.0
        noise[m] = {
            "avg_std_dev": avg_std,
            "regression_threshold": round(avg_std * noise_multiplier, 4),
            "n_questions": len(stds),
            "n_passes": len(pass_results),
        }
    return noise


def compute_generation_noise(
    judged_gen_runs: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compute cross-generation score std dev across 2+ independent generation runs."""
    return compute_judge_noise(judged_gen_runs, noise_multiplier=2.0)


# ---------------------------------------------------------------------------
# DeepEval G-Eval metric builders (3 separate metrics with explicit steps)
# ---------------------------------------------------------------------------

def build_geval_metrics(judge_llm: Any) -> dict[str, Any]:
    """Create the three separate G-Eval metrics using explicit evaluation_steps."""
    if not DEEPEVAL_AVAILABLE:
        raise RuntimeError("deepeval is required to run judging. Install evals/requirements.txt.")

    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Read the user question and the expected (reference) answer.",
            "Compare the actual answer against the expected answer for factual accuracy.",
            "Penalize any factual contradiction, invented behavior, or wrong file path / symbol name.",
            "If the question is unanswerable from the codebase (or out-of-scope for a code copilot) and the expected answer indicates a refusal or absence, award full credit ONLY if the actual answer clearly refuses or states the feature does not exist; score 0 if it hallucinates an implementation or fulfills an out-of-scope request (e.g., writing a poem or recipe).",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge_llm,
        threshold=0.75,
    )

    completeness = GEval(
        name="Completeness",
        evaluation_steps=[
            "Extract the distinct key points, files, symbols, or sub-questions required by the expected answer and the user question.",
            "Check each key point one by one to see whether the actual answer covers it.",
            "For multi-part questions, verify that every part of the question is addressed (not just the first part).",
            "For unanswerable or out-of-scope items where the expected answer is a refusal, check that the refusal clearly states what is missing or why the request is outside RepoPilot's codebase scope.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge_llm,
        threshold=0.70,
    )

    style = GEval(
        name="Style",
        evaluation_steps=[
            "Check that the response is concise and direct, without conversational filler ('Great question!', 'Sure!') or unnecessary preamble.",
            "Check that codebase claims cite specific file paths (and line numbers/ranges when referencing code).",
            "Check that identifiers, file paths, and code snippets use markdown backticks or fenced code blocks.",
            "Check that there is no vague hedging when concrete evidence is presented.",
            "If the prompt is out-of-scope or unanswerable, check that the response uses a clear, professional refusal tone; score 0 if it fulfills an out-of-scope non-coding request.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=judge_llm,
        threshold=0.75,
    )

    return {
        "correctness": correctness,
        "completeness": completeness,
        "style": style,
    }


def build_geval_metrics_v2(judge_llm: Any) -> dict[str, Any]:
    """Create Rubric v2 G-Eval metrics: Correctness (anchored), Completeness
    (COVERED/MISSING), Style Tone (narrow), and Refusal (binary).
    """
    if not DEEPEVAL_AVAILABLE:
        raise RuntimeError("deepeval is required to run judging. Install evals/requirements.txt.")

    correctness_v2 = GEval(
        name="Correctness_v2",
        evaluation_steps=[
            "Read the user question, the expected (reference) answer, and the actual answer.",
            "Check whether every factual claim that IS present in the actual answer is consistent with the expected answer.",
            "Do NOT deduct points if the actual answer omits some expected points; omissions are scored separately by Completeness.",
            "Reduce the score ONLY for explicit factual contradictions or wrong file paths, wrong symbol names, or wrong constants/values.",
            "If the actual answer names a different file, function, or value than the expected answer for the thing the question asks about, that is a wrong file/symbol/value (score ~0.5), even if the answer also contains correct extra details. 'Extra detail' means additional facts or files that do not replace or conflict with the expected item; additional valid files/functions stay neutral.",
            "Line numbers and additional details that do not replace or contradict the expected file, symbol or value are never penalised here.",
            "Scoring anchors: 1.0 = no contradictions and no wrong file/symbol/value (even if some expected points are omitted); ~0.5 = one wrong or substituted file/symbol/value for the asked item while core location/topic is right; 0.0 = core claim or primary location is wrong.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge_llm,
        threshold=0.75,
    )

    completeness_v2 = GEval(
        name="Completeness_v2",
        evaluation_steps=[
            "The expected output provides a numbered list of fixed key points.",
            "Do NOT invent, split, or merge key points. Evaluate ONLY the exact numbered key points provided.",
            "Mark a key point `[COVERED]` ONLY if the specific file path, symbol name, or value appears accurately in the actual answer; a wrong path, wrong symbol, or wrong value counts as `[MISSING]`.",
            "In your reason text, output the numbered list with `[COVERED]` or `[MISSING]` for every key point and state the `covered / total` fraction.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge_llm,
        threshold=0.70,
    )

    style_tone_v2 = GEval(
        name="Style_Tone_v2",
        evaluation_steps=[
            "Ignore answer length, markdown formatting, and citation syntax (those are checked deterministically).",
            "Check ONLY whether the tone is direct, confident, and engineering-focused without unnecessary hedging on observed facts or conversational/apologetic filler.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model=judge_llm,
        threshold=0.75,
    )

    refusal_binary_v2 = GEval(
        name="Refusal_Binary_v2",
        evaluation_steps=[
            "Check whether the user question asks about something absent from the codebase (unanswerable) or outside software/codebase assistance (out-of-scope).",
            "Score 1.0 ONLY if the actual answer clearly declines the out-of-scope request or states the feature/component is not in the codebase, AND does NOT perform the out-of-scope task or fabricate nonexistent implementation.",
            "Score 0.0 if the actual answer performs the out-of-scope task (e.g. writes a poem, recipe, or physics essay) or invents nonexistent codebase behavior.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=judge_llm,
        threshold=0.5,
    )

    return {
        "correctness": correctness_v2,
        "completeness": completeness_v2,
        "style_tone": style_tone_v2,
        "refusal": refusal_binary_v2,
    }


# ---------------------------------------------------------------------------
# Judging execution
# ---------------------------------------------------------------------------

def judge_records(
    success_rows: list[dict[str, Any]],
    cache_prefix: str = "pass1",
) -> tuple[list[dict[str, Any]], int]:
    """Run the 3 G-Eval metrics sequentially on successful rows.

    Returns (judged_rows, http_requests_made).
    """
    from run_generator_eval import RepoPilotJudgeLLM  # noqa: E402

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in backend/.env.")

    gen_model = success_rows[0].get("model_name", "gemini-3.5-flash-lite") if success_rows else "gemini-3.5-flash-lite"
    judge_llm = RepoPilotJudgeLLM(
        judge_provider="gemini",
        judge_model=FIXED_JUDGE_MODEL,
        generator_model=gen_model,
        judge_cache_path=JUDGE_CACHE_PATH,
        api_key=api_key,
        temperature=0.0,
        concurrency=1,
        cache_prefix=cache_prefix,
    )
    metrics = build_geval_metrics(judge_llm)

    judged_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(success_rows, 1):
        qid = row["question_id"]
        print(f"  [{idx:02d}/{len(success_rows):02d}] Judging {qid} ({classify_slice(row)})...")
        tc = LLMTestCase(
            input=row["question"],
            actual_output=row.get("final_answer", ""),
            expected_output=row.get("ideal_answer", ""),
        )
        m_out: dict[str, Any] = {}
        for m_name, metric in metrics.items():
            try:
                score = float(metric.measure(tc))
                m_out[m_name] = {"score": round(score, 4), "reason": getattr(metric, "reason", "")}
            except Exception as exc:
                m_out[m_name] = {"score": None, "reason": f"ERROR: {exc}"}
        judged_rows.append({**row, "metrics": m_out})

    return judged_rows, judge_llm.http_requests_count


def format_key_points_expected(key_points: list[dict[str, Any]] | list[str]) -> str:
    """Format fixed key_points as a numbered list for Completeness_v2."""
    lines = ["Fixed key points (mark ONLY these [COVERED] or [MISSING]):"]
    for i, kp in enumerate(key_points, 1):
        text = kp["text"] if isinstance(kp, dict) else str(kp)
        lines.append(f"{i}. {text}")
    return "\n".join(lines)


def _measure_v2_metric(metrics_v2: dict[str, Any], m_key: str, row: dict[str, Any]) -> dict[str, Any]:
    """Measure one Rubric v2 metric on a row; runs literal checks in code first."""
    if m_key == "completeness" and row.get("key_points"):
        kps = row["key_points"]
        # Support structured {text, kind, match} objects
        if isinstance(kps[0], dict):
            lit_res, sem_pts = check_literal_key_points(row.get("final_answer", ""), kps)
            lit_cov = sum(1 for x in lit_res if x["covered"])
            if not sem_pts:
                total = len(lit_res)
                score = round(lit_cov / total, 4) if total else 1.0
                return {
                    "score": score,
                    "code_covered": lit_cov,
                    "code_total": total,
                    "literal_results": lit_res,
                    "llm_calls": 0,
                    "reason": f"All {total} key points checked deterministically in code ({lit_cov}/{total}).",
                }
            # Only semantic points go to the judge
            tc = LLMTestCase(
                input=row["question"],
                actual_output=row.get("final_answer", ""),
                expected_output=format_key_points_expected(sem_pts),
            )
            try:
                judge_score = round(float(metrics_v2[m_key].measure(tc)), 4)
                reason = getattr(metrics_v2[m_key], "reason", "") or ""
                _, sem_cov, sem_tot = compute_completeness_from_reason(reason)
                if sem_tot == 0:
                    sem_tot = len(sem_pts)
                    sem_cov = round(judge_score * sem_tot)
                total = len(lit_res) + len(sem_pts)
                total_cov = lit_cov + sem_cov
                score = round(total_cov / total, 4) if total else 1.0
                return {
                    "score": score,
                    "judge_score": judge_score,
                    "code_covered": total_cov,
                    "code_total": total,
                    "literal_results": lit_res,
                    "completeness_mismatch": check_completeness_mismatch(judge_score, round(sem_cov / sem_tot, 4) if sem_tot else None),
                    "llm_calls": 1,
                    "reason": reason,
                }
            except Exception as exc:
                return {"score": None, "reason": f"ERROR: {exc}"}

    expected = format_key_points_expected(row["key_points"]) if (m_key == "completeness" and row.get("key_points")) else row.get("ideal_answer", "")
    tc = LLMTestCase(
        input=row["question"],
        actual_output=row.get("final_answer", ""),
        expected_output=expected,
    )
    try:
        judge_score = round(float(metrics_v2[m_key].measure(tc)), 4)
        reason = getattr(metrics_v2[m_key], "reason", "") or ""
        if m_key == "completeness":
            code_s, cov, tot = compute_completeness_from_reason(reason)
            primary_score = code_s if code_s is not None else judge_score
            return {
                "score": primary_score,
                "judge_score": judge_score,
                "code_covered": cov,
                "code_total": tot,
                "completeness_mismatch": check_completeness_mismatch(judge_score, code_s),
                "reason": reason,
            }
        return {"score": judge_score, "reason": reason}
    except Exception as exc:
        return {"score": None, "reason": f"ERROR: {exc}"}


def run_pilot2(
    neg_controls_path: str,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run Rubric v2 Extra Pilot (Pilot 2, 9 requests + buffer)."""
    all_neg = load_cached_run(neg_controls_path)
    by_id = {r["question_id"]: r for r in all_neg}

    sub_ids = ("neg-right-file-wrong-fn", "neg-right-fn-wrong-mech", "neg-half-two-part")
    est = len(sub_ids) * 2 + 1 + 1 + 1  # 6 + 1 tone + 1 refusal + 1 neg-wrong-path completeness = 9

    print()
    print("=" * 72)
    print("RepoPilot — Application-Level Evaluation  Rubric v2 PILOT 2")
    print("=" * 72)
    print(f"  Judge model      : {FIXED_JUDGE_MODEL} (fixed)")
    print(f"  Subtle negs (3)  : {', '.join(sub_ids)} (Correctness_v2 + Completeness_v2 = 6)")
    print("  Tone control (1) : neg-hedging-tone (Style_Tone_v2 = 1)")
    print("  Refusal neg (1)  : neg-fabricated-rp019 (Refusal_Binary_v2 = 1)")
    print("  Key-point re-run : neg-wrong-path (Completeness_v2 = 1)")
    print(f"  Est. API calls   : {est} Gemini requests (with 10% buffer: {math.ceil(est * 1.1)})")
    print("=" * 72)
    print()

    if dry_run:
        print(f"DRY RUN — 0 Gemini requests made. Estimated Pilot 2 requests: {est}")
        return None

    from run_generator_eval import RepoPilotJudgeLLM  # noqa: E402

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in backend/.env.")

    judge_llm = RepoPilotJudgeLLM(
        judge_provider="gemini",
        judge_model=FIXED_JUDGE_MODEL,
        generator_model="gemini-3.5-flash-lite",
        judge_cache_path=JUDGE_CACHE_PATH,
        api_key=api_key,
        temperature=0.0,
        concurrency=1,
        cache_prefix="v2_pilot2",
    )
    metrics_v2 = build_geval_metrics_v2(judge_llm)

    results: list[dict[str, Any]] = []
    for qid in sub_ids:
        row = by_id[qid]
        print(f"  Judging subtle negative {qid}...")
        results.append({
            "question_id": qid,
            "group": "subtle_negative",
            "metrics": {k: _measure_v2_metric(metrics_v2, k, row) for k in ("correctness", "completeness")},
        })

    print("  Judging tone control neg-hedging-tone...")
    results.append({
        "question_id": "neg-hedging-tone",
        "group": "tone_control",
        "metrics": {"style_tone": _measure_v2_metric(metrics_v2, "style_tone", by_id["neg-hedging-tone"])},
    })

    print("  Judging fabricated refusal neg-fabricated-rp019...")
    results.append({
        "question_id": "neg-fabricated-rp019",
        "group": "fabricated_refusal",
        "metrics": {"refusal": _measure_v2_metric(metrics_v2, "refusal", by_id["neg-fabricated-rp019"])},
    })

    print("  Re-judging neg-wrong-path Completeness_v2 with fixed key_points...")
    results.append({
        "question_id": "neg-wrong-path",
        "group": "key_points_rerun",
        "metrics": {"completeness": _measure_v2_metric(metrics_v2, "completeness", by_id["neg-wrong-path"])},
    })

    append_quota_ledger(
        script_name="run_application_judge.py --pilot2",
        model=FIXED_JUDGE_MODEL,
        requests_used=judge_llm.http_requests_count,
        output_file=neg_controls_path,
    )
    today_total = sum_today_quota()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RUNS_DIR, f"application_v2_pilot2_{ts}.json")
    payload = {
        "timestamp": ts,
        "judge_model": FIXED_JUDGE_MODEL,
        "http_requests_used": judge_llm.http_requests_count,
        "today_quota_total": today_total,
        "results": results,
    }
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  HTTP requests used: {judge_llm.http_requests_count} | Today's Pacific ledger total: {today_total} / 500")
    print(f"  Saved Pilot 2 JSON : {os.path.relpath(out_path, REPO_ROOT)}")
    return payload


def run_pilot_v2(
    success_rows: list[dict[str, Any]],
    neg_controls_path: str,
    dry_run: bool = False,
) -> dict[str, Any] | None:
    """Run Rubric v2 pilot on rp-026,001,028,030 + rp-022,021,032 + 4 negative controls."""
    by_id = {r["question_id"]: r for r in success_rows}
    ans_ids = ("rp-026", "rp-001", "rp-028", "rp-030")
    ref_ids = ("rp-022", "rp-021", "rp-032")
    ans_rows = [by_id[q] for q in ans_ids if q in by_id]
    ref_rows = [by_id[q] for q in ref_ids if q in by_id]
    neg_rows = load_cached_run(neg_controls_path) if os.path.exists(neg_controls_path) else []

    est = len(ans_rows) * 3 + len(ref_rows) * 1 + len(neg_rows) * 2
    print()
    print("=" * 72)
    print("RepoPilot — Application-Level Evaluation  Rubric v2 PILOT")
    print("=" * 72)
    print(f"  Judge model      : {FIXED_JUDGE_MODEL} (fixed)")
    print(f"  Answerable ({len(ans_rows)})   : {', '.join(ans_ids)}  (3 metrics each = {len(ans_rows)*3})")
    print(f"  Refusal ({len(ref_rows)})      : {', '.join(ref_ids)}  (1 metric each  = {len(ref_rows)})")
    print(f"  Neg controls ({len(neg_rows)}) : {', '.join(r['question_id'] for r in neg_rows)}  (2 metrics each = {len(neg_rows)*2})")
    print(f"  Est. API calls   : {est} Gemini requests (with 10% buffer: {math.ceil(est * 1.1)})")
    print("=" * 72)
    print()

    if dry_run:
        print(f"DRY RUN — 0 Gemini requests made. Estimated pilot requests: {est}")
        return None

    from run_generator_eval import RepoPilotJudgeLLM  # noqa: E402

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set in backend/.env.")

    judge_llm = RepoPilotJudgeLLM(
        judge_provider="gemini",
        judge_model=FIXED_JUDGE_MODEL,
        generator_model="gemini-3.5-flash-lite",
        judge_cache_path=JUDGE_CACHE_PATH,
        api_key=api_key,
        temperature=0.0,
        concurrency=1,
        cache_prefix="v2_pilot",
    )
    metrics_v2 = build_geval_metrics_v2(judge_llm)

    def _measure_one(m_key: str, row: dict[str, Any]) -> dict[str, Any]:
        tc = LLMTestCase(
            input=row["question"],
            actual_output=row.get("final_answer", ""),
            expected_output=row.get("ideal_answer", ""),
        )
        try:
            score = round(float(metrics_v2[m_key].measure(tc)), 4)
            reason = getattr(metrics_v2[m_key], "reason", "") or ""
            out: dict[str, Any] = {"score": score, "reason": reason}
            if m_key == "completeness":
                code_s, cov, tot = compute_completeness_from_reason(reason)
                out["code_covered"] = cov
                out["code_total"] = tot
                out["code_score"] = code_s
                out["completeness_mismatch"] = check_completeness_mismatch(score, code_s)
            return out
        except Exception as exc:
            return {"score": None, "reason": f"ERROR: {exc}"}

    results: list[dict[str, Any]] = []
    for row in ans_rows:
        print(f"  Judging answerable {row['question_id']}...")
        m_res = {k: _measure_one(k, row) for k in ("correctness", "completeness", "style_tone")}
        results.append({"question_id": row["question_id"], "group": "answerable", "metrics": m_res})

    for row in ref_rows:
        print(f"  Judging refusal {row['question_id']}...")
        m_res = {"refusal": _measure_one("refusal", row)}
        results.append({"question_id": row["question_id"], "group": "refusal", "metrics": m_res})

    for row in neg_rows:
        print(f"  Judging negative control {row['question_id']}...")
        m_res = {k: _measure_one(k, row) for k in ("correctness", "completeness")}
        results.append({"question_id": row["question_id"], "group": "negative_control", "synthetic": True, "metrics": m_res})

    append_quota_ledger(
        script_name="run_application_judge.py --pilot-v2",
        model=FIXED_JUDGE_MODEL,
        requests_used=judge_llm.http_requests_count,
        output_file=neg_controls_path,
    )
    today_total = sum_today_quota()
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(RUNS_DIR, f"application_v2_pilot_{ts}.json")
    payload = {
        "timestamp": ts,
        "judge_model": FIXED_JUDGE_MODEL,
        "http_requests_used": judge_llm.http_requests_count,
        "today_quota_total": today_total,
        "results": results,
    }
    os.makedirs(RUNS_DIR, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\n  HTTP requests used: {judge_llm.http_requests_count} | Today's ledger total: {today_total} / 500")
    print(f"  Saved pilot JSON : {os.path.relpath(out_path, REPO_ROOT)}")
    return payload


def _latest_application_run() -> str | None:
    candidates = sorted(
        p for p in glob.glob(os.path.join(RUNS_DIR, "application_*.jsonl"))
        if "smoke" not in os.path.basename(p).lower()
    )
    return candidates[-1] if candidates else None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RepoPilot Application-Level Evaluation — Phase 2: Quality Judge",
    )
    parser.add_argument("--input", default=None, help="Path to Phase 1 generation JSONL cache")
    parser.add_argument("--thresholds", default=THRESHOLDS_PATH, help="Path to thresholds JSON")
    parser.add_argument("--dry-run", action="store_true", help="Print estimated Gemini judge requests and exit")
    parser.add_argument("--style-check-only", action="store_true",
                        help="Run zero-quota deterministic Style v2 checks on cached answerable rows and exit")
    parser.add_argument("--pilot", action="store_true",
                        help="Run a 3-row pilot (1 answerable, 1 unanswerable, rp-021) and print scores + reasons")
    parser.add_argument("--pilot-v2", action="store_true",
                        help="Run Rubric v2 pilot (rp-026,001,028,030 + rp-022,021,032 + 4 negative controls = 23 reqs)")
    parser.add_argument("--pilot2", action="store_true",
                        help="Run Rubric v2 Extra Pilot (Pilot 2: subtle negatives, tone, fabricated refusal, key_points rerun)")
    parser.add_argument("--baseline-save", "--save-baseline", dest="baseline_save", action="store_true",
                        help="Save results as evals/runs/application_baseline.json")
    parser.add_argument("--estimate-noise", action="store_true",
                        help="Re-judge cached outputs N times to estimate judge noise")
    parser.add_argument("--noise-passes", type=int, default=2,
                        help="Number of judge passes for --estimate-noise (default: 2)")
    parser.add_argument("--gen-noise-runs", nargs="*", default=None,
                        help="Optional 2+ cached generation JSONLs to compute generation noise separately")
    args = parser.parse_args()

    input_path = args.input or _latest_application_run()
    if not input_path or not os.path.exists(input_path):
        print("ERROR: No application generation cache found. Pass --input <path>.", file=sys.stderr)
        sys.exit(1)

    thresholds = load_thresholds(args.thresholds)
    records = load_cached_run(input_path)
    success_rows, error_rows = split_success_and_errors(records)
    ops = compute_ops_summary(records)

    if args.style_check_only:
        style_summary = summarize_deterministic_style(success_rows)
        print("\n" + "=" * 72)
        print("Deterministic Style v2 Check (Answerable Slice — 0 LLM calls)")
        print("=" * 72)
        print(f"  Input cache  : {os.path.relpath(input_path, REPO_ROOT)}")
        print(f"  Answerable n : {style_summary['n_answerable']}")
        print(f"  Word budget  : <= {style_summary['word_budget_heuristic']} PROSE words (heuristic)")
        print("  Pass rates   :")
        for k, rate in style_summary["pass_rates"].items():
            cnt = style_summary["pass_counts"][k]
            print(f"    - {k:<26}: {cnt}/{style_summary['n_answerable']} ({rate:.1%})")
        print(f"  Prose stats  : {style_summary['prose_word_count_stats']}")
        print("\n  Per-row word breakdown (prose / code / total):")
        for row in style_summary["per_row"]:
            flag = "FAIL" if not row["within_word_budget"] else "ok  "
            print(
                f"    {row['question_id']}: {flag}  prose={row['prose_words']:>3}  "
                f"code={row['code_words']:>3}  total={row['total_words']:>3}"
            )
        print(f"\n  Failures ({len(style_summary['failures'])}):")
        for f_row in style_summary["failures"]:
            print(
                f"    - {f_row['question_id']}: failed={f_row['failed_checks']} "
                f"(prose={f_row['prose_words']}, code={f_row['code_words']}, total={f_row['total_words']})"
            )
        return

    if args.pilot2:
        neg_path = os.path.join(REPO_ROOT, "evals", "data", "application_negative_controls.jsonl")
        run_pilot2(neg_path, dry_run=args.dry_run)
        return

    if args.pilot_v2:
        neg_path = os.path.join(REPO_ROOT, "evals", "data", "application_negative_controls.jsonl")
        run_pilot_v2(success_rows, neg_path, dry_run=args.dry_run)
        return

    if args.pilot:
        success_rows = select_pilot_rows(success_rows)

    passes = args.noise_passes if args.estimate_noise else 1
    est_requests = estimate_judge_requests(len(success_rows), passes=passes, metrics_per_row=3)

    print()
    print("=" * 72)
    print("RepoPilot — Application-Level Evaluation  Phase 2: Quality Judge")
    print("=" * 72)
    print(f"  Input cache      : {os.path.relpath(input_path, REPO_ROOT)}")
    print(f"  Judge model      : {FIXED_JUDGE_MODEL} (fixed)")
    print(f"  Total rows       : {ops['total_questions']} (to judge: {len(success_rows)}, error rows: {len(error_rows)})")
    print(f"  Passes           : {passes} (3 G-Eval metrics/row)")
    print(f"  Est. API calls   : {est_requests} Gemini requests (free tier: 500/day)")
    print(f"  {SELF_JUDGING_NOTE}")
    print("=" * 72)
    print()

    if args.dry_run:
        print(f"DRY RUN — 0 Gemini requests made. Estimated judge requests: {est_requests}")
        return

    total_http = 0
    if args.estimate_noise:
        pass_results: list[list[dict[str, Any]]] = []
        for p in range(1, passes + 1):
            print(f"--- Judge Noise Pass {p}/{passes} ---")
            j_rows, reqs = judge_records(success_rows, cache_prefix=f"noise_p{p}")
            total_http += reqs
            pass_results.append(j_rows)
        judged_rows = pass_results[0]
        judge_noise = compute_judge_noise(pass_results, thresholds.get("noise_multiplier", 2.0))
    else:
        judged_rows, total_http = judge_records(success_rows, cache_prefix="pass1")
        judge_noise = None

    gen_noise = None
    if args.gen_noise_runs and len(args.gen_noise_runs) >= 2:
        gen_pass_results: list[list[dict[str, Any]]] = []
        for idx, g_path in enumerate(args.gen_noise_runs, 1):
            g_succ, _ = split_success_and_errors(load_cached_run(g_path))
            j_rows, reqs = judge_records(g_succ, cache_prefix=f"gen_noise_r{idx}")
            total_http += reqs
            gen_pass_results.append(j_rows)
        gen_noise = compute_generation_noise(gen_pass_results)

    append_quota_ledger(
        script_name="run_application_judge.py",
        model=FIXED_JUDGE_MODEL,
        requests_used=total_http,
        output_file=input_path,
    )

    agg = aggregate_judge_results(judged_rows, error_rows, thresholds)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_sha = success_rows[0].get("dataset_sha256", "") if success_rows else ""
    gen_model = success_rows[0].get("model_name", "gemini-3.5-flash-lite") if success_rows else "unknown"

    report = {
        "timestamp": ts,
        "input_cache": os.path.relpath(input_path, REPO_ROOT),
        "dataset_sha256": dataset_sha,
        "generator_model": gen_model,
        "judge_model": FIXED_JUDGE_MODEL,
        "provisional": judge_noise is None,
        "noise_pending": judge_noise is None,
        "self_judging_note": SELF_JUDGING_NOTE,
        "http_requests_used": total_http,
        "operations": ops,
        "quality": agg,
        "judge_noise": judge_noise,
        "generation_noise": gen_noise,
        "per_question": judged_rows,
    }

    today_total = sum_today_quota()
    print(f"\n  Today's quota ledger total: {today_total} / 500 requests (this run HTTP calls: {total_http})")

    if args.pilot:
        print("\n" + "=" * 72)
        print("PILOT RESULTS (3 rows — scores + judge reasons)")
        print("=" * 72)
        for r in judged_rows:
            print(f"\n[{r['question_id']}] slice={classify_slice(r)}")
            print(f"  Question: {r['question']}")
            print(f"  Answer  : {(r.get('final_answer') or '')[:180].replace(chr(10), ' ')}")
            for m_name in ("correctness", "completeness", "style"):
                m_val = r["metrics"][m_name]
                print(f"  - {m_name:<12}: score={m_val['score']} | reason={m_val['reason']}")
        return

    failed_judge_calls = [
        (r["question_id"], m_name, r["metrics"][m_name].get("reason", ""))
        for r in judged_rows
        for m_name in ("correctness", "completeness", "style")
        if r.get("metrics", {}).get(m_name, {}).get("score") is None
    ]
    if failed_judge_calls:
        print(f"\n⚠️  Failed judge calls ({len(failed_judge_calls)}) — NOT retried automatically:")
        for qid, m_name, reason in failed_judge_calls:
            print(f"    - {qid} [{m_name}]: {reason}")

    os.makedirs(RUNS_DIR, exist_ok=True)
    out_path = os.path.join(RUNS_DIR, f"application_judge_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.baseline_save:
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"✓ Saved baseline to {os.path.relpath(BASELINE_PATH, REPO_ROOT)}")

    if judge_noise is not None or gen_noise is not None:
        noise_payload = {
            "timestamp": ts,
            "judge_model": FIXED_JUDGE_MODEL,
            "judge_noise": judge_noise,
            "generation_noise": gen_noise,
        }
        with open(NOISE_PATH, "w", encoding="utf-8") as f:
            json.dump(noise_payload, f, indent=2)
        print(f"✓ Saved noise report to {os.path.relpath(NOISE_PATH, REPO_ROOT)}")

    print(f"\nSaved full judge report to: {os.path.relpath(out_path, REPO_ROOT)}")
    print(f"Overall: Correctness={agg['overall']['correctness']['mean']}  "
          f"Completeness={agg['overall']['completeness']['mean']}  "
          f"Style={agg['overall']['style']['mean']}  "
          f"RefusalPass={agg['refusal_pass_rate']}")
    if agg["failing_items"]:
        print(f"Failing items ({len(agg['failing_items'])}): "
              + ", ".join(f"{x['question_id']}({x['slice']})" for x in agg["failing_items"]))


if __name__ == "__main__":
    main()
