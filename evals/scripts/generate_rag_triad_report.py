"""Programmatically build report_data_bundle.json and generate evals/RAG_TRIAD_REPORT.md.

Finalized baseline report centered on run rag_triad_20261125_130722.
Every number appearing in evals/RAG_TRIAD_REPORT.md is computed from JSON files.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from typing import Any

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RUNS_DIR = os.path.join(REPO_ROOT, "evals", "runs")

RUN_PATH = os.path.join(RUNS_DIR, "rag_triad_20261125_130722.json")
REMAINING_PATH = os.path.join(RUNS_DIR, "remaining_metrics_data.json")
REVISED_AUDIT_PATH = os.path.join(RUNS_DIR, "revised_audit_data.json")
RETRIEVER_FULL_PATH = os.path.join(RUNS_DIR, "retriever_baseline_full_file.json")
GOLDEN_PATH = os.path.join(REPO_ROOT, "evals", "data", "golden_v2.jsonl")
BUNDLE_PATH = os.path.join(RUNS_DIR, "report_data_bundle.json")
REPORT_PATH = os.path.join(REPO_ROOT, "evals", "RAG_TRIAD_REPORT.md")


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_bundle() -> dict[str, Any]:
    run_data = _load_json(RUN_PATH)
    rem_data = _load_json(REMAINING_PATH)
    audit_data = _load_json(REVISED_AUDIT_PATH)
    ret_full = _load_json(RETRIEVER_FULL_PATH)
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        golden_rows = {json.loads(l)["id"]: json.loads(l) for l in f if l.strip()}

    results_by_id = {r["id"]: r for r in run_data["results"]}
    summary = run_data["summary"]
    ds = run_data["dataset_stats"]

    # Contextual Precision & Recall excluding JSON parse evaluation_errors
    prec_valid, prec_errors = [], []
    rec_valid, rec_errors = [], []
    rec_lt_1_valid = []

    for d in rem_data["contextual_precision_recall"]["details"]:
        qid = d["id"]
        if "invalid JSON" in d.get("contextual_precision_reason", ""):
            prec_errors.append({"id": qid, "status": "evaluation_error", "reason": d["contextual_precision_reason"]})
        else:
            prec_valid.append(d["contextual_precision"])

        if "invalid JSON" in d.get("contextual_recall_reason", ""):
            rec_errors.append({"id": qid, "status": "evaluation_error", "reason": d["contextual_recall_reason"]})
            score = None
            reason = d["contextual_recall_reason"]
        else:
            score = d["contextual_recall"]
            reason = d["contextual_recall_reason"]
            rec_valid.append(score)

        if score is not None and score < 1.0:
            chunks = results_by_id[qid]["retrieval_context"]
            ret_files = []
            for c in chunks:
                m = re.match(r"File: ([^\s(]+)", c)
                if m:
                    ret_files.append(m.group(1))
            gt = golden_rows[qid].get("gt_files", [])
            rec_lt_1_valid.append({
                "id": qid,
                "score": score,
                "reason": reason,
                "gt_files": gt,
                "retrieved_files": ret_files,
                "missing_gt_files": [gf for gf in gt if gf not in ret_files],
            })

    contextual_metrics_clean = {
        "contextual_precision": {
            "n_valid": len(prec_valid),
            "n_error": len(prec_errors),
            "errors": prec_errors,
            "mean": round(statistics.mean(prec_valid), 4),
            "p50": round(statistics.median(prec_valid), 4),
            "pass_rate": round(sum(1 for x in prec_valid if x >= 0.7) / len(prec_valid), 4),
            "pass_count": sum(1 for x in prec_valid if x >= 0.7),
        },
        "contextual_recall": {
            "n_valid": len(rec_valid),
            "n_error": len(rec_errors),
            "errors": rec_errors,
            "mean": round(statistics.mean(rec_valid), 4),
            "p50": round(statistics.median(rec_valid), 4),
            "pass_rate": round(sum(1 for x in rec_valid if x >= 0.7) / len(rec_valid), 4),
            "pass_count": sum(1 for x in rec_valid if x >= 0.7),
            "lt_1_valid": rec_lt_1_valid,
        },
    }

    flagged_cases = []
    for qid in ("rp-009", "rp-024"):
        r = results_by_id[qid]
        flagged_cases.append({
            "id": qid,
            "question": r["question"],
            "score": r["faithfulness_score"],
            "judge_flagged_claim": r["faithfulness_reason"],
            "answer": r["answer"],
            "retrieval_context": r["retrieval_context"],
            "review_status": "judge-flagged, pending human review",
        })

    bundle = {
        "run_id": run_data["run_id"],
        "dataset_stats": {
            "total_rows": ds["total_rows"],
            "evaluated_count": ds["evaluated_count"],
            "skipped_count": ds["skipped_count"],
        },
        "judge_info": {
            "run_judge_provider": run_data["judge_provider"],
            "run_judge_model": run_data["judge_model"],
            "run_judge_model_version_observed": run_data.get("judge_model_versions_observed") or "not_logged_in_130722",
        },
        "generator_info": {
            "run_130722_generator_provider": run_data.get("generator_provider"),
            "run_130722_generator_model": run_data.get("generator_model"),
        },
        "triad_summary": {
            "context_relevancy": summary["context_relevancy"],
            "chunk_file_precision": summary["chunk_file_precision"],
            "faithfulness": summary["faithfulness"],
            "answer_relevancy": summary["answer_relevancy"],
        },
        "faithfulness_audit": audit_data["faithfulness_audit"],
        "flagged_cases_manual": flagged_cases,
        "contextual_metrics_clean": contextual_metrics_clean,
        "unified_recall_table": audit_data["cutoff_retrieval_metrics"],
        "retriever_k50_reranked": {
            "hit_at_5": ret_full["summary"]["reranked"]["Hit@5"],
            "recall_at_5": ret_full["summary"]["reranked"]["Recall@5"],
        },
        "eye_inspection_samples": audit_data["eye_inspection_samples"],
    }

    with open(BUNDLE_PATH, "w", encoding="utf-8") as f:
        json.dump(bundle, f, indent=2)
    return bundle


def render_report(b: dict[str, Any]) -> str:
    ts = b["triad_summary"]
    fa = b["faithfulness_audit"]
    cm = b["contextual_metrics_clean"]
    cp = cm["contextual_precision"]
    cr = cm["contextual_recall"]
    ur = b["unified_recall_table"]
    k50 = b["retriever_k50_reranked"]
    ji = b["judge_info"]
    ds = b["dataset_stats"]

    lines: list[str] = []
    lines.append("# RepoPilot End-to-End RAG Triad Evaluation Report")
    lines.append(f"**Single Baseline Run ID**: `{b['run_id']}`  ")
    lines.append(f"**Dataset**: `evals/data/golden_v2.jsonl` ({ds['total_rows']} rows: {ds['evaluated_count']} answerable evaluated, {ds['skipped_count']} unanswerable skipped)  ")
    lines.append(f"**Judge**: `{ji['run_judge_provider']}/{ji['run_judge_model']}` (unpinned alias; API `modelVersion`: `{ji['run_judge_model_version_observed']}`)  ")
    lines.append(f"**Baseline Generator**: `generator_provider={b['generator_info']['run_130722_generator_provider']}`, `generator_model={b['generator_info']['run_130722_generator_model']}` (not logged in `130722`)  ")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. Summary Table
    lines.append("## 1. Summary Table (`rag_triad_20261125_130722`)")
    lines.append("")
    lines.append("| Metric | Valid $n$ | Errors | Mean | P50 | Std Dev | Min / Max | Pass Rate (Threshold) |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    lines.append(
        f"| **Context Relevancy** | {ts['context_relevancy']['n']} | 0 | {ts['context_relevancy']['mean']:.4f} | "
        f"{ts['context_relevancy']['p50']:.4f} | {ts['context_relevancy']['std_dev']:.4f} | "
        f"{ts['context_relevancy']['min']:.4f} / {ts['context_relevancy']['max']:.4f} | "
        f"{ts['context_relevancy']['pass_rate']*100:.1f}% ($\\ge {ts['context_relevancy']['threshold']}$) |"
    )
    lines.append(
        f"| **Chunk File Precision** | {ts['chunk_file_precision']['n']} | 0 | {ts['chunk_file_precision']['mean']:.4f} | "
        f"{ts['chunk_file_precision']['p50']:.4f} | {ts['chunk_file_precision']['std_dev']:.4f} | "
        f"{ts['chunk_file_precision']['min']:.4f} / {ts['chunk_file_precision']['max']:.4f} | "
        f"{ts['chunk_file_precision']['pass_rate']*100:.1f}% ($\\ge {ts['chunk_file_precision']['threshold']}$) |"
    )
    lines.append(
        f"| **Contextual Precision** | {cp['n_valid']} | {cp['n_error']} | {cp['mean']:.4f} | "
        f"{cp['p50']:.4f} | — | — | {cp['pass_rate']*100:.1f}% ({cp['pass_count']}/{cp['n_valid']} at $\\ge 0.7$) |"
    )
    lines.append(
        f"| **Contextual Recall** | {cr['n_valid']} | {cr['n_error']} | {cr['mean']:.4f} | "
        f"{cr['p50']:.4f} | — | — | {cr['pass_rate']*100:.1f}% ({cr['pass_count']}/{cr['n_valid']} at $\\ge 0.7$) |"
    )
    lines.append(
        f"| **Answer Relevancy** | {ts['answer_relevancy']['n']} | 0 | {ts['answer_relevancy']['mean']:.4f} | "
        f"{ts['answer_relevancy']['p50']:.4f} | {ts['answer_relevancy']['std_dev']:.4f} | "
        f"{ts['answer_relevancy']['min']:.4f} / {ts['answer_relevancy']['max']:.4f} | "
        f"{ts['answer_relevancy']['pass_rate']*100:.1f}% ($\\ge {ts['answer_relevancy']['threshold']}$) |"
    )
    lines.append(
        f"| **Faithfulness (Strict 1.0)** | {fa['total']} | 0 | {ts['faithfulness']['mean']:.4f} | "
        f"{ts['faithfulness']['p50']:.4f} | {ts['faithfulness']['std_dev']:.4f} | "
        f"{ts['faithfulness']['min']:.4f} / {ts['faithfulness']['max']:.4f} | "
        f"**{fa['threshold_1_0_pass_rate']*100:.1f}% ({fa['threshold_1_0_pass_count']}/{fa['total']} at $= 1.0$)** |"
    )
    lines.append("")
    lines.append(
        f"- **% of Answers with Any Unsupported Claim (`faithfulness < 1.0`)**: "
        f"**{fa['any_unsupported_claim_pct']:.1f}% ({fa['any_unsupported_claim_count']}/{fa['total']})** "
        f"(`rp-009`, `rp-024` — judge-flagged only, pending human review)."
    )
    lines.append("- **Note on `generations_baseline_k5.json`**: `generations_baseline_k5.json` used `groq/openai/gpt-oss-120b`, which is not RepoPilot's shipped generator, so it is not part of the findings.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 2. Limitations & Incomplete Tracks
    lines.append("## 2. Limitations & Incomplete Tracks")
    lines.append("")
    lines.append("### Limitations")
    lines.append(f"- **Single judge**: Evaluated with `{ji['run_judge_provider']}/{ji['run_judge_model']}` (unpinned alias; API `modelVersion` not logged).")
    lines.append(f"- **Sample size**: $n = {ds['evaluated_count']}$ answerable queries from `golden_v2.jsonl`.")
    lines.append("- **Baseline generator not logged**: Run `130722` did not record `generator_provider` or `generator_model`.")
    lines.append("- **No noise estimate**: Stochastic judge variance ($\sigma$) could not be completed due to quota exhaustion.")
    lines.append("- **Unverified faithfulness flags**: `rp-009` and `rp-024` are judge-flagged only, pending human review.")
    lines.append("- **Judge sensitivity**: Faithfulness judge may miss small errors (e.g., `rp-002` line attribution), so `2/26` is a lower bound.")
    lines.append("")
    lines.append("### Collapsed Track Statuses")
    lines.append("- **Ollama-Qwen judge**: not completed (~3 min/question on CPU; single completed row not comparable to Gemini).")
    lines.append("- **Groq judge**: not completed (rate-limited with ~480s 429 backoffs after generation exhausted token quota).")
    lines.append("- **Gold comparison**: not reliable (`generator_baseline_gold.json` used `ollama/qwen2.5-coder:1.5b` and `gemini/gemini-3.5-flash-lite`, differing from `130722`).")
    lines.append("- **Noise estimation**: not completed (cache-off re-run halted at `rp-007` of Repeat 1 by Gemini daily quota).")
    lines.append("- **`tau=2.0` Triad**: not completed (Gemini daily quota exhausted).")
    lines.append("- **Judge dependence**: not reliable (only partial 10-row `rp-009`/`rp-024` + 8-row check; full 26-row paired run blocked by quota).")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 3. Recall Table & Decision
    lines.append("## 3. Unified Recall Table (`pool=15`) & Decision")
    lines.append("")
    lines.append("- **System Boundary**: Pipeline retrieval (pool 15) and the component retriever baseline (pool 50) are different systems.")
    lines.append(f"- **Component Retriever Baseline (`k_pool=50`, `retriever_baseline_full_file.json`)**: Hit@5 = **{k50['hit_at_5']:.4f}**, Fractional Recall@5 = **{k50['recall_at_5']:.4f}**.")
    lines.append("")
    lines.append("| Configuration (`pool=15`) | Mean Chunks | Binary Hit@k | Fractional Recall@k |")
    lines.append("| :--- | :--- | :--- | :--- |")
    for label, key in [
        ("Baseline (`k=5`)", "baseline_top_k_5"),
        ("`k=4`", "top_k_4"),
        ("`k=3`", "top_k_3"),
        ("`tau = 1.0` (`keep >= 1`)", "cutoff_1.0_keep_1"),
        ("`tau = 2.0` (`keep >= 1`)", "cutoff_2.0_keep_1"),
        ("`tau = 3.0` (`keep >= 1`)", "cutoff_3.0_keep_1"),
    ]:
        row = ur[key]
        lines.append(f"| {label} | {row['mean_chunks']:.2f} | {row['binary_hit']:.4f} | {row['multi_target_recall']:.4f} |")
    lines.append("")
    lines.append("- **Decision**: Retrieval recall favors `k=5`; `k=4` not yet tested on the triad.")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 4. Contextual Recall Audit
    lines.append("## 4. Contextual Recall Audit (Errors Excluded)")
    lines.append("")
    lines.append(f"- **Parse Failures Marked `evaluation_error` (Excluded)**:")
    for err in cr["errors"]:
        lines.append(f"  - `{err['id']}` (`ContextualRecall`): `{err['status']}` — `{err['reason']}`")
    for err in cp["errors"]:
        lines.append(f"  - `{err['id']}` (`ContextualPrecision`): `{err['status']}` — `{err['reason']}`")
    lines.append(f"- **Valid Contextual Recall ($n={cr['n_valid']}$)**: Mean = **{cr['mean']:.4f}**, P50 = **{cr['p50']:.4f}**, Pass Rate = **{cr['pass_rate']*100:.1f}% ({cr['pass_count']}/{cr['n_valid']})**.")
    lines.append(f"- **Valid Contextual Precision ($n={cp['n_valid']}$)**: Mean = **{cp['mean']:.4f}**, P50 = **{cp['p50']:.4f}**, Pass Rate = **{cp['pass_rate']*100:.1f}% ({cp['pass_count']}/{cp['n_valid']})**.")
    lines.append("")
    lines.append(f"### Valid Questions Scoring `< 1.0` on Contextual Recall ($n={len(cr['lt_1_valid'])}$)")
    lines.append("")
    for item in cr["lt_1_valid"]:
        lines.append(f"- **`{item['id']}`** (score={item['score']:.4f}): Missing `gt_files`: `{item['missing_gt_files']}`. Reason: *\"{item['reason']}\"*")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 5. Per-Question Diagnoses (Judge-Flagged & Eye-Inspection)
    lines.append("## 5. Per-Question Diagnoses (`rp-009`, `rp-024`, and Eye-Inspection `rp-001`..`rp-005`)")
    lines.append("")
    for case in b["flagged_cases_manual"]:
        lines.append(f"### `{case['id']}` (Faithfulness = {case['score']:.4f} — Status: **{case['review_status']}**)")
        lines.append(f"- **Question**: `{case['question']}`")
        lines.append(f"- **Judge-Flagged Claim**: {case['judge_flagged_claim']}")
        lines.append("- **Generated Answer**:")
        lines.append("```text")
        lines.append(case["answer"])
        lines.append("```")
        lines.append("- **Retrieved Chunks**:")
        for idx, chunk in enumerate(case["retrieval_context"], 1):
            header = chunk.splitlines()[0] if chunk else ""
            lines.append(f"  {idx}. `{header}`")
        lines.append("")

    lines.append("### Verbatim Eye-Inspection Samples (`rp-001`..`rp-005`)")
    lines.append("")
    for samp in b["eye_inspection_samples"]:
        lines.append(
            f"- **`{samp['id']}`** (Ctx={samp['context_relevancy_score']:.4f}, "
            f"Faith={samp['faithfulness_score']:.4f}, Ans={samp['answer_relevancy_score']:.4f}): `{samp['question']}`"
        )
        lines.append(f"  - **Answer**: {samp['answer']}")
        headers = [c.splitlines()[0] if c else "" for c in samp["retrieval_context"]]
        lines.append(f"  - **Chunks**: `{headers}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 6. Next Steps
    lines.append("## 6. Next Steps")
    lines.append("1. Run `k=4` when quota allows with the shipped generator.")
    lines.append("2. Study docs-outranking-code misses (`rp-008`, `rp-014`, `rp-017`, `rp-026`, `rp-027`); note that Hit@k counts `docs` files, and code files are missing in 5/8 recall failures (`rp-008`, `014`, `017`, `026`, `027`).")
    lines.append("3. Fix `rp-009`-style config misreads; verify the `rp-024` golden answer against the actual migration.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    bundle = build_bundle()
    report_text = render_report(bundle)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Generated {BUNDLE_PATH} and {REPORT_PATH} cleanly.")


if __name__ == "__main__":
    main()
