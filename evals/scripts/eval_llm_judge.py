"""LLM-as-a-Judge retrieval evaluation following DeepEval methodology.

Evaluates second-layer retrieval quality on a benchmark sample:
1. Contextual Recall: Does retrieved context contain what is needed to produce the ideal answer?
2. Contextual Precision: Are relevant chunks ranked higher than irrelevant ones?
3. Contextual Relevancy: How much of the retrieved context is relevant (signal vs noise)?

Uses an independent judge model distinct from the generator.
"""

import argparse
import json
import os
import re
import sys
import time
from typing import Any
import httpx
from dotenv import load_dotenv

# Ensure backend and root are in sys.path
sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

load_dotenv(os.path.join("backend", ".env"))

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

DEFAULT_JUDGE_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-flash-lite-latest",
    "gemma-4-26b-a4b-it",
]

def call_judge_llm(prompt: str, preferred_model: str = "gemini-3.1-flash-lite") -> tuple[str, str]:
    """Call Google Generative Language API with the judge model."""
    if not API_KEY:
        raise RuntimeError("GEMINI_API_KEY not configured in backend/.env")

    models_to_try = [preferred_model] + [m for m in DEFAULT_JUDGE_MODELS if m != preferred_model]

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": 1024,
            "responseMimeType": "application/json"
        }
    }

    last_err = ""
    with httpx.Client(timeout=30.0) as client:
        for model in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={API_KEY}"
            try:
                res = client.post(url, json=payload)
                if res.status_code == 200:
                    data = res.json()
                    cands = data.get("candidates", [])
                    if cands:
                        raw_text = cands[0]["content"]["parts"][0]["text"].strip()
                        return raw_text, model
                last_err = f"HTTP {res.status_code}: {res.text[:120]}"
            except Exception as e:
                last_err = str(e)
                continue

    raise RuntimeError(f"All judge models failed. Last error: {last_err}")


def evaluate_contextual_recall(question: str, ideal_answer: str, context: list[str], model: str) -> dict[str, Any]:
    """Contextual Recall: Statements in ideal answer supported by context / total statements."""
    context_str = "\n\n---\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context))
    prompt = f"""You are an expert retrieval judge evaluating Contextual Recall.
Task:
1. Break down the Ideal Answer into clear, distinct factual statements.
2. For each statement, determine whether it can be attributed to or directly supported by the Retrieved Context.
3. Output a score from 0.0 to 1.0 (attributed statements / total statements).

Question: {question}

Ideal Answer:
{ideal_answer}

Retrieved Context:
{context_str}

Respond STRICTLY in JSON format with this exact structure:
{{
  "statements": [
    {{"statement": "...", "attributed": true, "reason": "..."}}
  ],
  "recall_score": 0.85,
  "explanation": "..."
}}
"""
    raw_text, used_model = call_judge_llm(prompt, preferred_model=model)
    try:
        # Strip code fences if present
        clean = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
        clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE)
        parsed = json.loads(clean)
        return {"score": float(parsed.get("recall_score", 0.0)), "details": parsed, "model": used_model}
    except Exception as e:
        return {"score": 0.0, "error": f"JSON parse error: {e}", "raw": raw_text, "model": used_model}


def evaluate_contextual_precision(question: str, context: list[str], model: str) -> dict[str, Any]:
    """Contextual Precision: Are relevant chunks ranked higher than irrelevant ones?"""
    chunks_str = "\n\n".join(f"[Node {i+1}]:\n{c}" for i, c in enumerate(context))
    prompt = f"""You are an expert retrieval judge evaluating Contextual Precision.
Task:
1. For each retrieved node in order from rank 1 to {len(context)}, determine if it contains relevant information useful for answering the Question.
2. Mark relevant: true or false.

Question: {question}

Retrieved Nodes:
{chunks_str}

Respond STRICTLY in JSON format:
{{
  "verdicts": [
    {{"rank": 1, "relevant": true, "reason": "..."}},
    {{"rank": 2, "relevant": false, "reason": "..."}}
  ],
  "explanation": "..."
}}
"""
    raw_text, used_model = call_judge_llm(prompt, preferred_model=model)
    try:
        clean = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
        clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE)
        parsed = json.loads(clean)
        verdicts = parsed.get("verdicts", [])

        # DeepEval Contextual Precision formula:
        # Weighted sum of Precision@k for all relevant nodes divided by total relevant nodes
        relevant_count = 0
        prec_sum = 0.0
        for i, v in enumerate(verdicts, start=1):
            if v.get("relevant", False):
                relevant_count += 1
                prec_at_k = relevant_count / i
                prec_sum += prec_at_k

        score = (prec_sum / relevant_count) if relevant_count > 0 else 0.0
        return {"score": round(score, 4), "relevant_nodes": relevant_count, "total_nodes": len(verdicts), "details": parsed, "model": used_model}
    except Exception as e:
        return {"score": 0.0, "error": f"JSON parse error: {e}", "raw": raw_text, "model": used_model}


def evaluate_contextual_relevancy(question: str, context: list[str], model: str) -> dict[str, Any]:
    """Contextual Relevancy: Ratio of relevant content in context to total context content (noise reduction)."""
    context_str = "\n\n---\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(context))
    prompt = f"""You are an expert retrieval judge evaluating Contextual Relevancy.
Task:
Analyze the retrieved context chunks and evaluate how much of the information is actually relevant to the question versus irrelevant boilerplate/noise.
Assign a score from 0.0 (entirely irrelevant noise) to 1.0 (concise, highly relevant snippets).

Question: {question}

Retrieved Context:
{context_str}

Respond STRICTLY in JSON format:
{{
  "relevancy_score": 0.75,
  "relevant_points": ["..."],
  "irrelevant_points": ["..."],
  "explanation": "..."
}}
"""
    raw_text, used_model = call_judge_llm(prompt, preferred_model=model)
    try:
        clean = re.sub(r"^```json\s*", "", raw_text.strip(), flags=re.MULTILINE)
        clean = re.sub(r"```$", "", clean.strip(), flags=re.MULTILINE)
        parsed = json.loads(clean)
        score = float(parsed.get("relevancy_score", 0.0))
        return {"score": round(score, 4), "details": parsed, "model": used_model}
    except Exception as e:
        return {"score": 0.0, "error": f"JSON parse error: {e}", "raw": raw_text, "model": used_model}


def run_llm_judge_sample(
    sample_ids: list[str] | None = None,
    corpus: str = "code",
    judge_model: str = "gemma-4-26b-a4b-it",
    output_path: str = "evals/runs/llm_judge_evaluation.json"
):
    from backend.app.search import retrieve_all_stages

    # Repository IDs: 8 = full-index, 9 = code-only
    repo_id = 9 if corpus == "code" else 8

    # Default sample covers each question category
    if not sample_ids:
        sample_ids = ["rp-001", "rp-004", "rp-006", "rp-008", "rp-011", "rp-016"]

    with open("evals/data/golden_v1.jsonl", "r", encoding="utf-8") as f:
        all_queries = [json.loads(line) for line in f if json.loads(line).get("id") in sample_ids]

    print("\n========================================================")
    print(f"RepoPilot LLM-Judge Evaluation (DeepEval Methodology)")
    print(f"Corpus: {corpus.upper()} (Repo ID: {repo_id}) | Judge Model: {judge_model}")
    print(f"Sample size: {len(all_queries)} queries")
    print("========================================================\n")

    results = []
    agg_recall = []
    agg_precision = []
    agg_relevancy = []

    for idx, ex in enumerate(all_queries, start=1):
        qid = ex["id"]
        cat = ex["category"]
        q = ex["question"]
        ideal = ex["ideal_answer"]

        print(f"[{idx}/{len(all_queries)}] Evaluating {qid} ({cat}): {q[:60]}...")

        # Retrieve Top 5 reranked chunks
        stages = retrieve_all_stages(query=q, k_pool=50, repository_id=repo_id)
        reranked_chunks = stages.get("reranked", [])[:5]
        context_texts = [
            f"File: {c.get('file_path')} (lines {c.get('start_line')}-{c.get('end_line')})\n{c.get('content', '')}"
            for c in reranked_chunks
        ]

        t0 = time.perf_counter()
        recall_res = evaluate_contextual_recall(q, ideal, context_texts, judge_model)
        prec_res = evaluate_contextual_precision(q, context_texts, judge_model)
        rel_res = evaluate_contextual_relevancy(q, context_texts, judge_model)
        elapsed = round(time.perf_counter() - t0, 2)

        rec_s = recall_res.get("score", 0.0)
        prec_s = prec_res.get("score", 0.0)
        rel_s = rel_res.get("score", 0.0)

        agg_recall.append(rec_s)
        agg_precision.append(prec_s)
        agg_relevancy.append(rel_s)

        print(f"   -> Contextual Recall:    {rec_s:.3f}")
        print(f"   -> Contextual Precision: {prec_s:.3f}")
        print(f"   -> Contextual Relevancy: {rel_s:.3f} ({elapsed}s)")

        results.append({
            "id": qid,
            "category": cat,
            "question": q,
            "ideal_answer": ideal,
            "top_retrieved_files": [f"{c.get('file_path')}:L{c.get('start_line')}-L{c.get('end_line')}" for c in reranked_chunks],
            "contextual_recall": recall_res,
            "contextual_precision": prec_res,
            "contextual_relevancy": rel_res,
            "evaluation_time_seconds": elapsed,
        })

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "corpus": corpus,
        "judge_model": judge_model,
        "sample_count": len(all_queries),
        "mean_contextual_recall": round(sum(agg_recall) / len(agg_recall), 4) if agg_recall else 0.0,
        "mean_contextual_precision": round(sum(agg_precision) / len(agg_precision), 4) if agg_precision else 0.0,
        "mean_contextual_relevancy": round(sum(agg_relevancy) / len(agg_relevancy), 4) if agg_relevancy else 0.0,
        "per_query_results": results
    }

    print("\n" + "=" * 60)
    print("LLM-JUDGE BENCHMARK SUMMARY (SAMPLE MACRO-AVERAGE)")
    print("=" * 60)
    print(f"Contextual Recall:    {summary['mean_contextual_recall']:.3f}")
    print(f"Contextual Precision: {summary['mean_contextual_precision']:.3f}")
    print(f"Contextual Relevancy: {summary['mean_contextual_relevancy']:.3f}")
    print("=" * 60)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved LLM-judge results to {output_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM-Judge retrieval evaluation")
    parser.add_argument("--corpus", choices=["full", "code"], default="code", help="Corpus to evaluate")
    parser.add_argument("--model", default="gemini-3.1-flash-lite", help="Judge model name")
    parser.add_argument("--out", default="evals/runs/llm_judge_evaluation.json", help="Output JSON path")
    args = parser.parse_args()

    run_llm_judge_sample(corpus=args.corpus, judge_model=args.model, output_path=args.out)
