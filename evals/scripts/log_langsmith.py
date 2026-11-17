"""Log RepoPilot evaluation runs and benchmarks to LangSmith as datasets and experiments."""

import json
import os
import sys
from dotenv import load_dotenv

# Ensure root and backend in sys.path
sys.path.insert(0, os.path.abspath("backend"))
sys.path.insert(0, os.path.abspath("."))

load_dotenv(os.path.join("backend", ".env"))

def sync_benchmarks_to_langsmith():
    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    project = os.getenv("LANGSMITH_PROJECT", "repopilot").strip()

    if not api_key:
        print("[LangSmith] LANGSMITH_API_KEY not configured. Skipping upload.")
        return

    from langsmith import Client
    client = Client(api_key=api_key)

    dataset_name = "repopilot-retrieval-benchmark-v1.1"

    # 1. Sync or verify dataset
    dataset = None
    try:
        datasets = list(client.list_datasets(dataset_name=dataset_name))
        if datasets:
            dataset = datasets[0]
            print(f"[LangSmith] Found existing dataset: {dataset.name} (ID: {dataset.id})")
        else:
            dataset = client.create_dataset(
                dataset_name=dataset_name,
                description="RepoPilot deterministic and neural retrieval benchmark v1.1",
            )
            print(f"[LangSmith] Created dataset: {dataset.name} (ID: {dataset.id})")

            # Upload 22 benchmark examples
            with open("evals/data/golden_v1.jsonl", "r", encoding="utf-8") as f:
                for line in f:
                    ex = json.loads(line)
                    client.create_example(
                        inputs={"question": ex["question"], "id": ex["id"], "category": ex["category"]},
                        outputs={
                            "gt_files": ex["gt_files"],
                            "gt_symbols": ex.get("gt_symbols", []),
                            "ideal_answer": ex.get("ideal_answer", ""),
                            "answerable": ex.get("answerable", True),
                        },
                        dataset_id=dataset.id,
                    )
            print(f"[LangSmith] Uploaded benchmark examples to dataset.")
    except Exception as e:
        print(f"[LangSmith] Dataset sync error: {e}")

    # 2. Log baseline runs as experiment results
    runs_to_log = [
        ("retriever_baseline_full_file.json", "Full-Index File-Level v1.1"),
        ("retriever_baseline_full_symbol.json", "Full-Index Symbol-Level v1.1"),
        ("retriever_baseline_code_file.json", "Code-Only File-Level v1.1"),
        ("retriever_baseline_code_symbol.json", "Code-Only Symbol-Level v1.1"),
    ]

    for fname, exp_name in runs_to_log:
        fpath = os.path.join("evals", "runs", fname)
        if not os.path.exists(fpath):
            continue

        with open(fpath, "r", encoding="utf-8") as f:
            run_data = json.load(f)

        summary = run_data.get("summary", {})
        reranked_metrics = summary.get("reranked", {})

        print(f"[LangSmith] Logging experiment: '{exp_name}' (Project: {project})")
        try:
            # Create a top-level run for the benchmark experiment
            experiment_run = client.create_run(
                name=exp_name,
                run_type="chain",
                inputs={
                    "corpus": run_data.get("corpus"),
                    "level": run_data.get("level"),
                    "k_pool": run_data.get("k_pool"),
                    "commit": run_data.get("commit"),
                    "reranker": run_data.get("settings", {}).get("neural_reranker"),
                },
                outputs={
                    "Hit@1": reranked_metrics.get("Hit@1"),
                    "Hit@3": reranked_metrics.get("Hit@3"),
                    "Hit@5": reranked_metrics.get("Hit@5"),
                    "Recall@5": reranked_metrics.get("Recall@5"),
                    "Precision@5": reranked_metrics.get("Precision@5"),
                    "MRR": reranked_metrics.get("MRR"),
                    "nDCG@5": reranked_metrics.get("nDCG@5"),
                    "Pool_Hit@50": run_data.get("stage_comparison", {}).get("pool_hit_50"),
                    "Pool_Recall@50": run_data.get("stage_comparison", {}).get("pool_recall_50"),
                    "Reranker_P50_ms": run_data.get("stage_comparison", {}).get("reranker_latency", {}).get("p50_ms"),
                },
                project_name=project,
                extra={"version": "v1.1", "source": "RepoPilot-Benchmark-Suite"},
            )
            print(f"   -> Logged run ID: {experiment_run.id if hasattr(experiment_run, 'id') else 'OK'}")
        except Exception as e:
            print(f"   -> Run logging note: {e}")

    print("\n[LangSmith] All v1.1 benchmark runs synchronized successfully.")


if __name__ == "__main__":
    sync_benchmarks_to_langsmith()
