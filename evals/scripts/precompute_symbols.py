"""Precompute exact symbol line ranges at the frozen commit.

Uses Python's ast module to map every ground-truth function, class, and variable
to its start_line and end_line in the repository. Stores the result in
evals/data/symbol_ranges.json for symbol-level hit evaluations.
"""

import ast
import json
import os

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GOLDEN_V2_PATH = os.path.join(REPO_ROOT, "evals", "data", "golden_v2.jsonl")
GOLDEN_PATH = GOLDEN_V2_PATH if os.path.exists(GOLDEN_V2_PATH) else os.path.join(REPO_ROOT, "evals", "data", "golden_v1.jsonl")
OUTPUT_PATH = os.path.join(REPO_ROOT, "evals", "data", "symbol_ranges.json")


def precompute():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        examples = [json.loads(line) for line in f if line.strip()]

    symbol_ranges = {}

    for ex in examples:
        gt_files = ex.get("gt_files", [])
        gt_symbols = set(ex.get("gt_symbols", []))
        if not gt_symbols:
            continue

        for rel_file in gt_files:
            abs_file = os.path.join(REPO_ROOT, rel_file)
            if not os.path.exists(abs_file) or not abs_file.endswith(".py"):
                continue

            with open(abs_file, "r", encoding="utf-8", errors="ignore") as src:
                tree = ast.parse(src.read(), filename=abs_file)

            for node in ast.walk(tree):
                names = []
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.append(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names.append(target.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names.append(node.target.id)

                for name in names:
                    if name in gt_symbols:
                        start = node.lineno
                        end = getattr(node, "end_lineno", node.lineno)
                        key = f"{rel_file}::{name}"
                        symbol_ranges[key] = {
                            "file_path": rel_file,
                            "symbol": name,
                            "start_line": start,
                            "end_line": end,
                        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        json.dump(symbol_ranges, out, indent=2)

    print(f"Successfully saved {len(symbol_ranges)} symbol line ranges to {OUTPUT_PATH}")
    return symbol_ranges


if __name__ == "__main__":
    precompute()
