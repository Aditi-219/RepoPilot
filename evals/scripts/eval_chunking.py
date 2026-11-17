"""Structural evaluation of code chunking in RepoPilot (Step 4a).

Runs static assertions and statistical analyses across all stored chunks:
1. Token length vs embedder limit (WordPiece vs BPE/tiktoken truncate threshold 256).
2. Boundary integrity (AST chunking vs sliding-window split rate on Python functions/classes).
3. Size distribution (min, median, max, p95, and count of tiny chunks < 20 tokens).
4. Hygiene (empty chunks, duplicates, valid line numbers and ranges).
5. Coverage (ingested file chunk production and exclusion of binaries/noise).

Saves report to evals/runs/chunking_structural_report.json.
"""

import ast
import json
import os
import statistics
import sys
from collections import defaultdict
from typing import Any

# Ensure project paths are in sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))

from app.chunker import _chunk_by_tokens, _count_tokens
from app.database import SessionLocal
from app.github_service import (
    MAX_FILE_BYTES,
    SUPPORTED_EXTENSIONS,
    _get_file_extension,
)
from app.models import CodeChunk, Repository
from transformers import AutoTokenizer

REPORT_PATH = os.path.join(REPO_ROOT, "evals", "runs", "chunking_structural_report.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WORDPIECE_LIMIT = 256
TINY_CHUNK_THRESHOLD = 20

EXCLUDED_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".agents",
    "data", "brain", ".system_generated", "scripts", "evals"
}


def evaluate_chunking_structural() -> dict[str, Any]:
    print("\n" + "=" * 80)
    print("STEP 4a: CHUNKING STRUCTURAL EVALUATION")
    print("=" * 80 + "\n")

    db = SessionLocal()
    try:
        repo = db.query(Repository).filter(Repository.full_name == "repopilot").first()
        if not repo:
            raise ValueError("Repository 'repopilot' not found in database. Run index_local_snapshot.py first.")

        chunks = (
            db.query(CodeChunk)
            .filter(CodeChunk.repository_id == repo.id)
            .order_by(CodeChunk.file_path, CodeChunk.start_line)
            .all()
        )
        total_chunks = len(chunks)
        print(f"Loaded {total_chunks} chunks for repository '{repo.full_name}' (ID={repo.id}).")

        # ---------------------------------------------------------------------
        # 1. Token Length vs Embedder Limit
        # ---------------------------------------------------------------------
        print("\n[1/5] Evaluating Token Length vs Embedder Limit (WordPiece 256 limit)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        bpe_tokens_list = []
        wp_tokens_list = []
        truncated_count = 0
        truncated_chunks_sample = []

        for c in chunks:
            bpe_len = _count_tokens(c.content)
            wp_len = len(tokenizer.encode(c.content, add_special_tokens=True))
            bpe_tokens_list.append(bpe_len)
            wp_tokens_list.append(wp_len)

            if wp_len > WORDPIECE_LIMIT:
                truncated_count += 1
                if len(truncated_chunks_sample) < 5:
                    truncated_chunks_sample.append({
                        "file_path": c.file_path,
                        "lines": f"{c.start_line}-{c.end_line}",
                        "tiktoken_bpe": bpe_len,
                        "wordpiece_tokens": wp_len,
                        "truncated_tokens": wp_len - WORDPIECE_LIMIT,
                    })

        pct_truncated = (truncated_count / total_chunks * 100) if total_chunks else 0.0

        token_length_stats = {
            "total_chunks": total_chunks,
            "embedder_max_tokens": WORDPIECE_LIMIT,
            "chunks_exceeding_limit": truncated_count,
            "pct_exceeding_limit": round(pct_truncated, 2),
            "wordpiece_min": min(wp_tokens_list) if wp_tokens_list else 0,
            "wordpiece_median": int(statistics.median(wp_tokens_list)) if wp_tokens_list else 0,
            "wordpiece_mean": round(statistics.mean(wp_tokens_list), 1) if wp_tokens_list else 0,
            "wordpiece_max": max(wp_tokens_list) if wp_tokens_list else 0,
            "tiktoken_bpe_median": int(statistics.median(bpe_tokens_list)) if bpe_tokens_list else 0,
            "tiktoken_bpe_mean": round(statistics.mean(bpe_tokens_list), 1) if bpe_tokens_list else 0,
            "sample_truncated_chunks": truncated_chunks_sample,
        }
        print(f"  -> Chunks exceeding {WORDPIECE_LIMIT} WordPiece tokens: {truncated_count}/{total_chunks} ({pct_truncated:.2f}%)")
        print(f"  -> WordPiece Tokens: Min={token_length_stats['wordpiece_min']}, Median={token_length_stats['wordpiece_median']}, Max={token_length_stats['wordpiece_max']}")

        # ---------------------------------------------------------------------
        # 2. Boundary Integrity (AST vs Sliding Window)
        # ---------------------------------------------------------------------
        print("\n[2/5] Evaluating Boundary Integrity on Python definitions...")

        # Group stored chunks by file
        chunks_by_file = defaultdict(list)
        for c in chunks:
            chunks_by_file[c.file_path].append(c)

        python_files = [f for f in chunks_by_file if f.endswith(".py")]

        total_defs = 0
        ast_split_defs = 0
        sliding_split_defs = 0

        for rel_file in python_files:
            abs_file = os.path.join(REPO_ROOT, rel_file)
            if not os.path.exists(abs_file):
                continue

            with open(abs_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()

            try:
                tree = ast.parse(content, filename=rel_file)
            except SyntaxError:
                continue

            # Extract top-level or method definitions
            defs = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = node.lineno
                    end = getattr(node, "end_lineno", node.lineno)
                    defs.append((node.name, start, end))

            total_defs += len(defs)

            # Check stored AST chunks
            file_chunks = chunks_by_file[rel_file]
            for _name, start, end in defs:
                # A def is preserved intact if at least one chunk fully contains [start, end]
                is_contained_ast = any(
                    c.start_line <= start and c.end_line >= end
                    for c in file_chunks
                )
                if not is_contained_ast:
                    # Check if it was touched by multiple chunks (split) or exceeds chunk size
                    touching = [c for c in file_chunks if max(c.start_line, start) <= min(c.end_line, end)]
                    if len(touching) > 1:
                        ast_split_defs += 1

            # Simulate sliding window chunks on the exact same file
            lines = content.splitlines(keepends=True)
            sliding_chunks = _chunk_by_tokens(
                lines=lines,
                file_path=rel_file,
                language="python",
                chunk_size=200,
                overlap=40,
            )
            for _name, start, end in defs:
                is_contained_sliding = any(
                    c.start_line <= start and c.end_line >= end
                    for c in sliding_chunks
                )
                if not is_contained_sliding:
                    touching = [c for c in sliding_chunks if max(c.start_line, start) <= min(c.end_line, end)]
                    if len(touching) > 1:
                        sliding_split_defs += 1

        ast_split_rate = (ast_split_defs / total_defs * 100) if total_defs else 0.0
        sliding_split_rate = (sliding_split_defs / total_defs * 100) if total_defs else 0.0

        boundary_stats = {
            "python_files_inspected": len(python_files),
            "total_functions_classes": total_defs,
            "ast_chunks_split_count": ast_split_defs,
            "ast_chunks_split_rate_pct": round(ast_split_rate, 2),
            "sliding_window_split_count": sliding_split_defs,
            "sliding_window_split_rate_pct": round(sliding_split_rate, 2),
        }
        print(f"  -> Total Python Functions & Classes: {total_defs}")
        print(f"  -> AST-aware Splitting: {ast_split_defs} split ({ast_split_rate:.2f}%)")
        print(f"  -> Sliding Window Splitting: {sliding_split_defs} split ({sliding_split_rate:.2f}%)")

        # ---------------------------------------------------------------------
        # 3. Size Distribution & Tiny Chunks
        # ---------------------------------------------------------------------
        print("\n[3/5] Evaluating Size Distribution & Tiny Chunks (< 20 tokens)...")

        tiny_chunks = [
            {
                "file_path": c.file_path,
                "lines": f"{c.start_line}-{c.end_line}",
                "tokens": bpe,
                "content_preview": c.content[:60].replace("\n", " "),
            }
            for c, bpe in zip(chunks, bpe_tokens_list, strict=True)
            if bpe < TINY_CHUNK_THRESHOLD
        ]
        pct_tiny = (len(tiny_chunks) / total_chunks * 100) if total_chunks else 0.0

        size_stats = {
            "min_tokens": min(bpe_tokens_list) if bpe_tokens_list else 0,
            "median_tokens": int(statistics.median(bpe_tokens_list)) if bpe_tokens_list else 0,
            "mean_tokens": round(statistics.mean(bpe_tokens_list), 1) if bpe_tokens_list else 0,
            "max_tokens": max(bpe_tokens_list) if bpe_tokens_list else 0,
            "p95_tokens": int(statistics.quantiles(bpe_tokens_list, n=20)[18]) if len(bpe_tokens_list) >= 20 else 0,
            "tiny_chunk_threshold": TINY_CHUNK_THRESHOLD,
            "tiny_chunk_count": len(tiny_chunks),
            "tiny_chunk_pct": round(pct_tiny, 2),
            "sample_tiny_chunks": tiny_chunks[:5],
        }
        print(f"  -> Chunk Size (BPE): Min={size_stats['min_tokens']}, Median={size_stats['median_tokens']}, Max={size_stats['max_tokens']}, P95={size_stats['p95_tokens']}")
        print(f"  -> Tiny Chunks (< {TINY_CHUNK_THRESHOLD} tokens): {len(tiny_chunks)} ({pct_tiny:.2f}%)")

        # ---------------------------------------------------------------------
        # 4. Hygiene Checks
        # ---------------------------------------------------------------------
        print("\n[4/5] Running Chunk Hygiene Checks...")

        empty_chunks = [c.id for c in chunks if not c.content or not c.content.strip()]
        seen_chunks = set()
        exact_duplicates = []
        invalid_line_ranges = []

        for c in chunks:
            # Duplicate check by (file_path, start_line, end_line)
            key = (c.file_path, c.start_line, c.end_line)
            if key in seen_chunks:
                exact_duplicates.append({"chunk_id": c.id, "file_path": c.file_path, "lines": f"{c.start_line}-{c.end_line}"})
            seen_chunks.add(key)

            # Line range validity
            if c.start_line <= 0 or c.end_line < c.start_line or not c.file_path:
                invalid_line_ranges.append({"chunk_id": c.id, "file_path": c.file_path, "lines": f"{c.start_line}-{c.end_line}"})

        hygiene_stats = {
            "empty_chunk_count": len(empty_chunks),
            "exact_duplicate_count": len(exact_duplicates),
            "invalid_line_range_count": len(invalid_line_ranges),
            "all_hygiene_checks_passed": (len(empty_chunks) == 0 and len(exact_duplicates) == 0 and len(invalid_line_ranges) == 0),
        }
        print(f"  -> Empty chunks: {len(empty_chunks)}")
        print(f"  -> Exact duplicate chunks: {len(exact_duplicates)}")
        print(f"  -> Invalid line ranges: {len(invalid_line_ranges)}")
        print(f"  -> Hygiene Check Status: {'PASSED [OK]' if hygiene_stats['all_hygiene_checks_passed'] else 'FAILED'}")

        # ---------------------------------------------------------------------
        # 5. Coverage & Exclusion Verification
        # ---------------------------------------------------------------------
        print("\n[5/5] Checking Repository File Coverage & Exclusion Rules...")

        # Find all valid files in local repo
        expected_files = set()
        for root, dirs, filenames in os.walk(REPO_ROOT):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            for fn in filenames:
                ext = _get_file_extension(fn)
                if ext in SUPPORTED_EXTENSIONS:
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, REPO_ROOT).replace("\\", "/")
                    if os.path.getsize(full) <= MAX_FILE_BYTES:
                        expected_files.add(rel)

        indexed_files = {c.file_path for c in chunks}
        missing_files = expected_files - indexed_files

        # Verify excluded directories and binary extensions
        leakage_count = 0
        for f in indexed_files:
            parts = f.split("/")
            if any(p in EXCLUDED_DIRS for p in parts):
                leakage_count += 1
            if _get_file_extension(f) not in SUPPORTED_EXTENSIONS:
                leakage_count += 1

        coverage_stats = {
            "total_expected_files": len(expected_files),
            "total_indexed_files": len(indexed_files),
            "missing_file_count": len(missing_files),
            "leakage_count": leakage_count,
            "coverage_pct": round(len(indexed_files & expected_files) / len(expected_files) * 100, 2) if expected_files else 100.0,
        }
        print(f"  -> Files Expected: {len(expected_files)}, Files Indexed: {len(indexed_files)}")
        print(f"  -> Coverage: {coverage_stats['coverage_pct']}%")
        print(f"  -> Excluded Directory/Binary Leakage: {leakage_count}")

        # Assemble full report
        report = {
            "token_length_vs_embedder": token_length_stats,
            "boundary_integrity": boundary_stats,
            "size_distribution": size_stats,
            "hygiene": hygiene_stats,
            "coverage": coverage_stats,
        }

        os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
        with open(REPORT_PATH, "w", encoding="utf-8") as out:
            json.dump(report, out, indent=2)

        print(f"\nReport written to: {REPORT_PATH}")
        return report

    finally:
        db.close()


if __name__ == "__main__":
    evaluate_chunking_structural()
