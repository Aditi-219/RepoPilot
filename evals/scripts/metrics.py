"""Deterministic evaluation metrics for RepoPilot retrieval evaluation.

Implements standard Information Retrieval (IR) metrics without requiring LLMs:
- Hit@k (k = 1, 3, 5, 10)
- Recall@k (with file-level deduplication)
- Precision@k
- MRR (Mean Reciprocal Rank)
- nDCG@k (Normalized Discounted Cumulative Gain)

Supports two relevance criteria:
1. File-level (default): Chunk file_path matches any gt_files.
2. Symbol-level (stricter): Chunk file_path matches and its line range overlaps
   the line range of a ground-truth symbol.
"""

import math
from typing import Any


def is_chunk_relevant(
    chunk: dict[str, Any],
    gt_files: list[str],
    gt_symbols: list[str] | None = None,
    symbol_ranges: dict[str, dict[str, Any]] | None = None,
    level: str = "file",
) -> bool:
    """Check if a single retrieved chunk counts as a hit.

    Args:
        chunk: Retrieved chunk dictionary with 'file_path', 'start_line', 'end_line'.
        gt_files: List of ground-truth file paths.
        gt_symbols: Optional list of ground-truth symbol names.
        symbol_ranges: Precomputed line ranges dictionary keyed by 'file_path::symbol'.
        level: 'file' for file-level match, 'symbol' for line-overlap with a gt_symbol.

    Returns:
        bool: True if relevant, False otherwise.
    """
    file_path = chunk.get("file_path", "")
    if file_path not in gt_files:
        return False

    if level == "file":
        return True

    # Symbol-level relevance: must overlap at least one symbol in this file
    if not gt_symbols or not symbol_ranges:
        return False

    c_start = chunk.get("start_line", 0)
    c_end = chunk.get("end_line", 0)

    for sym in gt_symbols:
        key = f"{file_path}::{sym}"
        if key in symbol_ranges:
            s_start = symbol_ranges[key]["start_line"]
            s_end = symbol_ranges[key]["end_line"]
            # Line range overlap condition: max(start) <= min(end)
            if max(c_start, s_start) <= min(c_end, s_end):
                return True

    return False


def hit_at_k(
    ranked_chunks: list[dict[str, Any]],
    gt_files: list[str],
    k: int,
    gt_symbols: list[str] | None = None,
    symbol_ranges: dict[str, dict[str, Any]] | None = None,
    level: str = "file",
) -> float:
    """Is at least one relevant chunk in the top k? (Returns 1.0 or 0.0)."""
    if not gt_files:
        return 0.0
    for chunk in ranked_chunks[:k]:
        if is_chunk_relevant(chunk, gt_files, gt_symbols, symbol_ranges, level):
            return 1.0
    return 0.0


def recall_at_k(
    ranked_chunks: list[dict[str, Any]],
    gt_files: list[str],
    k: int,
    gt_symbols: list[str] | None = None,
    symbol_ranges: dict[str, dict[str, Any]] | None = None,
    level: str = "file",
) -> float:
    """Fraction of ground-truth targets retrieved in top k.

    For file-level: Deduplicates files across chunks so multiple chunks from the
    same file do not artificially inflate recall.
    """
    if not gt_files:
        return 0.0

    if level == "file":
        found_files = set()
        for chunk in ranked_chunks[:k]:
            f = chunk.get("file_path", "")
            if f in gt_files:
                found_files.add(f)
        return len(found_files) / len(gt_files)

    # Symbol-level recall: fraction of gt_symbols retrieved
    if not gt_symbols:
        return 0.0
    found_symbols = set()
    for chunk in ranked_chunks[:k]:
        f = chunk.get("file_path", "")
        if f not in gt_files:
            continue
        c_start = chunk.get("start_line", 0)
        c_end = chunk.get("end_line", 0)
        for sym in gt_symbols:
            key = f"{f}::{sym}"
            if key in (symbol_ranges or {}):
                s_start = symbol_ranges[key]["start_line"]
                s_end = symbol_ranges[key]["end_line"]
                if max(c_start, s_start) <= min(c_end, s_end):
                    found_symbols.add(sym)
    return len(found_symbols) / len(gt_symbols)


def precision_at_k(
    ranked_chunks: list[dict[str, Any]],
    gt_files: list[str],
    k: int,
    gt_symbols: list[str] | None = None,
    symbol_ranges: dict[str, dict[str, Any]] | None = None,
    level: str = "file",
) -> float:
    """Fraction of top k retrieved chunks that are relevant."""
    if k <= 0:
        return 0.0
    top_k = ranked_chunks[:k]
    if not top_k:
        return 0.0
    relevant_count = sum(
        1 for c in top_k
        if is_chunk_relevant(c, gt_files, gt_symbols, symbol_ranges, level)
    )
    return relevant_count / k


def mrr(
    ranked_chunks: list[dict[str, Any]],
    gt_files: list[str],
    gt_symbols: list[str] | None = None,
    symbol_ranges: dict[str, dict[str, Any]] | None = None,
    level: str = "file",
) -> float:
    """Reciprocal rank of the first relevant chunk (1 / rank), or 0.0 if not found."""
    if not gt_files:
        return 0.0
    for rank_idx, chunk in enumerate(ranked_chunks, start=1):
        if is_chunk_relevant(chunk, gt_files, gt_symbols, symbol_ranges, level):
            return 1.0 / rank_idx
    return 0.0


def ndcg_at_k(
    ranked_chunks: list[dict[str, Any]],
    gt_files: list[str],
    k: int,
    gt_symbols: list[str] | None = None,
    symbol_ranges: dict[str, dict[str, Any]] | None = None,
    level: str = "file",
) -> float:
    """Normalized Discounted Cumulative Gain at rank k with target-level deduplication.

    For file-level:
        Each unique ground-truth file contributes relevance gain 1.0 on its first
        retrieval in top-k. Subsequent duplicate chunks from the same file yield
        gain 0.0 to prevent DCG inflation from multi-chunk files.

    For symbol-level:
        Each unique ground-truth symbol contributes relevance gain 1.0 on the first
        retrieved chunk overlapping its line range. Subsequent duplicate chunks
        overlapping already-seen symbols yield gain 0.0.

    IDCG is computed over min(k, total unique ground-truth targets). This guarantees
    DCG <= IDCG and nDCG in [0.0, 1.0].
    """
    if not gt_files or k <= 0:
        return 0.0

    top_k = ranked_chunks[:k]
    if not top_k:
        return 0.0

    if level == "file":
        targets = set(gt_files)
        seen_files: set[str] = set()
        dcg = 0.0
        for i, chunk in enumerate(top_k, start=1):
            f = chunk.get("file_path", "")
            if f in targets and f not in seen_files:
                seen_files.add(f)
                dcg += 1.0 / math.log2(i + 1)
        ideal_count = min(k, len(targets))
    else:
        if not gt_symbols:
            return 0.0
        targets = set(gt_symbols)
        seen_symbols: set[str] = set()
        dcg = 0.0
        for i, chunk in enumerate(top_k, start=1):
            f = chunk.get("file_path", "")
            if f not in gt_files:
                continue
            c_start = chunk.get("start_line", 0)
            c_end = chunk.get("end_line", 0)
            claimed = False
            for sym in gt_symbols:
                if sym in targets and sym not in seen_symbols:
                    key = f"{f}::{sym}"
                    if key in (symbol_ranges or {}):
                        s_start = symbol_ranges[key]["start_line"]
                        s_end = symbol_ranges[key]["end_line"]
                        if max(c_start, s_start) <= min(c_end, s_end):
                            seen_symbols.add(sym)
                            claimed = True
                            break
            if claimed:
                dcg += 1.0 / math.log2(i + 1)
        ideal_count = min(k, len(targets))

    if ideal_count <= 0:
        return 0.0

    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_count + 1))
    return dcg / idcg if idcg > 0.0 else 0.0


def compute_all_metrics(
    ranked_chunks: list[dict[str, Any]],
    gt_files: list[str],
    gt_symbols: list[str] | None = None,
    symbol_ranges: dict[str, dict[str, Any]] | None = None,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    level: str = "file",
) -> dict[str, float]:
    """Compute all evaluation metrics for a single ranked list."""
    results: dict[str, float] = {}

    for k in k_values:
        results[f"Hit@{k}"] = round(hit_at_k(ranked_chunks, gt_files, k, gt_symbols, symbol_ranges, level), 4)
        results[f"Recall@{k}"] = round(recall_at_k(ranked_chunks, gt_files, k, gt_symbols, symbol_ranges, level), 4)
        results[f"Precision@{k}"] = round(precision_at_k(ranked_chunks, gt_files, k, gt_symbols, symbol_ranges, level), 4)

    results["MRR"] = round(mrr(ranked_chunks, gt_files, gt_symbols, symbol_ranges, level), 4)

    for k in (3, 5, 10):
        results[f"nDCG@{k}"] = round(ndcg_at_k(ranked_chunks, gt_files, k, gt_symbols, symbol_ranges, level), 4)

    return results
