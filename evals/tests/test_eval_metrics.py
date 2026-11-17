"""Unit tests for deterministic IR evaluation metrics."""

import math
import os
import sys

# Ensure evals/scripts is in sys.path
EVALS_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if EVALS_SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, EVALS_SCRIPTS_DIR)

from metrics import (
    hit_at_k,
    is_chunk_relevant,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_file_level_hit():
    chunk = {"file_path": "backend/app/search.py", "start_line": 10, "end_line": 50}
    gt_files = ["backend/app/search.py", "backend/app/indexer.py"]
    assert is_chunk_relevant(chunk, gt_files, level="file") is True

    chunk_other = {"file_path": "backend/app/main.py", "start_line": 1, "end_line": 20}
    assert is_chunk_relevant(chunk_other, gt_files, level="file") is False


def test_symbol_level_hit_overlap():
    chunk_overlap = {"file_path": "backend/app/search.py", "start_line": 150, "end_line": 170}
    chunk_no_overlap = {"file_path": "backend/app/search.py", "start_line": 10, "end_line": 40}

    gt_files = ["backend/app/search.py"]
    gt_symbols = ["_rrf_fusion"]
    symbol_ranges = {
        "backend/app/search.py::_rrf_fusion": {
            "start_line": 158,
            "end_line": 210,
        }
    }

    # Overlaps [158, 210] with [150, 170] -> True
    assert is_chunk_relevant(chunk_overlap, gt_files, gt_symbols, symbol_ranges, level="symbol") is True
    # Does not overlap [158, 210] with [10, 40] -> False
    assert is_chunk_relevant(chunk_no_overlap, gt_files, gt_symbols, symbol_ranges, level="symbol") is False


def test_hit_at_k():
    ranked = [
        {"file_path": "other.py"},
        {"file_path": "target.py"},
        {"file_path": "third.py"},
    ]
    gt_files = ["target.py"]

    assert hit_at_k(ranked, gt_files, k=1) == 0.0
    assert hit_at_k(ranked, gt_files, k=2) == 1.0
    assert hit_at_k(ranked, gt_files, k=5) == 1.0


def test_recall_at_k_file_deduplication():
    # Multiple chunks from the same ground-truth file
    ranked = [
        {"file_path": "file_a.py", "start_line": 1, "end_line": 20},
        {"file_path": "file_a.py", "start_line": 25, "end_line": 50},
        {"file_path": "file_b.py", "start_line": 1, "end_line": 30},
    ]
    gt_files = ["file_a.py", "file_c.py"]

    # At k=2, only file_a is found. Even though there are 2 chunks, recall should be 1/2 = 0.5 (deduplicated)
    assert recall_at_k(ranked, gt_files, k=2) == 0.5


def test_precision_at_k():
    ranked = [
        {"file_path": "target.py"},
        {"file_path": "noise.py"},
        {"file_path": "target.py"},
        {"file_path": "noise.py"},
    ]
    gt_files = ["target.py"]

    assert precision_at_k(ranked, gt_files, k=1) == 1.0  # 1/1
    assert precision_at_k(ranked, gt_files, k=2) == 0.5  # 1/2
    assert precision_at_k(ranked, gt_files, k=4) == 0.5  # 2/4


def test_mrr_first_rank():
    ranked = [
        {"file_path": "noise.py"},
        {"file_path": "noise2.py"},
        {"file_path": "target.py"},  # 3rd position
    ]
    gt_files = ["target.py"]

    assert mrr(ranked, gt_files) == 1.0 / 3.0

    # No match
    assert mrr(ranked, ["absent.py"]) == 0.0


def test_ndcg_at_k():
    # Perfect ranking: target at rank 1
    ranked_perfect = [{"file_path": "target.py"}, {"file_path": "noise.py"}]
    gt_files = ["target.py"]
    assert ndcg_at_k(ranked_perfect, gt_files, k=2) == 1.0

    # Suboptimal ranking: target at rank 2
    ranked_suboptimal = [{"file_path": "noise.py"}, {"file_path": "target.py"}]
    # DCG = 1 / log2(3); IDCG = 1 / log2(2) = 1.0
    expected_ndcg = (1.0 / math.log2(3)) / 1.0
    assert abs(ndcg_at_k(ranked_suboptimal, gt_files, k=2) - expected_ndcg) < 1e-6


def test_ndcg_at_k_file_deduplication():
    # Regression test for nDCG > 1.0 bug:
    # 5 chunks all from the same ground-truth file must NOT accumulate gain multiple times.
    ranked = [
        {"file_path": "search.py", "start_line": 1, "end_line": 50},
        {"file_path": "search.py", "start_line": 51, "end_line": 100},
        {"file_path": "search.py", "start_line": 101, "end_line": 150},
        {"file_path": "search.py", "start_line": 151, "end_line": 200},
        {"file_path": "search.py", "start_line": 201, "end_line": 250},
    ]
    gt_files = ["search.py"]

    # Rank 1 gains 1.0, ranks 2-5 are duplicates with gain 0.0.
    # DCG = 1.0 / log2(2) = 1.0. IDCG = 1.0. nDCG@5 MUST be exactly 1.0.
    assert ndcg_at_k(ranked, gt_files, k=5) == 1.0


def test_ndcg_at_k_multiple_ground_truth_files():
    ranked = [
        {"file_path": "file_b.py"},  # target 1 at rank 1 -> gain 1.0 / log2(2) = 1.0
        {"file_path": "noise.py"},   # noise at rank 2 -> gain 0.0
        {"file_path": "file_a.py"},  # target 2 at rank 3 -> gain 1.0 / log2(4) = 0.5
    ]
    gt_files = ["file_a.py", "file_b.py"]

    expected_dcg = (1.0 / math.log2(2)) + (1.0 / math.log2(4))  # 1.0 + 0.5 = 1.5
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))  # 1.0 + 0.63092975...
    expected_ndcg = expected_dcg / expected_idcg

    actual_ndcg = ndcg_at_k(ranked, gt_files, k=3)
    assert abs(actual_ndcg - expected_ndcg) < 1e-6
    assert 0.0 <= actual_ndcg <= 1.0


def test_ndcg_at_k_symbol_deduplication():
    # Multiple chunks touching the same symbol
    gt_files = ["search.py"]
    gt_symbols = ["_rrf_fusion"]
    symbol_ranges = {
        "search.py::_rrf_fusion": {"start_line": 158, "end_line": 210}
    }

    ranked = [
        {"file_path": "search.py", "start_line": 158, "end_line": 180},  # overlap 1st -> gain 1.0
        {"file_path": "search.py", "start_line": 180, "end_line": 210},  # overlap duplicate -> gain 0.0
        {"file_path": "search.py", "start_line": 1, "end_line": 50},     # no overlap -> gain 0.0
    ]

    # DCG@3 = 1.0 / log2(2) = 1.0; IDCG@3 = 1.0; nDCG@3 = 1.0
    assert ndcg_at_k(ranked, gt_files, k=3, gt_symbols=gt_symbols, symbol_ranges=symbol_ranges, level="symbol") == 1.0


def test_ndcg_at_k_symbol_multi_symbol():
    gt_files = ["search.py"]
    gt_symbols = ["RRF_K", "_rrf_fusion"]
    symbol_ranges = {
        "search.py::RRF_K": {"start_line": 28, "end_line": 28},
        "search.py::_rrf_fusion": {"start_line": 158, "end_line": 210},
    }

    ranked = [
        {"file_path": "search.py", "start_line": 1, "end_line": 20},     # noise
        {"file_path": "search.py", "start_line": 25, "end_line": 30},    # hits RRF_K at rank 2
        {"file_path": "search.py", "start_line": 25, "end_line": 35},    # duplicate RRF_K at rank 3
        {"file_path": "search.py", "start_line": 158, "end_line": 200},  # hits _rrf_fusion at rank 4
    ]

    expected_dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(5))
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    expected_ndcg = expected_dcg / expected_idcg

    actual_ndcg = ndcg_at_k(ranked, gt_files, k=4, gt_symbols=gt_symbols, symbol_ranges=symbol_ranges, level="symbol")
    assert abs(actual_ndcg - expected_ndcg) < 1e-6
    assert 0.0 <= actual_ndcg <= 1.0


def test_candidate_pool_metrics():
    # Test pool metrics (k=50 ceiling)
    gt_files = ["file_a.py", "file_b.py"]
    # 50 candidates, target at rank 45
    ranked_pool = [{"file_path": f"noise_{i}.py"} for i in range(44)]
    ranked_pool.append({"file_path": "file_a.py"})
    ranked_pool.extend([{"file_path": f"noise_{i}.py"} for i in range(45, 50)])

    assert len(ranked_pool) == 50
    # At k=5, target is not in top 5
    assert hit_at_k(ranked_pool, gt_files, k=5) == 0.0
    assert recall_at_k(ranked_pool, gt_files, k=5) == 0.0
    # At k=50, target is in pool
    assert hit_at_k(ranked_pool, gt_files, k=50) == 1.0
    assert recall_at_k(ranked_pool, gt_files, k=50) == 0.5  # 1 out of 2 found
