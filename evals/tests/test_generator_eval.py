"""Unit tests for data-loading and aggregation logic in run_generator_eval.py.

No real LLM calls, database connections, or DeepEval operations are performed.
All external dependencies (backend app, deepeval, httpx, dotenv) are mocked
at the sys.modules level before the module under test is imported.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from unittest import mock

# ---------------------------------------------------------------------------
# Mock all external dependencies BEFORE importing run_generator_eval
# so the module loads cleanly without a live backend or deepeval.
# ---------------------------------------------------------------------------
_MOCKED_MODULES = [
    "app",
    "app.database",
    "app.llm_service",
    "app.models",
    "app.reranker",
    "app.search",
    "sqlalchemy",
    "sqlalchemy.orm",
    "dotenv",
    "httpx",
    "deepeval",
    "deepeval.models",
    "deepeval.metrics",
    "deepeval.test_case",
]
for _mod_name in _MOCKED_MODULES:
    if _mod_name not in sys.modules:
        _m = mock.MagicMock()
        _spec = mock.MagicMock()
        _spec.name = _mod_name
        _m.__spec__ = _spec
        sys.modules[_mod_name] = _m

# Provide a no-op load_dotenv so the module-level call does nothing.
sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None

# Ensure evals/scripts is importable.
_EVALS_SCRIPTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if _EVALS_SCRIPTS not in sys.path:
    sys.path.insert(0, _EVALS_SCRIPTS)

# Import the pure utility functions under test.
from run_generator_eval import (  # noqa: E402
    NOISE_SUBSET_IDS,
    REFUSAL_PHRASES,
    THRESHOLDS,
    aggregate_scores,
    check_citation_validity,
    compute_citation_file_hit,
    compute_symbol_hit,
    extract_citations,
    filter_answerable,
    filter_by_category,
    filter_unanswerable,
    find_worst_n,
    format_table,
    is_refusal,
    load_dataset,
    load_symbol_ranges,
    make_answer_cache_key,
    make_judge_cache_key,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATASET = [
    {
        "id": "rp-001",
        "answerable": True,
        "category": "locate",
        "question": "Where is RRF implemented?",
        "ideal_answer": "In _rrf_fusion inside backend/app/search.py",
        "gt_files": ["backend/app/search.py"],
        "gt_symbols": ["_rrf_fusion"],
    },
    {
        "id": "rp-002",
        "answerable": True,
        "category": "explain",
        "question": "How does hybrid search work?",
        "ideal_answer": "...",
        "gt_files": ["backend/app/search.py"],
        "gt_symbols": [],
    },
    {
        "id": "rp-019",
        "answerable": False,
        "category": "unanswerable",
        "question": "How does Redis caching work?",
        "ideal_answer": "Redis is not used in RepoPilot.",
        "gt_files": [],
        "gt_symbols": [],
    },
    {
        "id": "rp-021",
        "answerable": False,
        "category": "out_of_scope",
        "question": "Write a poem about the sea.",
        "ideal_answer": "Out of scope.",
        "gt_files": [],
        "gt_symbols": [],
    },
]


# ---------------------------------------------------------------------------
# load_dataset
# ---------------------------------------------------------------------------


def test_load_dataset_parses_valid_jsonl(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"id": "q1", "answerable": true}\n'
        '{"id": "q2", "answerable": false}\n',
        encoding="utf-8",
    )
    result = load_dataset(str(p))
    assert len(result) == 2
    assert result[0]["id"] == "q1"
    assert result[1]["id"] == "q2"


def test_load_dataset_skips_blank_lines(tmp_path):
    p = tmp_path / "data.jsonl"
    p.write_text(
        '{"id": "q1"}\n\n\n{"id": "q2"}\n',
        encoding="utf-8",
    )
    result = load_dataset(str(p))
    assert len(result) == 2


def test_load_symbol_ranges_missing_file(tmp_path):
    result = load_symbol_ranges(str(tmp_path / "nonexistent.json"))
    assert result == {}


def test_load_symbol_ranges_valid_file(tmp_path):
    data = {"search.py::_rrf_fusion": {"start_line": 158, "end_line": 210}}
    p = tmp_path / "ranges.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    result = load_symbol_ranges(str(p))
    assert "search.py::_rrf_fusion" in result
    assert result["search.py::_rrf_fusion"]["start_line"] == 158


# ---------------------------------------------------------------------------
# filter_* functions
# ---------------------------------------------------------------------------


def test_filter_answerable_keeps_only_answerable():
    result = filter_answerable(SAMPLE_DATASET)
    assert len(result) == 2
    assert all(r["answerable"] for r in result)
    assert {r["id"] for r in result} == {"rp-001", "rp-002"}


def test_filter_unanswerable_keeps_only_non_answerable():
    result = filter_unanswerable(SAMPLE_DATASET)
    assert len(result) == 2
    assert all(not r["answerable"] for r in result)
    assert {r["id"] for r in result} == {"rp-019", "rp-021"}


def test_filter_by_category_unanswerable():
    result = filter_by_category(SAMPLE_DATASET, "unanswerable")
    assert len(result) == 1
    assert result[0]["id"] == "rp-019"


def test_filter_by_category_out_of_scope():
    result = filter_by_category(SAMPLE_DATASET, "out_of_scope")
    assert len(result) == 1
    assert result[0]["id"] == "rp-021"


def test_filter_by_category_no_match():
    result = filter_by_category(SAMPLE_DATASET, "nonexistent")
    assert result == []


def test_filter_answerable_default_true_when_key_missing():
    # An example without an 'answerable' key should be treated as answerable.
    dataset = [{"id": "q1"}, {"id": "q2", "answerable": False}]
    result = filter_answerable(dataset)
    assert len(result) == 1
    assert result[0]["id"] == "q1"


# ---------------------------------------------------------------------------
# Cache key functions
# ---------------------------------------------------------------------------


def test_make_answer_cache_key_is_deterministic():
    k1 = make_answer_cache_key("qwen", 0.2, "sys", "What is X?", ["ctx_a", "ctx_b"])
    k2 = make_answer_cache_key("qwen", 0.2, "sys", "What is X?", ["ctx_a", "ctx_b"])
    assert k1 == k2


def test_make_answer_cache_key_is_context_order_insensitive():
    # Contexts are sorted before hashing so order must not matter.
    k1 = make_answer_cache_key("qwen", 0.2, "sys", "Q?", ["ctx_a", "ctx_b"])
    k2 = make_answer_cache_key("qwen", 0.2, "sys", "Q?", ["ctx_b", "ctx_a"])
    assert k1 == k2


def test_make_answer_cache_key_differs_by_generator_model():
    k1 = make_answer_cache_key("qwen2.5-coder:1.5b", 0.2, "sys", "Q?", ["ctx"])
    k2 = make_answer_cache_key("llama3.2", 0.2, "sys", "Q?", ["ctx"])
    assert k1 != k2


def test_make_answer_cache_key_differs_by_temperature():
    k1 = make_answer_cache_key("qwen", 0.2, "sys", "Q?", ["ctx"])
    k2 = make_answer_cache_key("qwen", 0.7, "sys", "Q?", ["ctx"])
    assert k1 != k2


def test_make_answer_cache_key_differs_by_system_prompt():
    k1 = make_answer_cache_key("qwen", 0.2, "System prompt version 1", "Q?", ["ctx"])
    k2 = make_answer_cache_key("qwen", 0.2, "System prompt version 2", "Q?", ["ctx"])
    assert k1 != k2


def test_make_answer_cache_key_differs_by_question():
    k1 = make_answer_cache_key("qwen", 0.2, "sys", "Q1?", ["ctx"])
    k2 = make_answer_cache_key("qwen", 0.2, "sys", "Q2?", ["ctx"])
    assert k1 != k2


def test_make_answer_cache_key_differs_by_context():
    k1 = make_answer_cache_key("qwen", 0.2, "sys", "Q?", ["ctx_a"])
    k2 = make_answer_cache_key("qwen", 0.2, "sys", "Q?", ["ctx_b"])
    assert k1 != k2


def test_make_judge_cache_key_deterministic():
    k = make_judge_cache_key("gemini-3.5-flash-lite", 0.0, "some judge prompt")
    assert make_judge_cache_key("gemini-3.5-flash-lite", 0.0, "some judge prompt") == k


def test_make_judge_cache_key_unique_per_prompt():
    k1 = make_judge_cache_key("gemini-3.5-flash-lite", 0.0, "prompt A")
    k2 = make_judge_cache_key("gemini-3.5-flash-lite", 0.0, "prompt B")
    assert k1 != k2


def test_make_judge_cache_key_unique_per_judge_model():
    k1 = make_judge_cache_key("gemini-3.1-flash-lite", 0.0, "prompt")
    k2 = make_judge_cache_key("gemini-3.5-flash-lite", 0.0, "prompt")
    assert k1 != k2


def test_cache_miss_on_changed_system_prompt_judge_model_and_context():
    """Unit test explicitly required by user:
    Proves that a changed system prompt, a changed judge model, and a changed
    context each independently produce a cache miss.
    """
    base_answer_cache = {}
    base_judge_cache = {}

    # 1. Base keys
    ans_key_base = make_answer_cache_key(
        generator_model="qwen2.5-coder:1.5b",
        temperature=0.2,
        system_prompt="You are RepoPilot. Base answers on snippets.",
        question="Where is RRF?",
        context_chunks=["File: reranker.py:L1-10\ndef _rrf(): pass"],
    )
    judge_key_base = make_judge_cache_key(
        judge_model="gemini-3.5-flash-lite",
        temperature=0.0,
        prompt="Score this answer for faithfulness: ...",
    )

    base_answer_cache[ans_key_base] = "cached answer"
    base_judge_cache[judge_key_base] = "cached judgment"

    # Case A: Changed system prompt produces answer cache miss
    ans_key_changed_prompt = make_answer_cache_key(
        generator_model="qwen2.5-coder:1.5b",
        temperature=0.2,
        system_prompt="You are RepoPilot v2. Always cite line numbers.",  # changed!
        question="Where is RRF?",
        context_chunks=["File: reranker.py:L1-10\ndef _rrf(): pass"],
    )
    assert ans_key_changed_prompt not in base_answer_cache, "Changed system prompt must cause cache miss"
    assert ans_key_changed_prompt != ans_key_base

    # Case B: Changed judge model produces judge cache miss
    judge_key_changed_model = make_judge_cache_key(
        judge_model="gemini-3.1-flash-lite",  # changed model!
        temperature=0.0,
        prompt="Score this answer for faithfulness: ...",
    )
    assert judge_key_changed_model not in base_judge_cache, "Changed judge model must cause cache miss"
    assert judge_key_changed_model != judge_key_base

    # Case C: Changed context produces answer cache miss
    ans_key_changed_context = make_answer_cache_key(
        generator_model="qwen2.5-coder:1.5b",
        temperature=0.2,
        system_prompt="You are RepoPilot. Base answers on snippets.",
        question="Where is RRF?",
        context_chunks=["File: search.py:L20-30\ndef hybrid_search(): pass"],  # changed context!
    )
    assert ans_key_changed_context not in base_answer_cache, "Changed context must cause cache miss"
    assert ans_key_changed_context != ans_key_base


# ---------------------------------------------------------------------------
# is_refusal
# ---------------------------------------------------------------------------


def test_is_refusal_detects_known_phrases():
    assert is_refusal("Redis is not found in the codebase.")
    assert is_refusal("Celery is not used in RepoPilot.")
    assert is_refusal("This is outside the scope of RepoPilot.")
    assert is_refusal("I cannot answer questions about quantum physics.")
    assert is_refusal("There is no information about this in the codebase.")
    assert is_refusal("This feature is not present in the codebase.")
    assert is_refusal("I can't answer this question.")
    assert is_refusal("ChromaDB is not part of RepoPilot.")


def test_is_refusal_case_insensitive():
    assert is_refusal("REDIS IS NOT FOUND IN THE CODEBASE")
    assert is_refusal("Not Part Of RepoPilot")
    assert is_refusal("OUTSIDE THE SCOPE")


def test_is_refusal_negative_on_real_answers():
    # Real technical answers should NOT trigger refusal detection.
    assert not is_refusal(
        "RRF is implemented in backend/app/search.py in the _rrf_fusion function "
        "using the constant RRF_K = 60."
    )
    assert not is_refusal(
        "The reranker uses a cross-encoder/ms-marco-MiniLM-L-6-v2 model."
    )
    assert not is_refusal(
        "The function is defined at backend/app/search.py:L193-L245."
    )


def test_is_refusal_no_false_positive_on_code_path():
    # File paths and function names should not trigger refusal.
    assert not is_refusal(
        "See backend/app/main.py for the lifespan context manager."
    )


def test_refusal_phrases_are_all_lowercase():
    """All phrases must be lowercase — is_refusal() lowercases the input before matching."""
    for phrase in REFUSAL_PHRASES:
        assert phrase == phrase.lower(), (
            f"REFUSAL_PHRASES entry '{phrase}' is not lowercase. "
            "is_refusal() lowercases the answer before matching."
        )


# ---------------------------------------------------------------------------
# aggregate_scores
# ---------------------------------------------------------------------------


def test_aggregate_scores_basic():
    scores = [("q1", 0.8, "ok"), ("q2", 0.4, "bad"), ("q3", 0.9, "great")]
    result = aggregate_scores(scores, threshold=0.7)
    assert result["n"] == 3
    expected_mean = round((0.8 + 0.4 + 0.9) / 3, 4)
    assert abs(result["mean"] - expected_mean) < 1e-6
    # q1 (0.8 ≥ 0.7) and q3 (0.9 ≥ 0.7) pass → 2/3
    assert result["pass_rate"] == round(2 / 3, 4)
    assert result["threshold"] == 0.7


def test_aggregate_scores_all_pass():
    scores = [("q1", 1.0, None), ("q2", 0.8, None), ("q3", 0.75, None)]
    result = aggregate_scores(scores, threshold=0.7)
    assert result["pass_rate"] == 1.0


def test_aggregate_scores_none_pass():
    scores = [("q1", 0.3, None), ("q2", 0.1, None)]
    result = aggregate_scores(scores, threshold=0.7)
    assert result["pass_rate"] == 0.0


def test_aggregate_scores_empty():
    result = aggregate_scores([], threshold=0.5)
    assert result["mean"] == 0.0
    assert result["pass_rate"] == 0.0
    assert result["n"] == 0


def test_aggregate_scores_exact_threshold_passes():
    # score == threshold must count as passing (≥ not >)
    scores = [("q1", 0.7, None)]
    result = aggregate_scores(scores, threshold=0.7)
    assert result["pass_rate"] == 1.0


def test_aggregate_scores_just_below_threshold_fails():
    scores = [("q1", 0.699, None)]
    result = aggregate_scores(scores, threshold=0.7)
    assert result["pass_rate"] == 0.0


# ---------------------------------------------------------------------------
# find_worst_n
# ---------------------------------------------------------------------------


def test_find_worst_n_sorts_ascending():
    per_question = {
        "q1": {"question": "Q1", "scores": {"faithfulness": {"score": 0.8, "reason": "ok"}}},
        "q2": {"question": "Q2", "scores": {"faithfulness": {"score": 0.3, "reason": "bad"}}},
        "q3": {"question": "Q3", "scores": {"faithfulness": {"score": 0.5, "reason": "meh"}}},
    }
    worst = find_worst_n(per_question, "faithfulness", n=2)
    assert len(worst) == 2
    assert worst[0]["score"] <= worst[1]["score"]
    assert worst[0]["score"] == 0.3
    assert worst[0]["id"] == "q2"


def test_find_worst_n_respects_limit():
    per_question = {
        f"q{i}": {
            "question": f"Q{i}",
            "scores": {"faithfulness": {"score": i * 0.1, "reason": ""}},
        }
        for i in range(10)
    }
    worst = find_worst_n(per_question, "faithfulness", n=3)
    assert len(worst) == 3
    # Lowest three scores should be 0.0, 0.1, 0.2
    assert worst[0]["score"] == 0.0
    assert worst[2]["score"] == 0.2


def test_find_worst_n_missing_metric_returns_empty():
    per_question = {
        "q1": {
            "question": "Q1",
            "scores": {"answer_relevancy": {"score": 0.5, "reason": ""}},
        },
    }
    worst = find_worst_n(per_question, "faithfulness", n=5)
    assert worst == []


def test_find_worst_n_empty_per_question():
    assert find_worst_n({}, "faithfulness", n=5) == []


# ---------------------------------------------------------------------------
# format_table
# ---------------------------------------------------------------------------


def test_format_table_contains_all_values():
    headers = ["Metric", "Mean", "Pass Rate"]
    rows = [["faithfulness", "0.800", "75.0%"]]
    table = format_table(headers, rows)
    assert "Metric" in table
    assert "faithfulness" in table
    assert "0.800" in table
    assert "75.0%" in table


def test_format_table_has_correct_line_count():
    headers = ["A", "B"]
    rows = [["row1a", "row1b"], ["row2a", "row2b"]]
    table = format_table(headers, rows)
    lines = table.strip().split("\n")
    # 1 header + 1 separator + 2 data rows = 4 lines
    assert len(lines) == 4


def test_format_table_columns_are_pipe_separated():
    headers = ["X", "Y"]
    rows = [["a", "b"]]
    table = format_table(headers, rows)
    assert "|" in table


# ---------------------------------------------------------------------------
# Constants sanity checks
# ---------------------------------------------------------------------------


def test_thresholds_are_in_valid_range():
    for name, val in THRESHOLDS.items():
        assert 0.0 < val <= 1.0, f"Threshold for '{name}' = {val} is out of range (0, 1]"


def test_noise_subset_has_eight_questions():
    assert len(NOISE_SUBSET_IDS) == 8, (
        "NOISE_SUBSET_IDS must contain exactly 8 questions "
        f"(got {len(NOISE_SUBSET_IDS)})"
    )


def test_noise_subset_ids_have_correct_prefix():
    for qid in NOISE_SUBSET_IDS:
        assert qid.startswith("rp-"), f"Noise subset ID '{qid}' should start with 'rp-'"


def test_noise_subset_covers_distinct_questions():
    assert len(NOISE_SUBSET_IDS) == len(set(NOISE_SUBSET_IDS)), (
        "NOISE_SUBSET_IDS contains duplicate question IDs"
    )


def test_refusal_phrases_not_empty():
    assert len(REFUSAL_PHRASES) > 0, "REFUSAL_PHRASES must not be empty"


def test_refusal_phrases_no_duplicates():
    assert len(REFUSAL_PHRASES) == len(set(REFUSAL_PHRASES)), (
        "REFUSAL_PHRASES contains duplicate entries"
    )


# ---------------------------------------------------------------------------
# Citation analysis tests (Part A)
# ---------------------------------------------------------------------------


def test_extract_citations_finds_standard_formats():
    text = (
        "See `backend/app/search.py:L158-L210` for RRF fusion. "
        "Also defined at alembic/versions/001_schema.py:10-50."
    )
    citations = extract_citations(text)
    assert len(citations) == 2
    assert citations[0] == ("backend/app/search.py", 158, 210)
    assert citations[1] == ("alembic/versions/001_schema.py", 10, 50)


def test_extract_citations_new_bracket_format():
    text = (
        "RRF is implemented in [backend/app/search.py:L193-L245, _rrf_fusion]. "
        "Another reference is [docs/08-backend.md:1-20]."
    )
    citations = extract_citations(text)
    assert len(citations) == 2
    assert citations[0] == ("backend/app/search.py", 193, 245, "_rrf_fusion")
    assert citations[1] == ("docs/08-backend.md", 1, 20)


def test_extract_citations_mixed_old_and_new_formats():
    text = (
        "Old format: backend/app/search.py:L100-L150. "
        "New format: [backend/app/reranker.py:L10-L40, rerank]."
    )
    citations = extract_citations(text)
    assert len(citations) == 2
    assert citations[0] == ("backend/app/reranker.py", 10, 40, "rerank")
    assert citations[1] == ("backend/app/search.py", 100, 150)


def test_check_citation_validity_with_symbol_match():
    from unittest.mock import MagicMock

    chunk1 = MagicMock()
    chunk1.file_path = "backend/app/search.py"
    chunk1.start_line = 100
    chunk1.end_line = 200
    chunk1.content = "def _rrf_fusion(semantic, keyword):\n    pass"

    citations = [("backend/app/search.py", 110, 150, "_rrf_fusion")]
    validity, invalid = check_citation_validity(citations, [chunk1])
    assert validity == 1.0
    assert invalid == []


def test_check_citation_validity_with_symbol_mismatch():
    from unittest.mock import MagicMock

    chunk1 = MagicMock()
    chunk1.file_path = "backend/app/search.py"
    chunk1.start_line = 100
    chunk1.end_line = 200
    chunk1.content = "def other_function():\n    pass"

    citations = [("backend/app/search.py", 110, 150, "_rrf_fusion")]
    validity, invalid = check_citation_validity(citations, [chunk1])
    assert validity == 0.0
    assert len(invalid) == 1
    assert invalid[0] == "[backend/app/search.py:L110-L150, _rrf_fusion]"


def test_compute_citation_file_hit_matches_path():
    ans = "In backend/app/search.py, RRF is implemented."
    assert compute_citation_file_hit(ans, ["backend/app/search.py"])
    assert compute_citation_file_hit(ans, ["docs/overview.md", "backend/app/search.py"])
    assert not compute_citation_file_hit(ans, ["backend/app/reranker.py"])


def test_compute_symbol_hit_matches_symbol():
    ans = "The function _rrf_fusion combines rankings."
    assert compute_symbol_hit(ans, ["_rrf_fusion"])
    assert not compute_symbol_hit(ans, ["rerank"])
    assert compute_symbol_hit(ans, []) is None


def test_check_citation_validity_valid_overlap():
    from unittest.mock import MagicMock

    chunk1 = MagicMock()
    chunk1.file_path = "backend/app/search.py"
    chunk1.start_line = 100
    chunk1.end_line = 200

    citations = [("backend/app/search.py", 150, 180)]
    validity, invalid = check_citation_validity(citations, [chunk1])
    assert validity == 1.0
    assert invalid == []


def test_check_citation_validity_out_of_range():
    from unittest.mock import MagicMock

    chunk1 = MagicMock()
    chunk1.file_path = "backend/app/search.py"
    chunk1.start_line = 100
    chunk1.end_line = 200

    citations = [("backend/app/search.py", 300, 350)]
    validity, invalid = check_citation_validity(citations, [chunk1])
    assert validity == 0.0
    assert len(invalid) == 1


# ---------------------------------------------------------------------------
# Rate limit & Concurrency tests
# ---------------------------------------------------------------------------


def test_parse_retry_delay_from_header():
    from unittest.mock import MagicMock
    from run_generator_eval import _parse_retry_delay

    resp = MagicMock()
    resp.headers = {"retry-after": "45"}
    resp.json.return_value = {}
    assert _parse_retry_delay(resp) == 45.0


def test_parse_retry_delay_from_json_details():
    from unittest.mock import MagicMock
    from run_generator_eval import _parse_retry_delay

    resp = MagicMock()
    resp.headers = {}
    resp.json.return_value = {
        "error": {
            "details": [
                {"@type": "RetryInfo", "retryDelay": "39083s"}
            ]
        }
    }
    assert _parse_retry_delay(resp) == 39083.0


def test_parse_retry_delay_from_text_message():
    from unittest.mock import MagicMock
    from run_generator_eval import _parse_retry_delay

    resp = MagicMock()
    resp.headers = {}
    resp.json.return_value = {
        "error": {
            "message": "Quota exceeded. Please retry in 10h51m23s."
        }
    }
    # 10*3600 + 51*60 + 23 = 39083
    assert _parse_retry_delay(resp) == 39083.0


def test_judge_concurrency_initialization():
    from run_generator_eval import RepoPilotJudgeLLM

    judge = RepoPilotJudgeLLM(
        judge_provider="ollama",
        judge_model="qwen2.5-coder:1.5b",
        generator_model=None,
        judge_cache_path="",
        concurrency=3,
    )
    assert judge.concurrency == 3
    assert judge._sync_sem._value == 3


def test_judge_request_instrumentation():
    from unittest.mock import patch, MagicMock
    from run_generator_eval import RepoPilotJudgeLLM

    judge = RepoPilotJudgeLLM(
        judge_provider="ollama",
        judge_model="qwen2.5-coder:1.5b",
        generator_model=None,
        judge_cache_path="",
    )
    assert judge.http_requests_count == 0
    assert judge.cache_hits_count == 0

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"response": '{"score": 1.0}'}

    mock_client = MagicMock()
    mock_client.__enter__.return_value.post.return_value = mock_resp

    with patch("run_generator_eval.httpx.Client", return_value=mock_client):
        judge.generate("Test prompt")
        assert judge.http_requests_count == 1
        assert judge.cache_hits_count == 0

        # Second call with same prompt hits memory cache
        judge.generate("Test prompt")
        assert judge.http_requests_count == 1
        assert judge.cache_hits_count == 1


def test_skip_judge_run_generator_eval():
    from unittest.mock import patch, MagicMock
    from run_generator_eval import run_generator_eval

    mock_chunks = [
        MagicMock(file_path="app/main.py", start_line=1, end_line=10, content="code", score=0.95)
    ]
    with patch("run_generator_eval._resolve_repository_id", return_value=1), \
         patch("run_generator_eval._fetch_gold_context", return_value=mock_chunks), \
         patch("run_generator_eval._generate_answer", return_value=("In app/main.py:L1-L10 lifespan is set", "ollama")), \
         patch("run_generator_eval._save_cache"):

        payload = run_generator_eval(
            contexts_mode="gold",
            limit=1,
            skip_judge=True,
            output_path="test_skip_judge_run.json",
        )

        assert payload["skip_judge"] is True
        assert "citation_file_hit" in payload["metrics"]
        # Verify no judge metrics were computed
        assert "faithfulness" not in payload["metrics"]
        assert "correctness" not in payload["metrics"]

    if os.path.exists("test_skip_judge_run.json"):
        os.remove("test_skip_judge_run.json")



