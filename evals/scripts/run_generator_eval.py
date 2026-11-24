"""Generator evaluation runner for RepoPilot using DeepEval.

Evaluates the LLM generation step in isolation, separately from the retriever,
using DeepEval metrics with a configurable judge LLM.

Modes
-----
--contexts gold
    Feed golden/ideal chunks (from gt_files, boosted by gt_symbols) to test
    the generator alone — bad scores here mean a generator problem.

--contexts retrieved
    Use live retriever output to test the full RAG pipeline — bad scores here
    relative to gold mode mean a retriever problem.

Usage
-----
    # Quick smoke test (5 questions)
    python evals/scripts/run_generator_eval.py --contexts gold --limit 5

    # Full evaluation
    python evals/scripts/run_generator_eval.py --contexts gold --save-baseline

    # Full pipeline evaluation
    python evals/scripts/run_generator_eval.py --contexts retrieved

    # Estimate judge noise (expensive — 3x judge calls on 8 questions)
    python evals/scripts/run_generator_eval.py --contexts gold --estimate-noise

Environment (backend/.env)
--------------------------
    EVAL_JUDGE_PROVIDER   gemini | ollama   (default: gemini)
    EVAL_JUDGE_MODEL      model name        (default: gemini-2.0-flash-lite)
    GEMINI_API_KEY        required for gemini judge
    OLLAMA_BASE_URL       optional override (default: http://localhost:11434)

Requirements
------------
    pip install -r evals/requirements.txt   (deepeval)
"""

from __future__ import annotations

import argparse
import asyncio
import datetime
import hashlib
import json
import logging
import os
import re
import sys
import threading
import time
from typing import Any

# ---------------------------------------------------------------------------
# Path setup — mirrors run_eval.py so backend modules are importable
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO_ROOT, "backend", ".env"))

# Backend imports (available once backend/ is on sys.path)
import httpx  # noqa: E402
from app.database import SessionLocal  # noqa: E402
from app.llm_service import ask_with_llm  # noqa: E402
from app.models import Repository  # noqa: E402
from app.reranker import rerank  # noqa: E402
from app.search import SearchResult  # noqa: E402
from sqlalchemy import text  # noqa: E402

# ---------------------------------------------------------------------------
# DeepEval imports with graceful fallback so the module stays importable in
# unit tests that mock deepeval away.
# ---------------------------------------------------------------------------
try:
    from deepeval.models import DeepEvalBaseLLM as _DeepEvalBaseLLM

    # Verify it is a real class (not a MagicMock from test mocking)
    type("_ProbeInheritance", (_DeepEvalBaseLLM,), {})

    from deepeval.metrics import (  # noqa: E402
        AnswerRelevancyMetric,
        ContextualRelevancyMetric,
        FaithfulnessMetric,
        GEval,
    )
    from deepeval.test_case import LLMTestCase  # noqa: E402

    try:
        from deepeval.test_case import SingleTurnParams as LLMTestCaseParams  # noqa: E402
    except ImportError:
        from deepeval.test_case import LLMTestCaseParams  # type: ignore[no-redef]

    DEEPEVAL_AVAILABLE = True

except (ImportError, TypeError, Exception):
    DEEPEVAL_AVAILABLE = False
    _DeepEvalBaseLLM = object  # type: ignore[assignment,misc]

    # Dummy stubs so the rest of the module stays parseable
    class FaithfulnessMetric:  # type: ignore[no-redef]
        pass

    class AnswerRelevancyMetric:  # type: ignore[no-redef]
        pass

    class ContextualRelevancyMetric:  # type: ignore[no-redef]
        pass

    class GEval:  # type: ignore[no-redef]
        pass

    class LLMTestCase:  # type: ignore[no-redef]
        pass

    class LLMTestCaseParams:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
GOLDEN_PATH = os.path.join(REPO_ROOT, "evals", "data", "golden_v2.jsonl")
SYMBOL_RANGES_PATH = os.path.join(REPO_ROOT, "evals", "data", "symbol_ranges.json")
RUNS_DIR = os.path.join(REPO_ROOT, "evals", "runs")

PROMPT_VERSION = "v1"
GENERATOR_TEMPERATURE: float = 0.2

# Metric thresholds — provisional, do not tune to make first run pass.
# The primary numbers are per-metric means; thresholds drive pass rate only.
THRESHOLDS: dict[str, float] = {
    "faithfulness": 0.7,
    "answer_relevancy": 0.6,
    "contextual_relevancy": 0.5,
    "correctness": 0.5,
    "refusal_correctness": 0.5,
    "citation_file_hit": 1.0,
    "symbol_hit": 1.0,
    "citation_validity": 1.0,
}

# Fixed 8-question subset for noise estimation — covers all question categories
NOISE_SUBSET_IDS: list[str] = [
    "rp-001",  # locate
    "rp-004",  # locate
    "rp-006",  # explain
    "rp-008",  # explain
    "rp-011",  # cross-file
    "rp-012",  # cross-file
    "rp-015",  # debug
    "rp-016",  # debug
]

# Standard canned refusal message used by retrieval confidence guards
CANNED_REFUSAL_MESSAGE = (
    "I cannot find enough relevant code or documentation in this repository to answer your question. "
    "Please ensure your request is within the scope of this indexed codebase."
)

# Lowercase phrases that signal an explicit refusal or out-of-scope response.
# Used as a cheap first-pass before the GEval refusal metric.
REFUSAL_PHRASES: list[str] = [
    "not found in the codebase",
    "not in the codebase",
    "not used in repopilot",
    "not implemented",
    "doesn't use",
    "does not use",
    "not part of repopilot",
    "outside the scope",
    "out of scope",
    "not relevant to",
    "i cannot",
    "i can't",
    "no information",
    "cannot answer",
    "not present in",
    "not exist in",
    "i don't have",
    "is not used",
    "are not used",
    "not applicable",
    "beyond the scope",
    CANNED_REFUSAL_MESSAGE.lower(),
]


# ---------------------------------------------------------------------------
# Pure utility functions — all are testable without an LLM or database
# ---------------------------------------------------------------------------


def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load JSONL golden dataset from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_symbol_ranges(path: str) -> dict[str, dict[str, Any]]:
    """Load precomputed symbol → line-range mapping (may not exist)."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_answerable(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return examples where answerable=True."""
    return [ex for ex in dataset if ex.get("answerable", True)]


def filter_unanswerable(dataset: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return examples where answerable=False (unanswerable + out_of_scope)."""
    return [ex for ex in dataset if not ex.get("answerable", True)]


def filter_by_category(
    dataset: list[dict[str, Any]], category: str
) -> list[dict[str, Any]]:
    """Return examples matching the given category string."""
    return [ex for ex in dataset if ex.get("category") == category]


def make_answer_cache_key(
    generator_model: str,
    temperature: float,
    system_prompt: str,
    question: str,
    context_chunks: list[str],
    seed: int = 42,
) -> str:
    """Deterministic SHA-256 key for a generated answer.

    Incorporates generator model, temperature, seed, full system prompt text,
    question, and context chunk texts (sorted for order-insensitivity).
    """
    payload = json.dumps(
        {
            "generator_model": generator_model,
            "temperature": float(temperature),
            "seed": int(seed),
            "system_prompt": system_prompt,
            "question": question,
            "contexts": sorted(context_chunks),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_legacy_answer_cache_key(question: str, context_texts: list[str], mode: str) -> str:
    """Legacy SHA-256 key for backward-compatibility with previously cached answers."""
    payload = json.dumps(
        {"question": question, "mode": mode, "contexts": sorted(context_texts)},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_judge_cache_key(judge_model: str, temperature: float, prompt: str) -> str:
    """Deterministic SHA-256 key for a judge LLM response.

    Incorporates judge model name, temperature, and prompt so that
    switching judge models or temperature parameters produces a cache miss.
    """
    payload = json.dumps(
        {
            "judge_model": judge_model,
            "temperature": float(temperature),
            "prompt": prompt,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_refusal(answer: str) -> bool:
    """Quick string-match check: does the answer explicitly refuse / go out-of-scope?

    Used as a cheap first pass before running the GEval refusal metric.
    Returns True when any refusal phrase is found (case-insensitive).
    """
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def aggregate_scores(
    scores_with_ids: list[tuple[str, float, str | None]],
    threshold: float,
    total_evaluated: int | None = None,
) -> dict[str, Any]:
    """Aggregate per-question metric scores into summary statistics.

    Args:
        scores_with_ids: List of (question_id, score, reason) tuples for successful evaluations.
        threshold: Pass/fail cut-off; score >= threshold is a pass.
        total_evaluated: Total attempted items. If omitted, len(scores_with_ids) is used.

    Returns:
        dict with keys: mean, pass_rate, n, total_attempted, threshold.
    """
    total_att = total_evaluated if total_evaluated is not None else len(scores_with_ids)
    if not scores_with_ids:
        return {
            "mean": 0.0,
            "pass_rate": 0.0,
            "n": 0,
            "total_attempted": total_att,
            "threshold": threshold,
        }
    vals = [s for _, s, _ in scores_with_ids]
    passing = sum(1 for v in vals if v >= threshold)
    return {
        "mean": round(sum(vals) / len(vals), 4),
        "pass_rate": round(passing / len(vals), 4),
        "n": len(vals),
        "total_attempted": total_att,
        "threshold": threshold,
    }


def find_worst_n(
    per_question: dict[str, dict[str, Any]],
    metric_name: str,
    n: int = 5,
) -> list[dict[str, Any]]:
    """Return the N questions with the lowest score for the given metric."""
    entries = [
        {
            "id": qid,
            "question": info.get("question", ""),
            "metric": metric_name,
            "score": info["scores"].get(metric_name, {}).get("score", 1.0),
            "reason": info["scores"].get(metric_name, {}).get("reason", ""),
        }
        for qid, info in per_question.items()
        if metric_name in info.get("scores", {})
    ]
    return sorted(entries, key=lambda x: x["score"])[:n]


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    """Format an ASCII/Markdown aligned table (mirrors run_eval.py)."""
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


# ---------------------------------------------------------------------------
# Citation analysis & deterministic metrics (Part A)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Citation extraction & verification (supports old and new [path:Lx-Ly, sym] format)
# ---------------------------------------------------------------------------

NEW_BRACKET_CITATION_REGEX = re.compile(
    r"\[\s*([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9_]+):(?:L|line|lines)?\s*(\d+)\s*[-–—]\s*(?:L|line|lines)?\s*(\d+)(?:\s*,\s*([a-zA-Z0-9_]+))?\s*\]",
    re.IGNORECASE,
)
OLD_CITATION_REGEX = re.compile(
    r"([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9_]+):(?:L|line|lines)?\s*(\d+)\s*[-–—]\s*(?:L|line|lines)?\s*(\d+)",
    re.IGNORECASE,
)


def extract_citations(text: str) -> list[tuple[Any, ...]]:
    """Extract (file_path, start_line, end_line[, symbol]) citations from generated text.

    Supports both:
      - Old format: `path/to/file.py:L14-L58` or `path/to/file.py:14-58`
      - New format: `[path/to/file.py:L14-L58, symbol_name]` or `[path/to/file.py:14-58]`
    """
    citations: list[tuple[Any, ...]] = []
    matched_spans: list[tuple[int, int]] = []

    # 1. First extract bracketed citations: [path:Lx-Ly, symbol] or [path:Lx-Ly]
    for m in NEW_BRACKET_CITATION_REGEX.finditer(text):
        matched_spans.append(m.span())
        path, start_s, end_s, sym = m.groups()
        try:
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            norm_path = path.replace("\\", "/")
            if sym:
                citations.append((norm_path, start, end, sym.strip()))
            else:
                citations.append((norm_path, start, end))
        except ValueError:
            pass

    # 2. Extract unbracketed citations: path:Lx-Ly (skip if inside an existing bracket match)
    for m in OLD_CITATION_REGEX.finditer(text):
        m_start, m_end = m.span()
        if any(b_s <= m_start and m_end <= b_e for b_s, b_e in matched_spans):
            continue
        path, start_s, end_s = m.groups()
        try:
            start, end = int(start_s), int(end_s)
            if start > end:
                start, end = end, start
            citations.append((path.replace("\\", "/"), start, end))
        except ValueError:
            pass

    return citations


def compute_citation_file_hit(answer: str, gt_files: list[str]) -> bool:
    """Return True if the answer mentions any ground truth file path."""
    if not gt_files:
        return False
    answer_norm = answer.replace("\\", "/").lower()
    for f in gt_files:
        f_norm = f.replace("\\", "/").lower()
        base = os.path.basename(f_norm)
        if f_norm in answer_norm or (len(base) > 3 and base in answer_norm):
            return True
    return False


def compute_symbol_hit(answer: str, gt_symbols: list[str]) -> bool | None:
    """Return True if answer mentions any ground truth symbol.

    Returns None if gt_symbols is empty so it can be excluded from evaluation.
    """
    if not gt_symbols:
        return None
    return any(sym in answer for sym in gt_symbols)


def check_citation_validity(
    citations: list[tuple[Any, ...]],
    snippets: list[SearchResult],
) -> tuple[float, list[str]]:
    """Check whether every cited path:range (and optional symbol) overlaps a chunk in context.

    Returns:
        (validity_ratio, list_of_invalid_citations)
    """
    if not citations:
        return 1.0, []

    invalid: list[str] = []
    for item in citations:
        if len(item) == 4:
            path, start, end, symbol = item
        else:
            path, start, end = item[:3]
            symbol = None

        matched = False
        clean_path = path.lower()
        for c in snippets:
            chunk_path = c.file_path.replace("\\", "/").lower()
            path_ok = (
                clean_path in chunk_path
                or chunk_path in clean_path
                or os.path.basename(clean_path) == os.path.basename(chunk_path)
            )
            if path_ok:
                if max(c.start_line, start) <= min(c.end_line, end):
                    if symbol:
                        if symbol in c.content:
                            matched = True
                            break
                    else:
                        matched = True
                        break

        if not matched:
            if symbol:
                invalid.append(f"[{path}:L{start}-L{end}, {symbol}]")
            else:
                invalid.append(f"{path}:L{start}-L{end}")

    invalid_frac = len(invalid) / len(citations)
    return round(1.0 - invalid_frac, 4), invalid


# ---------------------------------------------------------------------------
# Disk cache helpers
# ---------------------------------------------------------------------------


def _load_cache(cache_path: str) -> dict[str, Any]:
    """Load a JSON cache file; return empty dict on any failure."""
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache_path: str, cache: dict[str, Any]) -> None:
    """Write cache to disk.  No-op when cache_path is empty (noise mode)."""
    if not cache_path:
        return
    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _clean_json_text(raw: str) -> str:
    """Extract valid JSON substring, stripping markdown fences and preambles."""
    s = raw.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()

    start_brace = s.find("{")
    start_bracket = s.find("[")
    if start_brace != -1 and start_bracket != -1:
        start = min(start_brace, start_bracket)
    elif start_brace != -1:
        start = start_brace
    elif start_bracket != -1:
        start = start_bracket
    else:
        start = -1

    end_brace = s.rfind("}")
    end_bracket = s.rfind("]")
    end = max(end_brace, end_bracket)

    if start != -1 and end != -1 and end >= start:
        return s[start : end + 1]
    return s


# ---------------------------------------------------------------------------
# Internal rate-limit sentinel
# ---------------------------------------------------------------------------


class _RateLimitError(Exception):
    """Raised on HTTP 429 or 503 to trigger backoff in the retry loop."""

    def __init__(self, message: str, retry_delay: float | None = None) -> None:
        super().__init__(message)
        self.retry_delay = retry_delay


def _parse_retry_delay(resp: httpx.Response) -> float | None:
    """Extract retry delay in seconds from HTTP headers or response body."""
    # 1. Header Retry-After (seconds)
    ra = resp.headers.get("retry-after")
    if ra:
        try:
            return float(ra)
        except ValueError:
            pass

    # 2. Response JSON: details[].retryDelay (e.g. "39083s")
    try:
        data = resp.json()
        error_obj = data.get("error", {})
        details = error_obj.get("details", [])
        for d in details:
            if isinstance(d, dict) and "retryDelay" in d:
                delay_str = str(d["retryDelay"]).rstrip("s")
                return float(delay_str)

        # 3. Text search in error.message (e.g. "Please retry in 10h51m23s")
        msg = error_obj.get("message", "")
        m = re.search(r"retry in\s+(?:(\d+)h)?(?:(\d+)m)?(?:(\d+(?:\.\d+)?)s)?", msg, re.IGNORECASE)
        if m:
            h = float(m.group(1) or 0)
            mins = float(m.group(2) or 0)
            s = float(m.group(3) or 0)
            total = h * 3600.0 + mins * 60.0 + s
            if total > 0:
                return total
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# RepoPilot DeepEval judge LLM wrapper (STEP 3)
# ---------------------------------------------------------------------------


class RepoPilotJudgeLLM(_DeepEvalBaseLLM):  # type: ignore[misc]
    """DeepEval-compatible judge LLM backed by Gemini or Ollama.

    Features
    --------
    - temperature=0 for deterministic judgments.
    - On-disk response cache (SHA-256 of prompt → raw response text).
    - Exponential backoff on HTTP 429 (rate limits).
    - Pacing lock to prevent burst rate limits on Gemini.
    - Up to MAX_RETRIES JSON-schema parse retries per call.
    - Implements both generate() and a_generate() as required by DeepEval.
    - Warns loudly when judge model matches generator model.
    """

    MAX_RETRIES: int = 5
    BASE_BACKOFF_S: float = 3.0
    MIN_INTERVAL_S: float = 1.5

    _pacing_lock = threading.Lock()
    _last_scheduled_call: float = 0.0

    def __init__(
        self,
        judge_provider: str,
        judge_model: str,
        generator_model: str | None,
        judge_cache_path: str,
        api_key: str = "",
        ollama_base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        concurrency: int = 2,
        cache_prefix: str = "",
    ) -> None:
        self.judge_provider = judge_provider.lower()
        self.judge_model = judge_model
        self.ollama_base_url = ollama_base_url
        self.api_key = api_key
        self.temperature = float(temperature)
        self.concurrency = max(1, int(concurrency))
        self.cache_prefix = cache_prefix
        self._sync_sem = threading.Semaphore(self.concurrency)
        self._async_sem_instance: asyncio.Semaphore | None = None
        self._cache_lock = threading.Lock()
        self._cache_path = judge_cache_path
        self._cache: dict[str, str] = _load_cache(judge_cache_path)
        self.http_requests_count: int = 0
        self.cache_hits_count: int = 0
        self.prompt_tokens: int = 0
        self.candidates_tokens: int = 0
        self.thinking_tokens: int = 0
        self.total_tokens: int = 0

        # Warn if the same model is used for generation and judgment.
        if generator_model and self._bare(generator_model) == self._bare(judge_model):
            print(
                f"\n⚠️  WARNING: judge model '{judge_model}' matches generator model "
                f"'{generator_model}'. This introduces self-preference bias and "
                f"invalidates the evaluation. Set EVAL_JUDGE_MODEL to a different model.\n",
                file=sys.stderr,
            )

        # DeepEvalBaseLLM.__init__ may not be callable when mocked in tests;
        # guard accordingly.
        try:
            super().__init__(judge_model)
        except Exception:
            pass

    def _get_async_sem(self) -> asyncio.Semaphore:
        if self._async_sem_instance is None:
            self._async_sem_instance = asyncio.Semaphore(self.concurrency)
        return self._async_sem_instance

    def _cache_key(self, prompt: str) -> str:
        return make_judge_cache_key(self.judge_model, self.temperature, f"{self.cache_prefix}{prompt}")

    # --- DeepEval interface ------------------------------------------------

    def get_model_name(self) -> str:
        return f"{self.judge_provider}/{self.judge_model}"

    def load_model(self) -> "RepoPilotJudgeLLM":  # noqa: F821
        return self

    def generate(self, prompt: str, schema: Any = None) -> Any:
        """Synchronous generation: cache → LLM → parse."""
        key = self._cache_key(prompt)
        with self._cache_lock:
            cached_val = self._cache.get(key)
        if cached_val is not None:
            self.cache_hits_count += 1
            try:
                return self._parse_response(cached_val, schema)
            except Exception:
                with self._cache_lock:
                    self._cache.pop(key, None)

        last_exc: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                raw = self._call_sync_with_retry(prompt)
                parsed = self._parse_response(raw, schema)
                with self._cache_lock:
                    self._cache[key] = raw
                    _save_cache(self._cache_path, self._cache)
                return parsed
            except Exception as exc:
                last_exc = exc
                logger.warning("Judge generate attempt %d/%d failed: %s", attempt + 1, self.MAX_RETRIES, exc)
        raise ValueError(
            f"Judge failed to generate and parse after {self.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    async def a_generate(self, prompt: str, schema: Any = None) -> Any:
        """Asynchronous generation: cache → LLM → parse."""
        key = self._cache_key(prompt)
        with self._cache_lock:
            cached_val = self._cache.get(key)
        if cached_val is not None:
            self.cache_hits_count += 1
            try:
                return self._parse_response(cached_val, schema)
            except Exception:
                with self._cache_lock:
                    self._cache.pop(key, None)

        last_exc: Optional[Exception] = None
        for attempt in range(self.MAX_RETRIES):
            try:
                raw = await self._call_async_with_retry(prompt)
                parsed = self._parse_response(raw, schema)
                with self._cache_lock:
                    self._cache[key] = raw
                    _save_cache(self._cache_path, self._cache)
                return parsed
            except Exception as exc:
                last_exc = exc
                logger.warning("Judge a_generate attempt %d/%d failed: %s", attempt + 1, self.MAX_RETRIES, exc)
        raise ValueError(
            f"Judge failed to a_generate and parse after {self.MAX_RETRIES} attempts: {last_exc}"
        ) from last_exc

    # --- Cache management --------------------------------------------------

    def _get_or_call_sync(self, prompt: str) -> str:
        key = make_judge_cache_key(self.judge_model, self.temperature, prompt)
        if key in self._cache:
            return self._cache[key]
        raw = self._call_sync_with_retry(prompt)
        self._cache[key] = raw
        _save_cache(self._cache_path, self._cache)
        return raw

    async def _get_or_call_async(self, prompt: str) -> str:
        key = make_judge_cache_key(self.judge_model, self.temperature, prompt)
        if key in self._cache:
            return self._cache[key]
        raw = await self._call_async_with_retry(prompt)
        self._cache[key] = raw
        _save_cache(self._cache_path, self._cache)
        return raw

    # --- Response parsing --------------------------------------------------

    def _parse_response(self, raw: str, schema: Any) -> Any:
        """Return schema(**data) when a Pydantic schema is given, else raw text."""
        if schema is None:
            return raw

        clean = _clean_json_text(raw)
        data = None
        try:
            data = json.loads(clean)
        except json.JSONDecodeError:
            # Attempt 1: fix unescaped backslashes (e.g. \_ or \( or \+)
            try:
                clean_escaped = re.sub(r'\\([^"\\/bfnrtuU])', r"\1", clean)
                data = json.loads(clean_escaped)
            except json.JSONDecodeError:
                # Attempt 2: strip trailing commas before closing braces/brackets
                try:
                    clean_trailing = re.sub(r',\s*([}\]])', r"\1", clean_escaped)
                    data = json.loads(clean_trailing)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Judge output is not valid JSON: {exc}. Cleaned string (first 300 chars): {clean[:300]}"
                    ) from exc

        if hasattr(schema, "model_validate"):
            return schema.model_validate(data)
        return schema(**data)

    # --- Retry loops -------------------------------------------------------

    def _call_sync_with_retry(self, prompt: str) -> str:
        last_error = ""
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.judge_provider == "gemini":
                    return self._call_gemini_sync(prompt)
                elif self.judge_provider == "ollama":
                    return self._call_ollama_sync(prompt)
                else:
                    raise ValueError(f"Unknown EVAL_JUDGE_PROVIDER: '{self.judge_provider}'")
            except _RateLimitError as exc:
                if exc.retry_delay is not None and exc.retry_delay > 120.0:
                    logger.error(
                        "Gemini daily quota exhausted (retryDelay=%.0fs / ~%.1fh). Aborting call to avoid hanging.",
                        exc.retry_delay, exc.retry_delay / 3600.0,
                    )
                    raise RuntimeError(
                        f"Gemini daily quota exhausted (retryDelay={exc.retry_delay:.0f}s). Please wait for daily reset."
                    ) from exc

                if exc.retry_delay is not None:
                    backoff = max(self.BASE_BACKOFF_S * (2**attempt), exc.retry_delay)
                else:
                    backoff = self.BASE_BACKOFF_S * (2**attempt)

                logger.warning(
                    "Rate limit on attempt %d/%d — backing off %.1fs (retry_delay=%s): %s",
                    attempt + 1, self.MAX_RETRIES, backoff, exc.retry_delay, exc,
                )
                time.sleep(backoff)
                last_error = str(exc)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Judge call attempt %d failed: %s", attempt + 1, last_error)
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.BASE_BACKOFF_S)
        raise RuntimeError(
            f"Judge call failed after {self.MAX_RETRIES} attempts. Last error: {last_error}"
        )

    async def _call_async_with_retry(self, prompt: str) -> str:
        last_error = ""
        for attempt in range(self.MAX_RETRIES):
            try:
                if self.judge_provider == "gemini":
                    return await self._call_gemini_async(prompt)
                elif self.judge_provider == "ollama":
                    return await self._call_ollama_async(prompt)
                else:
                    raise ValueError(f"Unknown EVAL_JUDGE_PROVIDER: '{self.judge_provider}'")
            except _RateLimitError as exc:
                if exc.retry_delay is not None and exc.retry_delay > 120.0:
                    logger.error(
                        "Gemini daily quota exhausted (retryDelay=%.0fs / ~%.1fh). Aborting call to avoid hanging.",
                        exc.retry_delay, exc.retry_delay / 3600.0,
                    )
                    raise RuntimeError(
                        f"Gemini daily quota exhausted (retryDelay={exc.retry_delay:.0f}s). Please wait for daily reset."
                    ) from exc

                if exc.retry_delay is not None:
                    backoff = max(self.BASE_BACKOFF_S * (2**attempt), exc.retry_delay)
                else:
                    backoff = self.BASE_BACKOFF_S * (2**attempt)

                logger.warning(
                    "Rate limit on attempt %d/%d — backing off %.1fs (retry_delay=%s): %s",
                    attempt + 1, self.MAX_RETRIES, backoff, exc.retry_delay, exc,
                )
                await asyncio.sleep(backoff)
                last_error = str(exc)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Async judge call attempt %d failed: %s", attempt + 1, last_error)
                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.BASE_BACKOFF_S)
        raise RuntimeError(
            f"Async judge call failed after {self.MAX_RETRIES} attempts. Last error: {last_error}"
        )

    # --- Provider-specific LLM calls ---------------------------------------

    def _gemini_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 8192,
            },
        }

    def _record_gemini_usage(self, data: dict[str, Any]) -> None:
        meta = data.get("usageMetadata", {})
        self.prompt_tokens += meta.get("promptTokenCount", 0)
        self.candidates_tokens += meta.get("candidatesTokenCount", 0)
        thinking = 0
        if "candidatesTokensDetails" in meta:
            for d in meta["candidatesTokensDetails"]:
                thinking += d.get("thinkingTokenCount", 0)
        elif "thinkingTokenCount" in meta:
            thinking += meta.get("thinkingTokenCount", 0)
        self.thinking_tokens += thinking
        self.total_tokens += meta.get("totalTokenCount", 0)

    def _extract_gemini_text(self, data: dict[str, Any]) -> str:
        candidates = data.get("candidates", [])
        if not candidates:
            raise ValueError("Empty candidates array in Gemini response")
        cand = candidates[0]
        if cand.get("finishReason") == "MAX_TOKENS":
            raise ValueError("Gemini response truncated due to MAX_TOKENS limit")
        parts = cand.get("content", {}).get("parts", [])
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict))
        return text.strip()

    def _call_gemini_sync(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set — required for Gemini judge")
        with self._sync_sem:
            with self.__class__._pacing_lock:
                now = time.monotonic()
                elapsed = now - self.__class__._last_scheduled_call
                wait_s = max(0.0, self.MIN_INTERVAL_S - elapsed)
                self.__class__._last_scheduled_call = now + wait_s
            if wait_s > 0:
                time.sleep(wait_s)

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.judge_model}:generateContent?key={self.api_key}"
            )
            with httpx.Client(headers={"Connection": "close"}, timeout=60.0) as client:
                self.http_requests_count += 1
                res = client.post(url, json=self._gemini_payload(prompt))
            if res.status_code in (429, 503):
                delay = _parse_retry_delay(res)
                raise _RateLimitError(f"Gemini {res.status_code}: {res.text[:120]}", retry_delay=delay)
            if res.status_code != 200:
                raise RuntimeError(f"Gemini HTTP {res.status_code}: {res.text[:200]}")
            data = res.json()
            self._record_gemini_usage(data)
            return self._extract_gemini_text(data)

    async def _call_gemini_async(self, prompt: str) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY is not set — required for Gemini judge")
        async with self._get_async_sem():
            with self.__class__._pacing_lock:
                now = time.monotonic()
                elapsed = now - self.__class__._last_scheduled_call
                wait_s = max(0.0, self.MIN_INTERVAL_S - elapsed)
                self.__class__._last_scheduled_call = now + wait_s
            if wait_s > 0:
                await asyncio.sleep(wait_s)

            url = (
                f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.judge_model}:generateContent?key={self.api_key}"
            )
            async with httpx.AsyncClient(headers={"Connection": "close"}, timeout=60.0) as client:
                self.http_requests_count += 1
                res = await client.post(url, json=self._gemini_payload(prompt))
            if res.status_code in (429, 503):
                delay = _parse_retry_delay(res)
                raise _RateLimitError(f"Gemini {res.status_code}: {res.text[:120]}", retry_delay=delay)
            if res.status_code != 200:
                raise RuntimeError(f"Gemini HTTP {res.status_code}: {res.text[:200]}")
            data = res.json()
            self._record_gemini_usage(data)
            return self._extract_gemini_text(data)

    def _call_ollama_sync(self, prompt: str) -> str:
        with httpx.Client(timeout=120.0) as client:
            self.http_requests_count += 1
            res = client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.judge_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.0},
                },
            )
        if res.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {res.status_code}: {res.text[:200]}")
        return res.json().get("response", "").strip()

    async def _call_ollama_async(self, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            self.http_requests_count += 1
            res = await client.post(
                f"{self.ollama_base_url}/api/generate",
                json={
                    "model": self.judge_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.0},
                },
            )
        if res.status_code != 200:
            raise RuntimeError(f"Ollama HTTP {res.status_code}: {res.text[:200]}")
        return res.json().get("response", "").strip()

    # --- Helpers -----------------------------------------------------------

    @staticmethod
    def _bare(model: str) -> str:
        """Strip provider prefix: 'ollama/qwen2.5' → 'qwen2.5'."""
        return model.split("/", 1)[-1] if "/" in model else model


# ---------------------------------------------------------------------------
# Gold context fetching (STEP 1 — option 1 with gt_symbols boost)
# ---------------------------------------------------------------------------


def _chunk_overlaps_symbol(
    chunk: SearchResult,
    gt_symbols: list[str],
    symbol_ranges: dict[str, Any],
) -> bool:
    """True when the chunk's line range overlaps any ground-truth symbol range."""
    for sym in gt_symbols:
        key = f"{chunk.file_path}::{sym}"
        if key in symbol_ranges:
            s_start: int = symbol_ranges[key]["start_line"]
            s_end: int = symbol_ranges[key]["end_line"]
            if max(chunk.start_line, s_start) <= min(chunk.end_line, s_end):
                return True
    return False


def _fetch_gold_context(
    repository_id: int,
    example: dict[str, Any],
    top_k: int,
    symbol_ranges: dict[str, Any],
) -> list[SearchResult]:
    """Fetch chunks from gt_files only, boosting symbol-matching chunks to the top.

    Strategy
    --------
    1. Query ALL chunks whose file_path is in gt_files.
    2. Run rerank() on them to get cross-encoder scores (same scorer as production).
    3. Split into two ordered lists: symbol-matching chunks and the rest.
    4. Concatenate (symbol hits first) and truncate to top_k.

    This keeps the prompt size identical to production while ensuring the
    generator receives the most informative context available.
    """
    gt_files: list[str] = example.get("gt_files", [])
    gt_symbols: list[str] = example.get("gt_symbols", [])
    question: str = example["question"]

    if not gt_files:
        return []

    # Build parameterised IN clause for arbitrary-length list
    placeholders = ", ".join(f":fp{i}" for i in range(len(gt_files)))
    sql = text(
        f"""
        SELECT id, repository_id, file_path, start_line, end_line, content, language
        FROM code_chunks
        WHERE repository_id = :repo_id
          AND file_path IN ({placeholders})
        """
    )
    params: dict[str, Any] = {"repo_id": repository_id}
    for i, fp in enumerate(gt_files):
        params[f"fp{i}"] = fp

    with SessionLocal() as session:
        rows = session.execute(sql, params).fetchall()

    if not rows:
        logger.warning(
            "Gold context: no chunks found for %s — gt_files=%s",
            example["id"], gt_files,
        )
        return []

    candidates = [
        SearchResult(
            chunk_id=row.id,
            repository_id=row.repository_id,
            file_path=row.file_path,
            start_line=row.start_line,
            end_line=row.end_line,
            content=row.content,
            language=row.language,
            score=0.0,
        )
        for row in rows
    ]

    # Apply the production reranker within the gold-file corpus
    reranked = rerank(question, candidates, top_k=len(candidates))

    # Symbol-matching chunks float to the top; fill remainder by reranker score
    symbol_hits: list[SearchResult] = []
    rest: list[SearchResult] = []
    for chunk in reranked:
        if gt_symbols and _chunk_overlaps_symbol(chunk, gt_symbols, symbol_ranges):
            symbol_hits.append(chunk)
        else:
            rest.append(chunk)

    return (symbol_hits + rest)[:top_k]


# ---------------------------------------------------------------------------
# Retrieved context fetching
# ---------------------------------------------------------------------------


def _fetch_retrieved_context(
    repository_id: int,
    question: str,
    top_k: int,
) -> list[SearchResult]:
    """Run the live hybrid retrieval pipeline and return top_k reranked chunks."""
    from app.search import hybrid_search  # noqa: PLC0415

    with SessionLocal() as session:
        candidates = hybrid_search(session, repository_id, question, limit=top_k * 3)
    return rerank(question, candidates, top_k=top_k)


# ---------------------------------------------------------------------------
# Answer generation with disk cache
# ---------------------------------------------------------------------------


def _generate_answer(
    question: str,
    snippets: list[SearchResult],
    mode: str,
    answer_cache: dict[str, Any],
    answer_cache_path: str,
    generator_model: str | None = None,
    temperature: float = 0.0,
    seed: int = 42,
    system_prompt: str | None = None,
) -> tuple[str, str]:
    """Generate an answer (Ollama with seed=42, temp=0 → Gemini fallback) with on-disk caching.

    Returns:
        (answer_text, model_identifier)
    """
    if system_prompt is None:
        try:
            from app.llm_service import _SYSTEM_INSTRUCTION  # noqa: PLC0415
            system_prompt = _SYSTEM_INSTRUCTION
        except Exception:
            system_prompt = ""

    gen_model = generator_model or os.getenv("OLLAMA_MODEL", "") or os.getenv("GEMINI_MODEL", "qwen2.5-coder:1.5b")

    context_texts = [
        f"File: {s.file_path} (lines {s.start_line}-{s.end_line})\n{s.content}"
        for s in snippets
    ]
    cache_key = make_answer_cache_key(
        generator_model=gen_model,
        temperature=temperature,
        seed=seed,
        system_prompt=system_prompt,
        question=question,
        context_chunks=context_texts,
    )

    if cache_key in answer_cache:
        cached = answer_cache[cache_key]
        return cached["answer"], cached["model_used"]

    # Fallback to legacy key to reuse cached answers from earlier runs without regeneration
    legacy_key = make_legacy_answer_cache_key(question, context_texts, mode)
    if legacy_key in answer_cache:
        cached = answer_cache[legacy_key]
        answer_cache[cache_key] = cached
        _save_cache(answer_cache_path, answer_cache)
        return cached["answer"], cached["model_used"]

    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        prompt_blocks = [
            f"File: {s.file_path} (lines {s.start_line}-{s.end_line})\n{s.content}"
            for s in snippets
        ]
        context_str = "\n\n---\n\n".join(prompt_blocks)
        full_prompt = (
            f"{system_prompt}\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": gen_model,
                    "prompt": full_prompt,
                    "stream": False,
                    "options": {
                        "temperature": float(temperature),
                        "seed": int(seed),
                        "num_predict": 1024,
                    },
                },
            )
            if resp.status_code == 200:
                answer = resp.json().get("response", "").strip()
                model_used = f"ollama/{gen_model}"
            else:
                answer, model_used = asyncio.run(ask_with_llm(question, snippets))
    except Exception:
        try:
            answer, model_used = asyncio.run(ask_with_llm(question, snippets))
        except RuntimeError as exc:
            print(
                f"\n❌  LLM generation failed: {exc}\n"
                "    Set GEMINI_API_KEY in backend/.env or start Ollama.\n",
                file=sys.stderr,
            )
            sys.exit(1)

    answer_cache[cache_key] = {
        "answer": answer,
        "model_used": model_used,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    _save_cache(answer_cache_path, answer_cache)
    return answer, model_used


# ---------------------------------------------------------------------------
# Repository resolution
# ---------------------------------------------------------------------------


def _resolve_repository_id(repo_id: int | None) -> int:
    """Auto-resolve the target repository ID if not explicitly provided."""
    with SessionLocal() as session:
        if repo_id is not None:
            repo = session.query(Repository).filter(Repository.id == repo_id).first()
            if repo is None:
                raise ValueError(f"Repository ID {repo_id} not found in database.")
            return repo.id

        # Prefer the full RepoPilot repository ('repopilot' contains code + docs)
        repo = (
            session.query(Repository)
            .filter(Repository.full_name == "repopilot")
            .first()
        )
        if repo is None:
            repo = (
                session.query(Repository)
                .filter(Repository.full_name.ilike("%repopilot%"))
                .first()
            )
        if repo is None:
            repo = (
                session.query(Repository)
                .filter(Repository.status == "completed")
                .first()
            )
        if repo is None:
            raise ValueError(
                "No indexed repository found in the database. "
                "Run the indexer first, or pass --repo-id explicitly."
            )
        return repo.id


# ---------------------------------------------------------------------------
# Metric construction (STEP 2)
# ---------------------------------------------------------------------------


def _build_metrics(
    judge: RepoPilotJudgeLLM,
    contexts_mode: str,
    selected_metrics: list[str] | None = None,
) -> dict[str, Any]:
    """Instantiate DeepEval metrics for answerable questions.

    ContextualRelevancy is opt-in (default off) via --metrics to conserve API requests.
    """
    metrics: dict[str, Any] = {
        "faithfulness": FaithfulnessMetric(
            threshold=THRESHOLDS["faithfulness"],
            model=judge,
            include_reason=True,
        ),
        "answer_relevancy": AnswerRelevancyMetric(
            threshold=THRESHOLDS["answer_relevancy"],
            model=judge,
            include_reason=True,
        ),
        "correctness": GEval(
            name="Correctness",
            evaluation_steps=[
                "Check whether the actual output correctly identifies the files, "
                "functions, and line numbers mentioned in the expected output.",
                "Check whether the actual output's explanation is factually consistent "
                "with the expected output.",
                "Penalise false technical claims: hallucinated file paths, function "
                "names, or behaviour not supported by the expected output.",
                "Minor rephrasing and partial answers are acceptable; focus on factual "
                "accuracy and coverage of the key technical points.",
            ],
            evaluation_params=[
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=THRESHOLDS["correctness"],
            model=judge,
        ),
    }

    # ContextualRelevancy is opt-in (default off)
    if selected_metrics and "contextual_relevancy" in selected_metrics:
        if contexts_mode == "retrieved":
            metrics["contextual_relevancy"] = ContextualRelevancyMetric(
                threshold=THRESHOLDS["contextual_relevancy"],
                model=judge,
                include_reason=True,
            )

    if selected_metrics is not None:
        metrics = {k: v for k, v in metrics.items() if k in selected_metrics}

    return metrics


def _build_refusal_metric(judge: RepoPilotJudgeLLM) -> Any:
    """GEval metric for checking correct refusal on unanswerable questions."""
    return GEval(
        name="Refusal Correctness",
        evaluation_steps=[
            "Check whether the actual output correctly refuses to answer or states "
            "that the concept/feature is NOT present in the RepoPilot codebase.",
            "Verify that the actual output does NOT hallucinate specific files, "
            "functions, or implementations that do not exist.",
            "A correct refusal must be explicit: it should state the feature is "
            "absent, not applicable to RepoPilot, or outside the scope of a "
            "codebase assistant.",
            "Vague or evasive responses that neither confirm nor deny should score "
            "lower than explicit refusals.",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        threshold=THRESHOLDS["refusal_correctness"],
        model=judge,
    )


def _measure_metric(metric: Any, test_case: Any) -> tuple[float | None, str | None]:
    """Call metric.measure(test_case) and return (score, reason).

    Catches metric-level exceptions and returns (None, error_msg) so that
    a failed judge call is excluded from metric averages rather than scored as 0.0.
    """
    try:
        metric.measure(test_case)
        return float(metric.score), getattr(metric, "reason", None)
    except Exception as exc:
        metric_name = getattr(metric, "__class__", type(metric)).__name__
        logger.error("Metric %s failed: %s", metric_name, exc)
        return None, f"ERROR: {exc}"


# ---------------------------------------------------------------------------
# Main evaluation function (STEPS 1-4)
# ---------------------------------------------------------------------------


def run_generator_eval(  # noqa: PLR0912, PLR0914, PLR0915
    contexts_mode: str = "gold",
    top_k: int = 5,
    limit: int | None = None,
    save_baseline: bool = False,
    estimate_noise: bool = False,
    repo_id: int | None = None,
    dataset_path: str = GOLDEN_PATH,
    output_path: str | None = None,
    judge_concurrency: int = 2,
    skip_judge: bool = False,
    metrics_list: list[str] | None = None,
) -> dict[str, Any]:
    """Run the generator evaluation and return the full run payload.

    Args:
        contexts_mode:     'gold' or 'retrieved'.
        top_k:             Number of context chunks per question (same as production).
        limit:             Cap answerable questions evaluated (None = all).
        save_baseline:     If True, also write a baseline file for this mode.
        estimate_noise:    If True, run the judge 3× on a fixed 8-question subset.
        repo_id:           Database repository ID (auto-resolved if None).
        dataset_path:      Path to the golden JSONL file.
        output_path:       Custom output JSON path (auto-generated if None).
        judge_concurrency: Maximum concurrent calls to the judge model (default: 2).
        skip_judge:        If True, skip all judge calls and compute deterministic metrics only.

    Returns:
        The full run payload dict (also written to disk).
    """
    if not skip_judge and not DEEPEVAL_AVAILABLE:
        raise RuntimeError(
            "deepeval is not installed.\n"
            "Run:  cd backend && .venv\\Scripts\\pip install -r ..\\evals\\requirements.txt"
        )

    # --- Read configuration from environment ------------------------------
    judge_provider = os.getenv("EVAL_JUDGE_PROVIDER", "gemini").strip().lower()
    judge_model = os.getenv(
        "EVAL_JUDGE_MODEL",
        "gemini-3.1-flash-lite" if judge_provider == "gemini" else "llama3.2",
    ).strip()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()

    # Best-effort: determine which model the generator will use.
    # The actual model is captured from ask_with_llm() during generation.
    generator_model: str = (
        os.getenv("OLLAMA_MODEL", "") or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    )

    os.makedirs(RUNS_DIR, exist_ok=True)
    answer_cache_path = os.path.join(RUNS_DIR, "generator_answer_cache.json")
    judge_cache_path = os.path.join(RUNS_DIR, "generator_judge_cache.json")
    answer_cache: dict[str, Any] = _load_cache(answer_cache_path)

    # --- Dataset ----------------------------------------------------------
    all_examples = load_dataset(dataset_path)
    answerable = filter_answerable(all_examples)
    unanswerable_all = filter_unanswerable(all_examples)
    unanswerable_only = filter_by_category(unanswerable_all, "unanswerable")
    out_of_scope_only = filter_by_category(unanswerable_all, "out_of_scope")

    if limit is not None:
        answerable = answerable[:limit]

    symbol_ranges = load_symbol_ranges(SYMBOL_RANGES_PATH)

    # --- Repository -------------------------------------------------------
    target_repo_id = _resolve_repository_id(repo_id)

    # --- Judge ------------------------------------------------------------
    if skip_judge:
        judge = None
        judge_model_name = "none (skipped --skip-judge)"
        metrics_map = {}
        refusal_metric = None
    else:
        judge = RepoPilotJudgeLLM(
            judge_provider=judge_provider,
            judge_model=judge_model,
            generator_model=generator_model,
            judge_cache_path=judge_cache_path,
            api_key=api_key,
            ollama_base_url=ollama_url,
            concurrency=judge_concurrency,
        )
        judge_model_name = judge.get_model_name()
        metrics_map = _build_metrics(judge, contexts_mode, selected_metrics=metrics_list)
        refusal_metric = _build_refusal_metric(judge)

    print("\n" + "=" * 72)
    print("RepoPilot Generator Evaluation  (DeepEval)")
    print(f"  Contexts mode  : {contexts_mode}")
    print(f"  Judge          : {judge_model_name}")
    print(f"  Generator      : {generator_model}")
    print(f"  Prompt version : {PROMPT_VERSION}")
    print(f"  Answerable     : {len(answerable)} questions (limit={limit})")
    print(f"  Unanswerable   : {len(unanswerable_only)} | Out-of-scope: {len(out_of_scope_only)}")
    print(f"  Top-k          : {top_k}")
    print(f"  Skip judge     : {skip_judge}")
    print("=" * 72 + "\n")

    # Accumulator: metric_name → [(qid, score, reason)]
    metric_scores: dict[str, list[tuple[str, float, str | None]]] = {
        m: [] for m in metrics_map
    }
    citation_file_hits: list[tuple[str, float, str | None]] = []
    symbol_hits: list[tuple[str, float, str | None]] = []
    citation_validities: list[tuple[str, float, str | None]] = []
    all_invalid_citations: list[tuple[str, str]] = []

    per_question: dict[str, dict[str, Any]] = {}
    # Track actual model used (may change if Ollama falls back to Gemini)
    last_model_used: str = generator_model

    # ---- Evaluate answerable questions -----------------------------------
    for idx, ex in enumerate(answerable, start=1):
        qid: str = ex["id"]
        cat: str = ex.get("category", "general")
        question: str = ex["question"]
        ideal: str = ex.get("ideal_answer", "")
        gt_files = ex.get("gt_files", [])
        gt_symbols = ex.get("gt_symbols", [])

        print(f"[{idx:02d}/{len(answerable):02d}] ({cat.upper()}) {qid}: {question[:60]}...")

        # 1. Fetch context
        if contexts_mode == "gold":
            snippets = _fetch_gold_context(target_repo_id, ex, top_k, symbol_ranges)
        else:
            snippets = _fetch_retrieved_context(target_repo_id, question, top_k)

        top_rerank_score = round(snippets[0].score, 4) if snippets else 0.0

        context_texts = [
            f"File: {s.file_path} (lines {s.start_line}-{s.end_line})\n{s.content}"
            for s in snippets
        ]

        # 2. Generate answer (cached)
        answer, last_model_used = _generate_answer(
            question, snippets, contexts_mode, answer_cache, answer_cache_path
        )
        print(f"         model={last_model_used} | chunks={len(snippets)} | top_score={top_rerank_score}")

        # 3. Deterministic citation metrics (Part A)
        file_hit = compute_citation_file_hit(answer, gt_files)
        sym_hit = compute_symbol_hit(answer, gt_symbols)
        citations = extract_citations(answer)
        validity, invalid_cites = check_citation_validity(citations, snippets)

        citation_file_hits.append((qid, 1.0 if file_hit else 0.0, None))
        if sym_hit is not None:
            symbol_hits.append((qid, 1.0 if sym_hit else 0.0, None))
        citation_validities.append((
            qid,
            validity,
            ", ".join(invalid_cites) if invalid_cites else None,
        ))
        for ic in invalid_cites:
            all_invalid_citations.append((qid, ic))

        # 4. Measure each metric (skipped if --skip-judge)
        q_scores: dict[str, dict[str, Any]] = {}
        if not skip_judge and DEEPEVAL_AVAILABLE:
            if answer.strip() == CANNED_REFUSAL_MESSAGE.strip():
                # Canned guard reply on an answerable question is deterministically an over-refusal
                for metric_name in metrics_map:
                    metric_scores[metric_name].append((qid, 0.0, "Canned guard refusal on answerable question"))
                    q_scores[metric_name] = {
                        "score": 0.0,
                        "reason": "Canned guard refusal on answerable question",
                        "passed": False,
                    }
                    print(f"         {metric_name}: 0.000 (deterministic canned refusal)")
            else:
                test_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    retrieval_context=context_texts,
                    expected_output=ideal,
                )
                for metric_name, metric in metrics_map.items():
                    score, reason = _measure_metric(metric, test_case)
                    if score is not None:
                        metric_scores[metric_name].append((qid, score, reason))
                        q_scores[metric_name] = {
                            "score": round(score, 4),
                            "reason": reason,
                            "passed": score >= THRESHOLDS[metric_name],
                        }
                        print(f"         {metric_name}: {score:.3f}")
                    else:
                        q_scores[metric_name] = {
                            "score": None,
                            "reason": reason,
                            "passed": None,
                            "error": True,
                        }
                        print(f"         {metric_name}: FAILED (excluded from average)")

        q_scores["citation_file_hit"] = {
            "score": 1.0 if file_hit else 0.0,
            "passed": file_hit,
        }
        if sym_hit is not None:
            q_scores["symbol_hit"] = {
                "score": 1.0 if sym_hit else 0.0,
                "passed": sym_hit,
            }
        q_scores["citation_validity"] = {
            "score": validity,
            "passed": validity >= 1.0,
            "invalid_citations": invalid_cites,
        }

        print(
            f"         cites: file_hit={file_hit} | "
            f"sym_hit={sym_hit} | validity={validity:.2f} ({len(invalid_cites)} invalid)"
        )

        per_question[qid] = {
            "id": qid,
            "question": question,
            "category": cat,
            "scores": q_scores,
            "actual_output": answer,
            "retrieval_context": context_texts,
            "expected_output": ideal,
            "model_used": last_model_used,
            "top_rerank_score": top_rerank_score,
            "citations_found": [
                f"[{c[0]}:L{c[1]}-L{c[2]}, {c[3]}]" if len(c) == 4 else f"{c[0]}:L{c[1]}-L{c[2]}"
                for c in citations
            ],
            "invalid_citations": invalid_cites,
        }

    # ---- Evaluate unanswerable / out-of-scope questions ------------------
    unanswerable_results: list[dict[str, Any]] = []
    out_of_scope_results: list[dict[str, Any]] = []

    for group_examples, results_list in [
        (unanswerable_only, unanswerable_results),
        (out_of_scope_only, out_of_scope_results),
    ]:
        for ex in group_examples:
            qid = ex["id"]
            cat = ex.get("category", "unanswerable")
            question = ex["question"]
            print(f"\n[NON-ANSWERABLE] ({cat.upper()}) {qid}: {question[:60]}...")

            # In retrieved mode, fetch real context (may confuse the LLM — intentional)
            snippets_unans: list[SearchResult] = []
            if contexts_mode == "retrieved":
                snippets_unans = _fetch_retrieved_context(target_repo_id, question, top_k)

            top_rerank_score_unans = (
                round(snippets_unans[0].score, 4) if snippets_unans else 0.0
            )

            answer, last_model_used = _generate_answer(
                question, snippets_unans, contexts_mode, answer_cache, answer_cache_path
            )

            # Cheap first pass: string match
            string_pass = is_refusal(answer)
            is_canned_guard = (answer.strip() == CANNED_REFUSAL_MESSAGE.strip())

            if is_canned_guard:
                # Canned guard reply scored deterministically with zero judge calls
                refusal_score_val = 1.0
                refusal_reason = "Deterministic exact match on canned guard refusal message."
                judge_pass = True
                string_pass = True
                overall_pass = True
            elif not skip_judge and DEEPEVAL_AVAILABLE and refusal_metric is not None:
                # GEval refusal metric (runs regardless of string pass)
                refusal_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    retrieval_context=[],
                    expected_output="",
                )
                refusal_score, refusal_reason = _measure_metric(refusal_metric, refusal_case)
                if refusal_score is not None:
                    judge_pass = refusal_score >= THRESHOLDS["refusal_correctness"]
                    overall_pass = judge_pass
                    refusal_score_val: float | None = round(refusal_score, 4)
                else:
                    judge_pass = False
                    overall_pass = False
                    refusal_score_val = None

                # Overall pass: judge_pass only (Part B requirement)
                if string_pass != judge_pass:
                    print(
                        f"⚠️  DIAGNOSTIC MISMATCH on {qid}: string_pass={string_pass} != judge_pass={judge_pass}"
                    )
            else:
                refusal_score_val = None
                refusal_reason = None
                judge_pass = None
                overall_pass = string_pass

            result: dict[str, Any] = {
                "id": qid,
                "category": cat,
                "question": question,
                "actual_output": answer,
                "model_used": last_model_used,
                "top_rerank_score": top_rerank_score_unans,
                "string_match_pass": string_pass,
                "refusal_score": refusal_score_val,
                "refusal_reason": refusal_reason,
                "judge_pass": judge_pass,
                "overall_pass": overall_pass,
            }
            results_list.append(result)
            print(
                f"         string_pass={string_pass} | "
                f"judge_pass={judge_pass} | overall_pass={overall_pass} | "
                f"top_score={top_rerank_score_unans}"
            )

    # ---- Aggregate metrics -----------------------------------------------
    total_ans = len(answerable)
    aggregated: dict[str, dict[str, Any]] = {
        metric_name: aggregate_scores(scores, THRESHOLDS[metric_name], total_evaluated=total_ans)
        for metric_name, scores in metric_scores.items()
    }
    aggregated["citation_file_hit"] = aggregate_scores(
        citation_file_hits, THRESHOLDS["citation_file_hit"], total_evaluated=total_ans
    )
    if symbol_hits:
        aggregated["symbol_hit"] = aggregate_scores(
            symbol_hits, THRESHOLDS["symbol_hit"], total_evaluated=total_ans
        )
    aggregated["citation_validity"] = aggregate_scores(
        citation_validities, THRESHOLDS["citation_validity"], total_evaluated=total_ans
    )

    # Over-refusal metric on answerable questions (Part C)
    over_refused_qids = [
        qid
        for qid, info in per_question.items()
        if is_refusal(info["actual_output"])
    ]
    over_refusal_rate = (
        len(over_refused_qids) / len(per_question) if per_question else 0.0
    )

    unanswerable_pass_rate = (
        sum(1 for r in unanswerable_results if r["overall_pass"]) / len(unanswerable_results)
        if unanswerable_results
        else 0.0
    )
    out_of_scope_pass_rate = (
        sum(1 for r in out_of_scope_results if r["overall_pass"]) / len(out_of_scope_results)
        if out_of_scope_results
        else 0.0
    )

    # Worst 5 questions across ALL judge metrics combined
    all_scored: list[dict[str, Any]] = []
    for metric_name in metrics_map:
        for qid, info in per_question.items():
            if metric_name in info.get("scores", {}):
                sc = info["scores"][metric_name]["score"]
                if sc is not None:
                    all_scored.append({
                        "id": qid,
                        "question": info["question"],
                        "metric": metric_name,
                        "score": sc,
                        "reason": info["scores"][metric_name].get("reason", ""),
                    })
    worst_5 = sorted(all_scored, key=lambda x: x["score"])[:5]

    # ---- Print summary table --------------------------------------------
    mode_refusal_label = "empty-context" if contexts_mode == "gold" else "retrieved"

    print("\n" + "=" * 72)
    print(f"GENERATOR EVALUATION SUMMARY  [{contexts_mode.upper()} mode]")
    print(f"Answerable questions evaluated: {len(answerable)}")
    print("=" * 72)

    table_rows = [
        [
            metric_name,
            f"{agg['mean']:.3f}",
            f"{agg['pass_rate']:.1%}",
            f"{agg['n']}" if agg.get("total_attempted", agg['n']) == agg['n'] else f"{agg['n']}/{agg['total_attempted']}",
            str(agg["threshold"]),
        ]
        for metric_name, agg in aggregated.items()
    ]
    print(format_table(["Metric", "Mean", "Pass Rate", "N (Evaluated)", "Threshold"], table_rows))

    print(f"\n  Citation Integrity:")
    print(
        f"    Total invalid citations: {len(all_invalid_citations)} across all questions"
    )
    if all_invalid_citations:
        for qid, c in all_invalid_citations[:5]:
            print(f"      - [{qid}] {c}")
        if len(all_invalid_citations) > 5:
            print(f"      ... and {len(all_invalid_citations) - 5} more")

    print(
        f"\n  Over-refusal rate on answerable Qs: {over_refusal_rate:.1%} "
        f"({len(over_refused_qids)}/{len(per_question)})"
    )
    if over_refused_qids:
        print(f"    Flagged answerable question IDs: {over_refused_qids}")
    else:
        print("    No over-refusals detected on answerable questions.")

    print(
        f"\n  Refusal metrics [{mode_refusal_label} mode - strictly excluded from averages]:"
    )
    print(
        f"    Unanswerable ({len(unanswerable_results)} Qs):  "
        f"pass rate = {unanswerable_pass_rate:.1%}"
    )
    print(
        f"    Out-of-scope  ({len(out_of_scope_results)} Qs):  "
        f"pass rate = {out_of_scope_pass_rate:.1%}"
    )

    if worst_5:
        print("\n  Worst 5 questions (across judge metrics):")
        for entry in worst_5:
            print(
                f"    [{entry['id']}] {entry['metric']:22s}  "
                f"score={entry['score']:.3f}  "
                f"{entry['question'][:55]}..."
            )
            if entry.get("reason"):
                print(f"       reason: {str(entry['reason'])[:130]}")

    if skip_judge:
        print("\n  [NOTE] Judge evaluation was SKIPPED (--skip-judge). Answers generated and cached.")

    # ---- Rerank Score Statistics (Part F) --------------------------------
    ans_scores = [
        info["top_rerank_score"]
        for info in per_question.values()
        if "top_rerank_score" in info
    ]
    if ans_scores:
        ans_scores_sorted = sorted(ans_scores)
        min_score = ans_scores_sorted[0]
        med_score = ans_scores_sorted[len(ans_scores_sorted) // 2]
        lowest_3 = ans_scores_sorted[:3]
        print(f"\n  Top Rerank Score Statistics (Answerable, N={len(ans_scores)}):")
        print(f"    Min: {min_score:.4f} | Median: {med_score:.4f} | 3 Lowest: {[round(s, 4) for s in lowest_3]}")

    # ---- Rerank Score Comparison Table (Part F) -------------------------
    if contexts_mode == "retrieved" or skip_judge:
        print("\n" + "=" * 72)
        print("RERANK SCORE DISTRIBUTION TABLE (Part F)")
        print("Comparing top rerank scores for answerable vs refusal questions")
        print("=" * 72)
        rerank_sample_ids = ["rp-001", "rp-003", "rp-006", "rp-011", "rp-016"]
        rerank_rows: list[list[str]] = []
        for qid in rerank_sample_ids:
            if qid in per_question:
                info = per_question[qid]
                score_val = info.get("top_rerank_score", 0.0)
                rerank_rows.append([
                    qid,
                    info.get("category", "answerable"),
                    "answerable",
                    f"{score_val:.4f}",
                    info.get("question", "")[:45] + "...",
                ])
        for r in unanswerable_results + out_of_scope_results:
            score_val = r.get("top_rerank_score", 0.0)
            rerank_rows.append([
                r["id"],
                r.get("category", "unanswerable"),
                "refusal",
                f"{score_val:.4f}",
                r.get("question", "")[:45] + "...",
            ])
        print(
            format_table(
                ["ID", "Category", "Type", "Top Rerank Score", "Question"],
                rerank_rows,
            )
        )

    print("=" * 72)

    # Compute dataset SHA-256
    try:
        with open(dataset_path, "rb") as _f_ds:
            dataset_sha256 = hashlib.sha256(_f_ds.read()).hexdigest()
    except Exception:
        dataset_sha256 = ""

    # ---- Build run payload (STEP 4) -------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_payload: dict[str, Any] = {
        "version": "v1.1",
        "timestamp": timestamp,
        "prompt_version": PROMPT_VERSION,
        "contexts_mode": contexts_mode,
        "refusal_mode_label": mode_refusal_label,
        "skip_judge": skip_judge,
        "generator_model": last_model_used,
        "judge_model": judge.get_model_name() if judge else "skipped (--skip-judge)",
        "dataset_version": os.path.basename(dataset_path),
        "dataset_sha256": dataset_sha256,
        "total_examples": len(all_examples),
        "answerable_count": len(filter_answerable(all_examples)),
        "unanswerable_count": len(unanswerable_only),
        "out_of_scope_count": len(out_of_scope_only),
        "evaluated_answerable": len(answerable),
        "top_k": top_k,
        "metrics": aggregated,
        "over_refusal_rate": round(over_refusal_rate, 4),
        "over_refused_qids": over_refused_qids,
        "invalid_citations_count": len(all_invalid_citations),
        "invalid_citations": all_invalid_citations,
        "unanswerable_pass_rate": round(unanswerable_pass_rate, 4),
        "out_of_scope_pass_rate": round(out_of_scope_pass_rate, 4),
        "worst_5": worst_5,
        "query_details": list(per_question.values()),
        "unanswerable_details": unanswerable_results,
        "out_of_scope_details": out_of_scope_results,
    }

    if judge:
        print(f"\n  Judge Activity & Token Usage:")
        print(f"    HTTP Requests     : {judge.http_requests_count}")
        print(f"    Cache Hits        : {judge.cache_hits_count}")
        if judge.judge_provider == "gemini":
            print(f"    Prompt Tokens     : {judge.prompt_tokens}")
            print(f"    Candidates Tokens : {judge.candidates_tokens}")
            print(f"    Thinking Tokens   : {judge.thinking_tokens}")
            print(f"    Total Tokens      : {judge.total_tokens}")
        run_payload["judge_usage"] = {
            "http_requests": judge.http_requests_count,
            "cache_hits": judge.cache_hits_count,
            "prompt_tokens": judge.prompt_tokens,
            "candidates_tokens": judge.candidates_tokens,
            "thinking_tokens": judge.thinking_tokens,
            "total_tokens": judge.total_tokens,
        }

    # ---- Save artifacts --------------------------------------------------
    run_file = output_path or os.path.join(
        RUNS_DIR, f"generator_{contexts_mode}_{timestamp}.json"
    )
    with open(run_file, "w", encoding="utf-8") as f:
        json.dump(run_payload, f, indent=2)
    print(f"\nRun saved → {run_file}")

    if save_baseline:
        baseline_file = os.path.join(
            RUNS_DIR, f"generator_baseline_{contexts_mode}.json"
        )
        with open(baseline_file, "w", encoding="utf-8") as f:
            json.dump(run_payload, f, indent=2)
        print(f"Baseline saved → {baseline_file}")

    # ---- Noise estimation (STEP 4) --------------------------------------
    if estimate_noise:
        _run_noise_estimation(
            contexts_mode=contexts_mode,
            top_k=top_k,
            target_repo_id=target_repo_id,
            all_examples=all_examples,
            symbol_ranges=symbol_ranges,
            judge_provider=judge_provider,
            judge_model=judge_model,
            api_key=api_key,
            ollama_url=ollama_url,
            generator_model=last_model_used,
            metrics_map=metrics_map,
            answer_cache=answer_cache,
            answer_cache_path=answer_cache_path,
            judge_concurrency=judge_concurrency,
        )

    return run_payload


# ---------------------------------------------------------------------------
# Noise estimation (STEP 4)
# ---------------------------------------------------------------------------


def _run_noise_estimation(  # noqa: PLR0913
    contexts_mode: str,
    top_k: int,
    target_repo_id: int,
    all_examples: list[dict[str, Any]],
    symbol_ranges: dict[str, Any],
    judge_provider: str,
    judge_model: str,
    api_key: str,
    ollama_url: str,
    generator_model: str,
    metrics_map: dict[str, Any],
    answer_cache: dict[str, Any],
    answer_cache_path: str,
    n_runs: int = 3,
    judge_concurrency: int = 2,
) -> None:
    """Run noise estimation on a fixed 8-question subset.

    Measures:
    1. Judge noise: Evaluates identical cached answers 3x with an uncached judge.
    2. Generator noise: If generator temperature > 0, regenerates answers 3x
       bypassing answer cache and evaluates them with a fixed judge.

    Results are saved to evals/runs/generator_noise.json and read by
    compare_generator.py to suppress changes within the noise floor.
    """
    import math  # noqa: PLC0415
    import statistics as _stats  # noqa: PLC0415

    noise_path = os.path.join(RUNS_DIR, "generator_noise.json")
    noise_cache_path = os.path.join(RUNS_DIR, "generator_noise_cache.json")
    subset = [
        ex
        for ex in all_examples
        if ex["id"] in NOISE_SUBSET_IDS and ex.get("answerable", True)
    ]

    if not subset:
        print("⚠️  Noise estimation: no matching questions found in dataset.")
        return

    print(f"\n{'=' * 72}")
    print(f"NOISE ESTIMATION ({n_runs} runs × {len(subset)} questions)")
    print(f"Generator Model: {generator_model} (temp={GENERATOR_TEMPERATURE})")
    print(f"Judge Model: {judge_model} (temp=0.0, concurrency={judge_concurrency})")
    print("=" * 72)

    # -----------------------------------------------------------------------
    # Phase 1: Judge Noise (identical answers, uncached judge)
    # -----------------------------------------------------------------------
    print("\n--- PHASE 1: Judge Noise Estimation (bypassing judge cache) ---")
    judge_metric_scores: dict[str, list[list[float | None]]] = {
        m: [[] for _ in range(n_runs)] for m in metrics_map
    }

    for run_i in range(n_runs):
        print(f"  Judge noise run {run_i + 1}/{n_runs}...")
        noise_judge = RepoPilotJudgeLLM(
            judge_provider=judge_provider,
            judge_model=judge_model,
            generator_model=None,
            judge_cache_path=noise_cache_path,
            api_key=api_key,
            ollama_base_url=ollama_url,
            concurrency=judge_concurrency,
            cache_prefix=f"noise_judge_run_{run_i}::",
        )
        noise_metrics = _build_metrics(noise_judge, contexts_mode)

        for ex in subset:
            question = ex["question"]
            ideal = ex.get("ideal_answer", "")

            if contexts_mode == "gold":
                snippets = _fetch_gold_context(target_repo_id, ex, top_k, symbol_ranges)
            else:
                snippets = _fetch_retrieved_context(target_repo_id, question, top_k)

            context_texts = [
                f"File: {s.file_path} (lines {s.start_line}-{s.end_line})\n{s.content}"
                for s in snippets
            ]

            # Answers served from cache to isolate judge variance
            answer, _ = _generate_answer(
                question, snippets, contexts_mode, answer_cache, answer_cache_path
            )

            noise_case = LLMTestCase(
                input=question,
                actual_output=answer,
                retrieval_context=context_texts,
                expected_output=ideal,
            )

            for metric_name, metric in noise_metrics.items():
                score, _ = _measure_metric(metric, noise_case)
                judge_metric_scores[metric_name][run_i].append(score)

    judge_noise_results: dict[str, dict[str, Any]] = {}
    for metric_name, runs_data in judge_metric_scores.items():
        n_q = len(subset)
        per_q_stdevs: list[float] = []
        for q_i in range(n_q):
            valid_scores = [runs_data[r][q_i] for r in range(n_runs) if runs_data[r][q_i] is not None]
            if len(valid_scores) >= 2:
                per_q_stdevs.append(_stats.stdev(valid_scores))
        avg_noise = round(sum(per_q_stdevs) / len(per_q_stdevs), 4) if per_q_stdevs else 0.0
        judge_noise_results[metric_name] = {
            "avg_std_dev": avg_noise,
            "n_questions": n_q,
            "n_runs": n_runs,
        }
        print(f"  [Judge]     {metric_name:22s}  std dev = {avg_noise:.4f}")

    # -----------------------------------------------------------------------
    # Phase 2: Generator Noise (if temperature > 0, bypass answer cache)
    # -----------------------------------------------------------------------
    generator_noise_results: dict[str, dict[str, Any]] = {}
    if GENERATOR_TEMPERATURE > 0:
        print(f"\n--- PHASE 2: Generator Noise Estimation (temp={GENERATOR_TEMPERATURE}, regenerating 3x) ---")
        gen_metric_scores: dict[str, list[list[float | None]]] = {
            m: [[] for _ in range(n_runs)] for m in metrics_map
        }
        # Fixed judge instance with noise cache to isolate generator variance
        eval_judge = RepoPilotJudgeLLM(
            judge_provider=judge_provider,
            judge_model=judge_model,
            generator_model=None,
            judge_cache_path=noise_cache_path,
            api_key=api_key,
            ollama_base_url=ollama_url,
            concurrency=judge_concurrency,
            cache_prefix="noise_gen::",
        )
        eval_metrics = _build_metrics(eval_judge, contexts_mode)

        for run_i in range(n_runs):
            print(f"  Generator run {run_i + 1}/{n_runs} (generating fresh answers)...")
            for ex in subset:
                question = ex["question"]
                ideal = ex.get("ideal_answer", "")

                if contexts_mode == "gold":
                    snippets = _fetch_gold_context(target_repo_id, ex, top_k, symbol_ranges)
                else:
                    snippets = _fetch_retrieved_context(target_repo_id, question, top_k)

                context_texts = [
                    f"File: {s.file_path} (lines {s.start_line}-{s.end_line})\n{s.content}"
                    for s in snippets
                ]

                # Bypass answer cache: call ask_with_llm directly
                try:
                    answer = ask_with_llm(question, snippets)
                except Exception as exc:
                    logger.warning("Generation failed in noise run %d for %s: %s", run_i + 1, ex["id"], exc)
                    answer = "Failed to generate answer."

                noise_case = LLMTestCase(
                    input=question,
                    actual_output=answer,
                    retrieval_context=context_texts,
                    expected_output=ideal,
                )

                for metric_name, metric in eval_metrics.items():
                    score, _ = _measure_metric(metric, noise_case)
                    gen_metric_scores[metric_name][run_i].append(score)

        for metric_name, runs_data in gen_metric_scores.items():
            n_q = len(subset)
            per_q_stdevs = []
            for q_i in range(n_q):
                valid_scores = [runs_data[r][q_i] for r in range(n_runs) if runs_data[r][q_i] is not None]
                if len(valid_scores) >= 2:
                    per_q_stdevs.append(_stats.stdev(valid_scores))
            avg_noise = round(sum(per_q_stdevs) / len(per_q_stdevs), 4) if per_q_stdevs else 0.0
            generator_noise_results[metric_name] = {
                "avg_std_dev": avg_noise,
                "n_questions": n_q,
                "n_runs": n_runs,
            }
            print(f"  [Generator] {metric_name:22s}  std dev = {avg_noise:.4f}")

    # Combine noise: sqrt(judge_noise^2 + generator_noise^2)
    combined_noise_results: dict[str, dict[str, Any]] = {}
    print("\n--- COMBINED NOISE FLOOR ---")
    for metric_name in metrics_map:
        j_sd = judge_noise_results.get(metric_name, {}).get("avg_std_dev", 0.0)
        g_sd = generator_noise_results.get(metric_name, {}).get("avg_std_dev", 0.0)
        comb_sd = round(math.sqrt(j_sd**2 + g_sd**2), 4)
        combined_noise_results[metric_name] = {
            "avg_std_dev": comb_sd,
            "judge_std_dev": j_sd,
            "generator_std_dev": g_sd,
            "n_questions": len(subset),
            "n_runs": n_runs,
        }
        print(f"  {metric_name:22s}  combined noise floor = {comb_sd:.4f} (judge={j_sd:.4f}, gen={g_sd:.4f})")

    noise_payload: dict[str, Any] = {
        "timestamp": datetime.datetime.now().isoformat(),
        "contexts_mode": contexts_mode,
        "n_runs": n_runs,
        "generator_temperature": GENERATOR_TEMPERATURE,
        "subset_ids": [ex["id"] for ex in subset],
        "judge_noise": judge_noise_results,
        "generator_noise": generator_noise_results,
        "per_metric_noise": combined_noise_results,
    }
    with open(noise_path, "w", encoding="utf-8") as f:
        json.dump(noise_payload, f, indent=2)
    print(f"\n  Noise estimates saved → {noise_path}")
    print("=" * 72)
    print(f"\n  Noise estimates saved → {noise_path}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RepoPilot Generator Evaluation Runner (DeepEval)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Quick smoke run (5 questions)
  python evals/scripts/run_generator_eval.py --contexts gold --limit 5

  # Full generator-isolation eval and save as baseline
  python evals/scripts/run_generator_eval.py --contexts gold --save-baseline

  # Full pipeline eval (retriever + generator)
  python evals/scripts/run_generator_eval.py --contexts retrieved

  # Estimate judge noise (expensive — run once, results persist)
  python evals/scripts/run_generator_eval.py --contexts gold --estimate-noise
""",
    )
    parser.add_argument(
        "--contexts",
        choices=["gold", "retrieved"],
        default="gold",
        help="'gold' = golden ideal chunks; 'retrieved' = live retriever output",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Context chunks per question (default: 5, same as production)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of answerable questions evaluated (cheap smoke runs)",
    )
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Also write generator_baseline_<mode>.json alongside the timestamped run",
    )
    parser.add_argument(
        "--estimate-noise",
        action="store_true",
        help="Run the judge 3× on 8 fixed questions to estimate per-metric noise",
    )
    parser.add_argument(
        "--repo-id",
        type=int,
        default=None,
        metavar="ID",
        help="Database repository ID (auto-resolved if omitted)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=GOLDEN_PATH,
        help="Path to golden JSONL dataset (default: evals/data/golden_v2.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output JSON path (auto-generated timestamp name if omitted)",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=2,
        metavar="N",
        help="Maximum concurrent requests to judge LLM (default: 2)",
    )
    parser.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip all judge calls; generate answers (via local Ollama), update answer cache, and compute deterministic metrics only",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Optional metric names to evaluate (e.g. faithfulness answer_relevancy correctness contextual_relevancy). "
             "Default: faithfulness answer_relevancy correctness. contextual_relevancy is opt-in.",
    )
    args = parser.parse_args()

    run_generator_eval(
        contexts_mode=args.contexts,
        top_k=args.top_k,
        limit=args.limit,
        save_baseline=args.save_baseline,
        estimate_noise=args.estimate_noise,
        repo_id=args.repo_id,
        dataset_path=args.dataset,
        output_path=args.output,
        judge_concurrency=args.judge_concurrency,
        skip_judge=args.skip_judge,
        metrics_list=args.metrics,
    )
