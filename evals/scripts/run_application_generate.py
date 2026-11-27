"""Application-level evaluation harness — Phase 1: Generation.

Calls the REAL RepoPilot chat endpoint (the same LangGraph path the product
uses) for every question in the golden dataset and saves answers to a JSONL
run file.  No LLM judge is invoked here; judging happens in Phase 2.

Design choices
--------------
* We talk to the **live HTTP API** via httpx (SSE streaming), so we exercise
  the same code path the user does — not an injected context shortcut.
* The script is resumable: it skips question IDs already written to the output
  file, so you can Ctrl-C and re-run safely.
* Every row records latency, TTFT, token counts (if emitted by SSE),
  model_name, hardware_note, the final answer text, and any error/timeout flag.
* Note: backend/app/routers/chat.py currently uses get_default_model() in
  graph.py and does not emit token counts in SSE events (logged as None
  unless a small instrumentation hook is added).

Usage
-----
    # Dry run — shows what would be called, no network requests
    python evals/scripts/run_application_generate.py --dry-run

    # Smoke test — first 3 answerable questions only
    python evals/scripts/run_application_generate.py --limit 3

    # Full run — all 32 questions (26 answerable + 3 unanswerable + 3 OOS)
    python evals/scripts/run_application_generate.py

    # Only answerable questions
    python evals/scripts/run_application_generate.py --answerable-only

    # Custom output file
    python evals/scripts/run_application_generate.py --output evals/runs/app_myrun.jsonl

Environment (backend/.env)
---------------------------
    API_BASE_URL   base URL of the running FastAPI server  (default: http://localhost:8000)
    REPO_ID        integer id of the indexed repository    (default: 8  = "repopilot")

Requirements
------------
    pip install httpx   (already in backend requirements)
    The FastAPI server must be running before invoking this script.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import sys
import time
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Path setup — same pattern as run_eval.py so backend modules are importable
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "backend"))
sys.path.insert(0, os.path.join(REPO_ROOT, "evals", "scripts"))

load_dotenv(os.path.join(REPO_ROOT, "backend", ".env"))

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
GOLDEN_PATH = os.path.join(REPO_ROOT, "evals", "data", "golden_v2.jsonl")
RUNS_DIR = os.path.join(REPO_ROOT, "evals", "runs")
QUOTA_LEDGER_PATH = os.path.join(RUNS_DIR, "quota_ledger.jsonl")

# Default server settings — override via env vars or CLI flags
DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_REPO_ID = int(os.getenv("REPO_ID", "8"))  # repopilot repo ID in DB
DEFAULT_MAX_REQUESTS = 160  # Guardrail cap for Gemini 500/day free-tier quota

# Per-question timeout in seconds — generous because Ollama can be slow
QUESTION_TIMEOUT_S = 180

# SSE event prefix emitted by the FastAPI stream endpoint
SSE_DATA_PREFIX = "data: "


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict[str, Any]]:
    """Load a JSONL golden dataset from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dataset_sha256(path: str) -> str:
    """SHA-256 of the dataset file contents (reproducibility marker)."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
# Resume helpers — read already-completed question IDs from an existing JSONL
# ---------------------------------------------------------------------------

def load_completed_ids(output_path: str, include_errors: bool = True) -> set[str]:
    """Return set of question IDs already written to the output file.

    When include_errors=False, only successful (error=False) IDs are returned so
    failed rows can be retried once with --retry-failed.
    """
    latest: dict[str, dict[str, Any]] = {}
    if not os.path.exists(output_path):
        return set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qid = rec.get("question_id")
                if qid:
                    latest[qid] = rec
            except json.JSONDecodeError:
                pass
    if include_errors:
        return set(latest.keys())
    return {qid for qid, rec in latest.items() if not rec.get("error")}


def load_failed_ids(output_path: str) -> set[str]:
    """Return set of question IDs whose latest record in output_path is an error."""
    latest: dict[str, dict[str, Any]] = {}
    if not os.path.exists(output_path):
        return set()
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                qid = rec.get("question_id")
                if qid:
                    latest[qid] = rec
            except json.JSONDecodeError:
                pass
    return {qid for qid, rec in latest.items() if rec.get("error")}


# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------

def _create_session(client: httpx.Client, api_base: str, repo_id: int) -> str:
    """Create a fresh chat session for this question and return the session ID."""
    url = f"{api_base}/api/chat/{repo_id}/sessions"
    resp = client.post(url, json={"title": "eval-run"}, timeout=30)
    resp.raise_for_status()
    return resp.json()["id"]


def _fetch_github_head_sha(client: httpx.Client, api_base: str, repo_id: int) -> str | None:
    """Fetch the GitHub HEAD commit SHA for repo_id via API + GitHub REST API (best-effort)."""
    try:
        repo_resp = client.get(f"{api_base}/api/repositories/{repo_id}", timeout=10)
        if repo_resp.status_code != 200:
            return None
        full_name = repo_resp.json().get("full_name", "")
        if "/" not in full_name:
            return None
        headers = {"Accept": "application/vnd.github.v3+json"}
        gh_tok = os.getenv("GITHUB_TOKEN", "").strip()
        if gh_tok:
            headers["Authorization"] = f"Bearer {gh_tok}"
        gh_resp = client.get(
            f"https://api.github.com/repos/{full_name}/commits",
            params={"per_page": 1},
            headers=headers,
            timeout=10,
        )
        if gh_resp.status_code == 200 and gh_resp.json():
            return gh_resp.json()[0].get("sha")
    except Exception:
        pass
    return None


def _stream_question(
    client: httpx.Client,
    api_base: str,
    repo_id: int,
    session_id: str,
    question: str,
    timeout_s: float = QUESTION_TIMEOUT_S,
) -> dict[str, Any]:
    """
    Send one question via the SSE streaming endpoint and collect the response.

    Returns a dict with:
        final_answer   — concatenated text tokens from the stream
        citations      — list of file paths extracted from the answer text
        retrieved_context — list of tool_end payloads (chunks used by the agent)
        total_latency_s — wall-clock time from request start to done event
        ttft_s         — time from request start to first text token
        input_tokens   — not available from SSE; logged as None
        output_tokens  — not available from SSE; logged as None
        error          — bool: True if an error event was received
        timeout        — bool: True if the request timed out
        error_message  — the error text if error=True
        tool_calls     — list of {tool, args} dicts (agent tool invocations)
    """
    url = f"{api_base}/api/chat/{repo_id}/sessions/{session_id}/stream"
    payload = {"message": question}

    answer_tokens: list[str] = []
    tool_events: list[dict] = []
    retrieved_chunks: list[str] = []
    tool_calls: list[dict] = []
    tool_durations: list[dict[str, Any]] = []
    open_tools: list[tuple[str, float]] = []
    first_token_time: float | None = None
    first_tool_time: float | None = None
    tool_rounds = 0
    in_tool_batch = False
    had_error = False
    had_timeout = False
    error_message: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None

    t_start = time.perf_counter()

    try:
        # Use a streaming context so we can read SSE events line by line.
        with client.stream(
            "POST",
            url,
            json=payload,
            timeout=timeout_s,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line.startswith(SSE_DATA_PREFIX):
                    continue
                data_str = raw_line[len(SSE_DATA_PREFIX):]
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")

                if event_type == "token":
                    # Text delta — build the final answer
                    delta = event.get("delta", "")
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        answer_tokens.append(delta)

                elif event_type == "tool_start":
                    # Agent started a tool call — record timing and args
                    now = time.perf_counter()
                    if first_tool_time is None:
                        first_tool_time = now
                    if not in_tool_batch:
                        tool_rounds += 1
                        in_tool_batch = True
                    tool_name = event.get("tool", "")
                    open_tools.append((tool_name, now))
                    tool_calls.append({
                        "tool": tool_name,
                        "args": event.get("args", {}),
                    })

                elif event_type == "tool_end":
                    # Tool execution completed — record duration and output
                    now = time.perf_counter()
                    tool_name = event.get("tool", "")
                    output = event.get("output", "")
                    start_t = None
                    for i, (open_name, t0) in enumerate(open_tools):
                        if open_name == tool_name:
                            start_t = t0
                            open_tools.pop(i)
                            break
                    if start_t is None and open_tools:
                        _, start_t = open_tools.pop(0)
                    if not open_tools:
                        in_tool_batch = False
                    tool_durations.append({
                        "tool": tool_name,
                        "duration_s": round(now - start_t, 3) if start_t is not None else None,
                    })
                    tool_events.append({"tool": tool_name, "output": output})
                    if output:
                        retrieved_chunks.append(f"[{tool_name}] {output}")

                elif event_type == "usage":
                    # Optional token usage event if emitted by the stream endpoint
                    if event.get("input_tokens") is not None:
                        input_tokens = (input_tokens or 0) + int(event["input_tokens"])
                    if event.get("output_tokens") is not None:
                        output_tokens = (output_tokens or 0) + int(event["output_tokens"])

                elif event_type == "done":
                    if event.get("input_tokens") is not None:
                        input_tokens = int(event["input_tokens"])
                    if event.get("output_tokens") is not None:
                        output_tokens = int(event["output_tokens"])
                    break  # Stream complete

                elif event_type == "error":
                    had_error = True
                    error_message = event.get("error", "unknown error")
                    break

    except httpx.TimeoutException:
        had_timeout = True
        error_message = f"Request timed out after {timeout_s}s"
    except httpx.HTTPStatusError as exc:
        had_error = True
        error_message = f"HTTP {exc.response.status_code}: {exc.response.text[:200]}"
    except Exception as exc:
        had_error = True
        error_message = str(exc)

    t_end = time.perf_counter()
    total_latency = round(t_end - t_start, 3)
    ttft = round(first_token_time - t_start, 3) if first_token_time else None
    time_to_first_tool = round(first_tool_time - t_start, 3) if first_tool_time else None
    llm_calls = (1 + tool_rounds) if not (had_error or had_timeout) else tool_rounds + 1

    final_answer = "".join(answer_tokens).strip()

    # Extract file citations from the answer text (simple heuristic: look for
    # patterns like `backend/app/foo.py` or `[path.py:L1-L10]`).
    citations = _extract_file_references(final_answer)

    return {
        "final_answer": final_answer,
        "citations": citations,
        "retrieved_context": retrieved_chunks,
        "tool_calls": tool_calls,
        "tool_durations": tool_durations,
        "time_to_first_tool_s": time_to_first_tool,
        "llm_calls": llm_calls,
        "total_latency_s": total_latency,
        "ttft_s": ttft,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "error": had_error or had_timeout,
        "timeout": had_timeout,
        "error_message": error_message,
    }


def _extract_file_references(text: str) -> list[str]:
    """
    Extract file path references from the answer text.

    Looks for patterns like:
      - backend/app/search.py
      - [backend/app/search.py:L14-L58]
      - `backend/app/search.py`
    Returns a deduplicated list of paths found.
    """
    import re
    # Match common path patterns: word/word.ext with optional line ranges
    pattern = re.compile(
        r"(?:^|[\s`\['\"])([a-zA-Z0-9_./\\-]+\.[a-zA-Z0-9]+)"
        r"(?::(?:L|line)?\d+(?:[-–]\d+)?)?",
        re.MULTILINE,
    )
    seen: set[str] = set()
    refs: list[str] = []
    for m in pattern.finditer(text):
        path = m.group(1).strip()
        # Only keep paths that look like project files (contain / or .)
        if ("/" in path or "\\" in path) and "." in path:
            if path not in seen:
                seen.add(path)
                refs.append(path)
    return refs


def _detect_hardware_note() -> str:
    """Return a short human-readable description of the local host hardware."""
    cpu_count = os.cpu_count() or 1
    return f"{platform.system()} {platform.release()} ({platform.machine()}, {cpu_count} CPUs)"


def _default_model_name() -> str:
    """Resolve the model name configured for the real chat agent (graph.py)."""
    return os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")


DEFAULT_CHAT_TEMPERATURE = 0.2  # Matches backend/app/agent/graph.py get_default_model()


# ---------------------------------------------------------------------------
# Summary, slice & quota-ledger helpers (pure logic, unit-tested)
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float | None:
    """Compute linear-interpolated percentile (0..100) for a non-empty list."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return round(float(xs[0]), 3)
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return round(xs[lo] * (1.0 - frac) + xs[hi] * frac, 3)


def classify_slice(rec: dict[str, Any]) -> str:
    """Classify a record into answerable / unanswerable / out_of_scope."""
    cat = (rec.get("category") or "").strip().lower()
    if cat == "out_of_scope":
        return "out_of_scope"
    if cat == "unanswerable" or not rec.get("answerable", True):
        return "unanswerable"
    return "answerable"


def summarize_run(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute latency, token, request, and slice stats for a list of run records."""
    # Deduplicate by question_id keeping the latest record (e.g. after --retry-failed)
    dedup: dict[str, dict[str, Any]] = {}
    for r in records:
        qid = r.get("question_id")
        if qid:
            dedup[qid] = r
    latest_records = list(dedup.values()) if dedup else records

    successes = [r for r in latest_records if not r.get("error")]
    errors = [r for r in latest_records if r.get("error")]
    failed_ids = [r.get("question_id") for r in errors if r.get("question_id")]

    # Exclude retried rows from latency percentiles so retry overhead doesn't skew ops stats
    latency_rows = [r for r in successes if not r.get("retried")]
    retried_excluded = sum(1 for r in successes if r.get("retried"))

    latencies = [float(r["total_latency_s"]) for r in latency_rows if r.get("total_latency_s") is not None]
    ttft_tools = [float(r["time_to_first_tool_s"]) for r in latency_rows if r.get("time_to_first_tool_s") is not None]
    in_toks = [int(r["input_tokens"]) for r in successes if r.get("input_tokens") is not None]
    out_toks = [int(r["output_tokens"]) for r in successes if r.get("output_tokens") is not None]
    total_requests = sum(int(r.get("llm_calls") or 0) for r in latest_records)

    slices: dict[str, dict[str, Any]] = {}
    for slice_name in ("answerable", "unanswerable", "out_of_scope"):
        s_recs = [r for r in latest_records if classify_slice(r) == slice_name]
        s_ok = [r for r in s_recs if not r.get("error")]
        s_lat = [float(r["total_latency_s"]) for r in s_ok if not r.get("retried") and r.get("total_latency_s") is not None]
        slices[slice_name] = {
            "total": len(s_recs),
            "success": len(s_ok),
            "errors": len(s_recs) - len(s_ok),
            "p50_latency_s": _percentile(s_lat, 50),
            "requests": sum(int(r.get("llm_calls") or 0) for r in s_recs),
        }

    return {
        "total": len(latest_records),
        "success": len(successes),
        "errors": len(errors),
        "failed_ids": failed_ids,
        "retried_excluded_from_latency": retried_excluded,
        "total_requests": total_requests,
        "p50_latency_s": _percentile(latencies, 50),
        "p95_latency_s": _percentile(latencies, 95),
        "p99_latency_s": _percentile(latencies, 99),
        "p50_ttft_tool_s": _percentile(ttft_tools, 50),
        "p95_ttft_tool_s": _percentile(ttft_tools, 95),
        "mean_input_tokens": round(sum(in_toks) / len(in_toks), 1) if in_toks else None,
        "max_input_tokens": max(in_toks) if in_toks else None,
        "mean_output_tokens": round(sum(out_toks) / len(out_toks), 1) if out_toks else None,
        "max_output_tokens": max(out_toks) if out_toks else None,
        "slices": slices,
    }


PACIFIC_TZ = ZoneInfo("America/Los_Angeles")


def append_quota_ledger(
    script_name: str,
    model: str,
    requests_used: int,
    output_file: str,
    ledger_path: str = QUOTA_LEDGER_PATH,
) -> None:
    """Append one entry to evals/runs/quota_ledger.jsonl using midnight-Pacific quota day."""
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    now_local = datetime.datetime.now().astimezone()
    now_pacific = now_local.astimezone(PACIFIC_TZ)
    entry = {
        "date": now_pacific.date().isoformat(),
        "quota_tz": "America/Los_Angeles",
        "timestamp": now_local.isoformat(timespec="seconds"),
        "script": script_name,
        "model": model,
        "requests_used": requests_used,
        "output_file": os.path.relpath(output_file, REPO_ROOT),
    }
    with open(ledger_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_generation(
    dataset_path: str = GOLDEN_PATH,
    output_path: str | None = None,
    api_base: str = DEFAULT_API_BASE,
    repo_id: int = DEFAULT_REPO_ID,
    limit: int | None = None,
    answerable_only: bool = False,
    dry_run: bool = False,
    model_name: str | None = None,
    hardware_note: str | None = None,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    retried: bool = False,
    retry_failed: bool = False,
) -> str:
    """
    Run application-level generation for every question in the dataset.

    Returns the output file path.
    """
    resolved_model_name = model_name or _default_model_name()
    resolved_hardware_note = hardware_note or _detect_hardware_note()
    # Resolve output file path (timestamp-stamped if not provided)
    os.makedirs(RUNS_DIR, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_path is None:
        output_path = os.path.join(RUNS_DIR, f"application_{timestamp}.jsonl")

    dataset = load_dataset(dataset_path)
    sha = dataset_sha256(dataset_path)

    # Optionally filter to answerable questions only
    if answerable_only:
        dataset = [ex for ex in dataset if ex.get("answerable", True)]

    # Apply --limit
    if limit is not None:
        dataset = dataset[:limit]

    # Find already-done IDs so we can resume a partial run
    completed_ids = load_completed_ids(output_path, include_errors=not retry_failed)
    previously_failed_ids = load_failed_ids(output_path) if retry_failed else set()

    pending = [ex for ex in dataset if ex["id"] not in completed_ids]

    print()
    print("=" * 70)
    print("RepoPilot — Application-Level Evaluation  Phase 1: Generate")
    print("=" * 70)
    print(f"  Dataset       : {os.path.relpath(dataset_path)}  (SHA-256: {sha[:12]}…)")
    print(f"  Total rows    : {len(dataset)}  (pending: {len(pending)}, already done: {len(completed_ids)})")
    print(f"  Output file   : {os.path.relpath(output_path)}")
    print(f"  API base      : {api_base}")
    print(f"  Repository ID : {repo_id}")
    print(f"  Model         : {resolved_model_name}")
    print(f"  Hardware      : {resolved_hardware_note}")
    print(f"  Timeout       : {QUESTION_TIMEOUT_S}s per question")
    if answerable_only:
        print("  Filter        : answerable only")
    if limit is not None:
        print(f"  Limit         : {limit} questions")
    print("=" * 70)
    print()

    if dry_run:
        print("DRY RUN — no API calls will be made.  Questions that would run:")
        for i, ex in enumerate(pending, 1):
            ans_flag = "answerable" if ex.get("answerable", True) else "NOT answerable"
            print(f"  [{i:02d}] {ex['id']}  ({ans_flag})  {ex['question'][:70]}...")
        print()
        print(f"Would write {len(pending)} rows to: {output_path}")
        return output_path

    if not pending:
        print("Nothing to do — all questions already written to output file.")
        return output_path

    # Open output file in append mode so resumption works naturally
    with httpx.Client(timeout=30) as client, open(output_path, "a", encoding="utf-8") as out_f:

        # --- Health-check the API before starting ---
        try:
            health_resp = client.get(f"{api_base}/api/health", timeout=10)
            health_resp.raise_for_status()
            print(f"✓ API health check passed ({api_base}/api/health)\n")
        except Exception as exc:
            print(
                f"ERROR: Cannot reach the API at {api_base}.\n"
                f"  Detail: {exc}\n"
                f"  Please start the FastAPI server first:\n"
                f"    cd backend && uvicorn app.main:app --reload\n",
                file=sys.stderr,
            )
            sys.exit(1)

        repo_head_sha = _fetch_github_head_sha(client, api_base, repo_id)
        if repo_head_sha:
            print(f"  GitHub HEAD SHA : {repo_head_sha[:12]}\n")

        requests_used = 0
        stopped_for_quota = False
        for idx, ex in enumerate(pending, 1):
            if requests_used > max_requests:
                print(
                    f"\n⚠️  STOPPING: Running request count ({requests_used}) "
                    f"exceeded max_requests ({max_requests})."
                )
                stopped_for_quota = True
                break

            qid = ex["id"]
            question = ex["question"]
            is_answerable = ex.get("answerable", True)
            category = ex.get("category", "general")
            gt_files = ex.get("gt_files", [])

            print(
                f"[{idx:02d}/{len(pending):02d}] {qid}  "
                f"({category}, {'answerable' if is_answerable else 'NOT answerable'})  "
                f"{question[:60]}..."
            )

            is_retry = retried or (qid in previously_failed_ids)
            # Create a fresh session for each question so conversations don't
            # contaminate each other.
            try:
                session_id = _create_session(client, api_base, repo_id)
            except Exception as exc:
                print(f"  ✗ Could not create session: {exc}")
                record = _make_error_record(
                    ex, sha, timestamp, str(exc),
                    model_name=resolved_model_name,
                    hardware_note=resolved_hardware_note,
                    retried=is_retry,
                )
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                continue

            # Stream the question and collect results (no automatic retries)
            result = _stream_question(client, api_base, repo_id, session_id, question)
            requests_used += int(result.get("llm_calls") or 0)

            if result["error"]:
                status_icon = "✗"
            else:
                status_icon = "✓"

            answer_preview = (result["final_answer"] or "")[:80].replace("\n", " ")
            latency = result["total_latency_s"]
            ttft = result["ttft_s"]

            print(
                f"  {status_icon}  latency={latency}s  ttft={ttft}s  "
                f"llm_calls={result['llm_calls']}  running_reqs={requests_used}  "
                f"in_tok={result['input_tokens']}  out_tok={result['output_tokens']}"
            )
            if result["error"]:
                print(f"     error: {result['error_message']}")
            else:
                print(f"     preview: {answer_preview!r}")

            # Build the output record — one JSON object per question
            record = {
                # Metadata
                "question_id": qid,
                "question": question,
                "category": category,
                "answerable": is_answerable,
                "gt_files": gt_files,
                "gt_symbols": ex.get("gt_symbols", []),
                "ideal_answer": ex.get("ideal_answer", ""),
                # Generation config
                "run_timestamp": timestamp,
                "dataset_sha256": sha,
                "api_base": api_base,
                "repo_id": repo_id,
                "repo_head_sha": repo_head_sha,
                "session_id": session_id,
                "model_name": resolved_model_name,
                "generator_config": {
                    "model": resolved_model_name,
                    "temperature": DEFAULT_CHAT_TEMPERATURE,
                },
                "hardware_note": resolved_hardware_note,
                # Results
                "final_answer": result["final_answer"],
                "citations": result["citations"],
                "retrieved_context": result["retrieved_context"],
                "tool_calls": result["tool_calls"],
                "tool_durations": result["tool_durations"],
                # Performance
                "time_to_first_tool_s": result["time_to_first_tool_s"],
                "llm_calls": result["llm_calls"],
                "total_latency_s": result["total_latency_s"],
                "ttft_s": result["ttft_s"],
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                # Status
                "error": result["error"],
                "timeout": result["timeout"],
                "error_message": result["error_message"],
                "retried": is_retry,
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()  # Write immediately so Ctrl-C doesn't lose data

    # Log requests used to quota ledger
    append_quota_ledger(
        script_name="run_application_generate.py",
        model=resolved_model_name,
        requests_used=requests_used,
        output_file=output_path,
    )

    # Print final summary
    all_records = load_all_records(output_path)
    dataset_ids = {ex["id"] for ex in dataset}
    done_records = [r for r in all_records if r.get("question_id") in dataset_ids]
    summary = summarize_run(done_records)

    print()
    print("=" * 70)
    print(f"Generation {'STOPPED (quota cap)' if stopped_for_quota else 'complete'}.  Output: {output_path}")
    print(f"  Success / Errors : {summary['success']} / {summary['errors']} (total: {summary['total']})")
    if summary["failed_ids"]:
        print(f"  Failed IDs       : {', '.join(summary['failed_ids'])}")
    print(f"  Requests used    : {summary['total_requests']} (this run: {requests_used})")
    if summary.get("retried_excluded_from_latency"):
        print(
            f"  Latency note     : excluded {summary['retried_excluded_from_latency']} "
            "retried row(s) from P50/P95/P99 latency stats"
        )
    print(
        f"  Latency (s)      : P50={summary['p50_latency_s']}  "
        f"P95={summary['p95_latency_s']}  P99={summary['p99_latency_s']}"
    )
    print(
        f"  Time to 1st tool : P50={summary['p50_ttft_tool_s']}s  "
        f"P95={summary['p95_ttft_tool_s']}s"
    )
    print(
        f"  Input tokens     : mean={summary['mean_input_tokens']}  max={summary['max_input_tokens']}"
    )
    print(
        f"  Output tokens    : mean={summary['mean_output_tokens']}  max={summary['max_output_tokens']}"
    )
    print("  Slices           :")
    for s_name, s_stats in summary["slices"].items():
        print(
            f"    - {s_name:<14} total={s_stats['total']}  ok={s_stats['success']}  "
            f"err={s_stats['errors']}  p50={s_stats['p50_latency_s']}s  reqs={s_stats['requests']}"
        )
    print("=" * 70)
    print()

    return output_path


def _make_error_record(
    ex: dict,
    sha: str,
    timestamp: str,
    error_msg: str,
    model_name: str | None = None,
    hardware_note: str | None = None,
    retried: bool = False,
) -> dict:
    """Build a minimal error record when session creation fails."""
    resolved_model = model_name or _default_model_name()
    return {
        "question_id": ex["id"],
        "question": ex["question"],
        "category": ex.get("category", "general"),
        "answerable": ex.get("answerable", True),
        "gt_files": ex.get("gt_files", []),
        "gt_symbols": ex.get("gt_symbols", []),
        "ideal_answer": ex.get("ideal_answer", ""),
        "run_timestamp": timestamp,
        "dataset_sha256": sha,
        "api_base": DEFAULT_API_BASE,
        "repo_id": DEFAULT_REPO_ID,
        "session_id": None,
        "model_name": resolved_model,
        "generator_config": {
            "model": resolved_model,
            "temperature": DEFAULT_CHAT_TEMPERATURE,
        },
        "hardware_note": hardware_note or _detect_hardware_note(),
        "final_answer": "",
        "citations": [],
        "retrieved_context": [],
        "tool_calls": [],
        "tool_durations": [],
        "time_to_first_tool_s": None,
        "llm_calls": 0,
        "total_latency_s": None,
        "ttft_s": None,
        "input_tokens": None,
        "output_tokens": None,
        "error": True,
        "timeout": False,
        "error_message": error_msg,
        "retried": retried,
    }


def load_all_records(path: str) -> list[dict]:
    """Load all JSONL records from the output file."""
    records = []
    if not os.path.exists(path):
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RepoPilot Application-Level Evaluation — Phase 1: Generate",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset",
        default=GOLDEN_PATH,
        help="Path to golden JSONL dataset (default: evals/data/golden_v2.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output JSONL path (default: evals/runs/application_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--api-base",
        default=DEFAULT_API_BASE,
        help=f"FastAPI base URL (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument(
        "--repo-id",
        type=int,
        default=DEFAULT_REPO_ID,
        help=f"Repository ID in the database (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N questions (useful for smoke tests)",
    )
    parser.add_argument(
        "--answerable-only",
        action="store_true",
        help="Skip unanswerable and out-of-scope rows",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without making any API calls",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override model name recorded in cache metadata",
    )
    parser.add_argument(
        "--hardware-note",
        default=None,
        help="Override hardware note recorded in cache metadata",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS,
        help=f"Stop early if running LLM request count exceeds this cap (default: {DEFAULT_MAX_REQUESTS})",
    )
    parser.add_argument(
        "--retried",
        action="store_true",
        help="Mark newly written records with retried=true",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-run previously errored rows (marked retried=true) in addition to pending rows",
    )
    args = parser.parse_args()

    run_generation(
        dataset_path=args.dataset,
        output_path=args.output,
        api_base=args.api_base,
        repo_id=args.repo_id,
        limit=args.limit,
        answerable_only=args.answerable_only,
        dry_run=args.dry_run,
        model_name=args.model_name,
        hardware_note=args.hardware_note,
        max_requests=args.max_requests,
        retried=args.retried,
        retry_failed=args.retry_failed,
    )
