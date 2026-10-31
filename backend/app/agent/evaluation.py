"""LangSmith evaluation and benchmark suite for RepoPilot LangGraph agent.

This module provides:
1. Benchmark dataset definitions (multi-turn codebase comprehension and tool tasks).
2. Custom LangSmith evaluators:
   - Tool selection correctness
   - Keyword & concept grounding
   - Source code citation fidelity
3. Dataset synchronization with LangSmith Cloud.
4. Evaluation runners (both online via LangSmith `evaluate()` and offline local testing).
"""

import logging
import os
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from app.agent.graph import create_agent_graph

logger = logging.getLogger(__name__)

DATASET_NAME = "repopilot-agent-benchmark"

# ---------------------------------------------------------------------------
# Golden Evaluation Examples
# ---------------------------------------------------------------------------

DEFAULT_BENCHMARK_EXAMPLES: list[dict[str, Any]] = [
    {
        "inputs": {
            "question": "What is the primary programming language and repository description?",
            "repo_owner": "tiangolo",
            "repo_name": "fastapi",
            "repo_id": 1,
        },
        "outputs": {
            "expected_tools": ["fetch_repo_metadata"],
            "expected_keywords": ["python", "fastapi", "framework"],
            "requires_citation": False,
        },
    },
    {
        "inputs": {
            "question": "Inspect the root file structure and list key directories.",
            "repo_owner": "psf",
            "repo_name": "requests",
            "repo_id": 2,
        },
        "outputs": {
            "expected_tools": ["inspect_file_tree"],
            "expected_keywords": ["requests", "setup.py", "tests"],
            "requires_citation": True,
        },
    },
    {
        "inputs": {
            "question": "Read lines 1 to 40 of README.md and summarize what this project does.",
            "repo_owner": "encode",
            "repo_name": "uvicorn",
            "repo_id": 3,
        },
        "outputs": {
            "expected_tools": ["read_source_file"],
            "expected_keywords": ["asgi", "uvicorn", "server"],
            "requires_citation": True,
        },
    },
]


# ---------------------------------------------------------------------------
# Evaluator Functions
# ---------------------------------------------------------------------------

def evaluate_tool_selection(run: Any, example: Any) -> dict[str, Any]:
    """Evaluate whether the agent invoked the expected tools."""
    expected_tools = set(example.outputs.get("expected_tools", []))
    if not expected_tools:
        return {"key": "tool_selection", "score": 1.0, "comment": "No specific tools required"}

    # Extract tool names called during the run
    called_tools = set()
    messages = run.outputs.get("messages", []) if run.outputs else []
    for msg in messages:
        if isinstance(msg, dict):
            for tc in msg.get("tool_calls", []):
                called_tools.add(tc.get("name"))
        elif hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                called_tools.add(tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None))

    overlap = expected_tools.intersection(called_tools)
    score = len(overlap) / len(expected_tools) if expected_tools else 1.0
    return {
        "key": "tool_selection",
        "score": score,
        "comment": f"Called: {called_tools}, Expected: {expected_tools}",
    }


def evaluate_keyword_grounding(run: Any, example: Any) -> dict[str, Any]:
    """Evaluate whether key concepts from reference expectations appear in the final response."""
    expected_keywords = example.outputs.get("expected_keywords", [])
    if not expected_keywords:
        return {"key": "keyword_grounding", "score": 1.0, "comment": "No keywords specified"}

    # Extract final assistant answer text
    answer_text = ""
    messages = run.outputs.get("messages", []) if run.outputs else []
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", None)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role in {"assistant", "ai"} and content:
            answer_text = str(content).lower()
            break

    if not answer_text and run.outputs and "output" in run.outputs:
        answer_text = str(run.outputs["output"]).lower()

    matches = [kw for kw in expected_keywords if kw.lower() in answer_text]
    score = len(matches) / len(expected_keywords) if expected_keywords else 1.0
    return {
        "key": "keyword_grounding",
        "score": score,
        "comment": f"Found {len(matches)} of {len(expected_keywords)} expected keywords: {matches}",
    }


def evaluate_citation_presence(run: Any, example: Any) -> dict[str, Any]:
    """Evaluate whether the answer includes grounded file citations when required."""
    requires_citation = example.outputs.get("requires_citation", False)
    if not requires_citation:
        return {"key": "citation_presence", "score": 1.0, "comment": "Citation not required"}

    answer_text = ""
    messages = run.outputs.get("messages", []) if run.outputs else []
    for msg in reversed(messages):
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "type", None)
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")
        if role in {"assistant", "ai"} and content:
            answer_text = str(content)
            break

    # Look for file paths (e.g. `foo/bar.py` or `Lines 1-40`)
    file_pattern = re.compile(r"[\w.-]+/[\w.-]+\.\w+|File:|Lines \d+-\d+", re.IGNORECASE)
    has_citation = bool(file_pattern.search(answer_text))
    return {
        "key": "citation_presence",
        "score": 1.0 if has_citation else 0.0,
        "comment": "Found file/line reference" if has_citation else "No file citation detected",
    }


# ---------------------------------------------------------------------------
# LangSmith Dataset Synchronization
# ---------------------------------------------------------------------------

def sync_evaluation_dataset(dataset_name: str = DATASET_NAME) -> str | None:
    """Sync benchmark examples to LangSmith if API credentials are configured.

    Returns:
        str | None: The created or existing dataset ID, or None if LangSmith is unconfigured.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        logger.warning("LANGSMITH_API_KEY is not set. Skipping LangSmith dataset sync.")
        return None

    try:
        from langsmith import Client

        client = Client()
        # Check if dataset already exists
        if client.has_dataset(dataset_name=dataset_name):
            logger.info("LangSmith dataset '%s' already exists.", dataset_name)
            dataset = client.read_dataset(dataset_name=dataset_name)
            return str(dataset.id)

        dataset = client.create_dataset(
            dataset_name=dataset_name,
            description="RepoPilot LangGraph agent benchmark evaluation dataset for codebase comprehension and tool invocation.",
        )

        for example in DEFAULT_BENCHMARK_EXAMPLES:
            client.create_example(
                inputs=example["inputs"],
                outputs=example["outputs"],
                dataset_id=dataset.id,
            )

        logger.info("Successfully created LangSmith dataset '%s' with %d examples.", dataset_name, len(DEFAULT_BENCHMARK_EXAMPLES))
        return str(dataset.id)
    except Exception as exc:
        logger.error("Failed to sync dataset to LangSmith: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Offline & Online Evaluation Runners
# ---------------------------------------------------------------------------

async def evaluate_agent_offline(
    dataset: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run an evaluation sweep locally without requiring external LangSmith servers.

    Returns:
        dict: Aggregated summary of scores for tool selection, keywords, and citations.
    """
    examples = dataset or DEFAULT_BENCHMARK_EXAMPLES
    results: list[dict[str, Any]] = []

    graph = create_agent_graph()

    for ex in examples:
        inputs = ex["inputs"]
        question = inputs["question"]
        owner = inputs["repo_owner"]
        repo = inputs["repo_name"]
        repo_id = inputs["repo_id"]

        agent_input = {
            "messages": [HumanMessage(content=question)],
            "repo_id": repo_id,
            "repo_owner": owner,
            "repo_name": repo,
        }

        try:
            state = await graph.ainvoke(agent_input)
            run_output = {"messages": state.get("messages", [])}
        except Exception as exc:
            run_output = {"messages": [AIMessage(content=f"Error executing agent: {exc}")]}

        # Adapt to mock run/example structure for evaluator functions
        class MockRun:
            def __init__(self, outputs: dict):
                self.outputs = outputs

        class MockExample:
            def __init__(self, outputs: dict):
                self.outputs = outputs

        mock_run = MockRun(run_output)
        mock_ex = MockExample(ex["outputs"])

        tool_eval = evaluate_tool_selection(mock_run, mock_ex)
        kw_eval = evaluate_keyword_grounding(mock_run, mock_ex)
        cite_eval = evaluate_citation_presence(mock_run, mock_ex)

        results.append({
            "question": question,
            "tool_selection": tool_eval,
            "keyword_grounding": kw_eval,
            "citation_presence": cite_eval,
        })

    avg_tool = sum(r["tool_selection"]["score"] for r in results) / len(results) if results else 0.0
    avg_kw = sum(r["keyword_grounding"]["score"] for r in results) / len(results) if results else 0.0
    avg_cite = sum(r["citation_presence"]["score"] for r in results) / len(results) if results else 0.0

    return {
        "total_examples": len(results),
        "mean_scores": {
            "tool_selection": round(avg_tool, 3),
            "keyword_grounding": round(avg_kw, 3),
            "citation_presence": round(avg_cite, 3),
        },
        "details": results,
    }
