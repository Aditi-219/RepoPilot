"""RepoPilot Model Context Protocol (MCP) Server.

Exposes RepoPilot's codebase intelligence, hybrid retrieval, and GitHub
inspection tools to external AI environments like Cursor and Claude Desktop.

Available Tools:
  - fetch_repo_metadata: Get GitHub repository stars, language, description
  - inspect_file_tree:   Browse repository files filtered by path prefix
  - read_source_file:    Read code with line numbers and safety bounds
  - search_codebase:     Hybrid search (pgvector + FTS + reranker) over indexed code
  - ask_codebase:        Ground Q&A with LLM and code citations

Usage:
  Run stdio transport (default for Cursor / Claude Desktop):
      python -m app.mcp_server
"""

import asyncio
import logging
from typing import Any

from mcp.server.mcpserver import MCPServer

from app.database import SessionLocal
from app.github_service import fetch_file_content, fetch_repo_tree, fetch_repository
from app.llm_service import ask_with_llm
from app.models import Repository
from app.reranker import rerank
from app.search import SearchResult, hybrid_search

logger = logging.getLogger(__name__)

# Initialize MCP Server
mcp_app = MCPServer("RepoPilot")


@mcp_app.tool()
async def fetch_repo_metadata(owner: str, repo: str) -> dict[str, Any]:
    """Fetch GitHub repository metadata including stars, language, and description.

    Args:
        owner: GitHub repository owner (e.g. 'psf' or 'tiangolo').
        repo:  GitHub repository name (e.g. 'requests' or 'fastapi').

    Returns:
        Dictionary with full_name, stars, language, description, and url.
    """
    try:
        url = f"https://github.com/{owner}/{repo}"
        return await fetch_repository(url)
    except Exception as exc:
        return {"error": f"Failed to fetch metadata for {owner}/{repo}: {exc}"}


@mcp_app.tool()
async def inspect_file_tree(owner: str, repo: str, prefix: str = "") -> list[dict[str, Any]]:
    """Browse repository files, optionally filtered by directory prefix.

    Args:
        owner:  GitHub repository owner.
        repo:   GitHub repository name.
        prefix: Optional directory path prefix to filter (e.g. 'src/utils' or 'app/').

    Returns:
        List of file entries, each containing 'path' and 'size'.
    """
    try:
        entries = await fetch_repo_tree(owner, repo)
        if not prefix:
            return entries

        normalized = prefix.strip("/")
        return [
            entry
            for entry in entries
            if entry["path"].startswith(f"{normalized}/") or entry["path"] == normalized
        ]
    except Exception as exc:
        return [{"error": f"Failed to inspect file tree: {exc}"}]


@mcp_app.tool()
async def read_source_file(
    owner: str,
    repo: str,
    file_path: str,
    start_line: int = 1,
    end_line: int = 100,
) -> str:
    """Read source code from a GitHub repository with line numbers and safety bounds.

    Args:
        owner:      GitHub repository owner.
        repo:       GitHub repository name.
        file_path:  Relative file path in the repository (e.g. 'app/main.py').
        start_line: 1-indexed first line to read (default 1).
        end_line:   1-indexed last line to read (default 100, maximum 200 lines per call).

    Returns:
        Numbered lines of source code, or an error message if unavailable.
    """
    try:
        content = await fetch_file_content(owner, repo, file_path)
    except Exception as exc:
        return f"Error reading file '{file_path}': {exc}"

    if content is None:
        return f"Error: Could not read '{file_path}'. File may be binary or missing."

    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return f"File '{file_path}' is empty."

    start = max(1, start_line)
    end = min(total, max(start, min(end_line, start + 199)))
    selected = lines[start - 1 : end]
    numbered = [f"{i:4d} | {line}" for i, line in enumerate(selected, start=start)]
    header = f"File: {file_path} (Lines {start}-{end} of {total})\n"
    return header + "\n".join(numbered)


@mcp_app.tool()
def search_codebase(
    repository_id: int,
    query: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Search indexed codebase using RepoPilot's hybrid retrieval (vector similarity + FTS + reranking).

    Args:
        repository_id: Primary key of the indexed repository in RepoPilot.
        query:         Natural language query or code search terms.
        limit:         Maximum number of results to return (default 10).

    Returns:
        List of relevant code snippets with file path, line numbers, and relevance scores.
    """
    with SessionLocal() as db:
        repo = db.get(Repository, repository_id)
        if repo is None:
            return [{"error": f"Repository with id={repository_id} not found."}]
        if repo.status != "completed":
            return [{"error": f"Repository '{repo.full_name}' is not indexed yet (status='{repo.status}')."}]

        # Retrieve candidates via hybrid search and rerank
        candidates = hybrid_search(db, repository_id, query, limit=max(limit * 2, 20))
        reranked = rerank(query, candidates, top_k=limit)

        return [
            {
                "chunk_id": r.chunk_id,
                "file_path": r.file_path,
                "start_line": r.start_line,
                "end_line": r.end_line,
                "language": r.language,
                "score": round(r.score, 4),
                "content": r.content,
            }
            for r in reranked
        ]


@mcp_app.tool()
async def ask_codebase(
    repository_id: int,
    query: str,
    top_k: int = 6,
) -> str:
    """Ask a question about an indexed repository and get an AI response grounded in source code.

    Args:
        repository_id: Primary key of the indexed repository in RepoPilot.
        query:         Technical question about the codebase.
        top_k:         Number of code chunks to retrieve for grounding (default 6).

    Returns:
        Synthesized answer citing source files and line ranges.
    """
    def _retrieve() -> list[SearchResult]:
        with SessionLocal() as db:
            repo = db.get(Repository, repository_id)
            if repo is None or repo.status != "completed":
                return []
            candidates = hybrid_search(db, repository_id, query, limit=top_k * 3)
            return rerank(query, candidates, top_k=top_k)

    chunks = await asyncio.to_thread(_retrieve)
    if not chunks:
        return (
            f"No indexed code found for repository_id={repository_id}. "
            "Please verify the repository exists and has completed indexing."
        )

    # Generate answer with citations via LLM service
    try:
        response = await ask_with_llm(query, chunks)
        citations_summary = "\n\nSources Referenced:\n" + "\n".join(
            f"- {c.file_path} (Lines {c.start_line}-{c.end_line})" for c in chunks
        )
        return response.answer + citations_summary
    except Exception as exc:
        return f"Error generating answer with LLM: {exc}"


def main() -> None:
    """Run the MCP server via standard I/O transport."""
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting RepoPilot MCP Server over stdio...")
    mcp_app.run(transport="stdio")


if __name__ == "__main__":
    main()
