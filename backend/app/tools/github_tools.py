"""GitHub tools for RepoPilot LangGraph agent.

These tools allow the conversational agent to inspect repositories, browse the
directory structure, and retrieve source code files on demand.
"""

from typing import Any

from app.github_service import fetch_file_content, fetch_repo_tree, fetch_repository
from fastapi import HTTPException
from langchain_core.tools import tool


@tool
async def fetch_repo_metadata(owner: str, repo: str) -> dict[str, Any]:
    """Fetch repository metadata from GitHub including stars, primary language, and description.

    Args:
        owner: Repository owner or organization username (e.g. 'psf').
        repo: Repository name (e.g. 'requests').

    Returns:
        dict: Repository details containing github_id, full_name, url, description, stars, and language.
    """
    try:
        url = f"https://github.com/{owner}/{repo}"
        return await fetch_repository(url)
    except HTTPException as exc:
        return {"error": exc.detail, "status_code": exc.status_code}
    except Exception as exc:
        return {"error": f"Failed to fetch metadata for {owner}/{repo}: {str(exc)}"}


@tool
async def inspect_file_tree(owner: str, repo: str, prefix: str = "") -> list[dict[str, Any]]:
    """Inspect the repository file tree to locate source files, optionally filtered by directory prefix.

    Args:
        owner: Repository owner username (e.g. 'tiangolo').
        repo: Repository name (e.g. 'fastapi').
        prefix: Optional directory path prefix to filter files (e.g. 'app/' or 'src/utils').

    Returns:
        list[dict]: List of indexable file entries matching the prefix, each containing 'path' and 'size'.
    """
    try:
        entries = await fetch_repo_tree(owner, repo)
        if not prefix:
            return entries

        normalized_prefix = prefix.strip("/")
        return [
            entry
            for entry in entries
            if entry["path"].startswith(f"{normalized_prefix}/") or entry["path"] == normalized_prefix
        ]
    except HTTPException as exc:
        return [{"error": exc.detail, "status_code": exc.status_code}]
    except Exception as exc:
        return [{"error": f"Failed to inspect file tree: {str(exc)}"}]


@tool
async def read_source_file(
    owner: str,
    repo: str,
    file_path: str,
    start_line: int = 1,
    end_line: int = 100,
) -> str:
    """Read source code from a file in a GitHub repository within specified line bounds.

    Args:
        owner: Repository owner username.
        repo: Repository name.
        file_path: Relative path to the file in the repository (e.g. 'app/main.py').
        start_line: 1-indexed first line to read (default 1).
        end_line: 1-indexed last line to read (default 100, maximum 200 lines per call).

    Returns:
        str: Sliced source code with line numbers, or an error message if the file is unavailable.
    """
    try:
        content = await fetch_file_content(owner, repo, file_path)
    except HTTPException as exc:
        return f"Error reading file '{file_path}': {exc.detail}"
    except Exception as exc:
        return f"Error reading file '{file_path}': {str(exc)}"

    if content is None:
        return f"Error: Could not read or decode file '{file_path}'. It may be binary, deleted, or exceeding size limits."

    lines = content.splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return f"File '{file_path}' is empty."

    start = max(1, start_line)
    # Limit max slice to 200 lines to prevent LLM context saturation
    end = min(total_lines, max(start, min(end_line, start + 199)))

    selected = lines[start - 1 : end]
    numbered = [f"{i} | {line}" for i, line in enumerate(selected, start=start)]
    header = f"File: {file_path} (Lines {start}-{end} of {total_lines})\n"
    return header + "\n".join(numbered)
