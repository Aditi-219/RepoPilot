"""GitHub integration service for validating and fetching repository metadata.

This service interacts with the public GitHub REST API to verify repository
existence and retrieve core details (stars, language, description, IDs).

Phase 2 additions:
- fetch_repo_tree(): walks the full file tree of a repository
- fetch_file_content(): downloads and decodes a single file's content
"""

import base64
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

# GitHub REST API v3 base endpoint
GITHUB_API = "https://api.github.com"

# File extensions RepoPilot will index.
# Binary files, images, lock files, and build artifacts are excluded.
SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".go", ".rb", ".rs", ".cpp",
    ".c", ".h", ".cs", ".php", ".swift",
    ".kt", ".md", ".yml", ".yaml", ".toml",
    ".html", ".css", ".sh", ".bash",
})

# Skip files larger than this many bytes to avoid indexing
# generated, minified, or vendor files that add noise.
# Reads MAX_FILE_SIZE_KB from the environment (default: 100 KB).
MAX_FILE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_KB", "100")) * 1024


def _build_headers() -> dict[str, str]:
    """Build standard GitHub API headers, optionally with an auth token.

    Adding a GITHUB_TOKEN raises the rate limit from 60 req/hr to 5,000 req/hr.
    """
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "RepoPilot-Agent",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _get_file_extension(path: str) -> str:
    """Extract the lowercased dot-prefixed extension from a file path.

    Args:
        path: A file path string (e.g., 'src/utils/parser.py').

    Returns:
        Lowercased extension with leading dot (e.g., '.py'), or '' if none.
    """
    return "." + path.rsplit(".", 1)[-1].lower() if "." in path else ""


def parse_github_url(url: str) -> tuple[str, str]:
    """Parse and validate a GitHub repository URL into (owner, repo).

    Performs strict checks:
    - Protocol must be http or https
    - Host must be 'github.com' (disallowing SSRF or non-GitHub hosts)
    - Disallows embedded credentials or custom ports
    - Path must match '/owner/repo'
    - Owner and repository names must follow GitHub's allowed naming rules

    Args:
        url: Full GitHub URL string (e.g., 'https://github.com/fastapi/fastapi')

    Returns:
        tuple[str, str]: Extracted (owner, repository_name)

    Raises:
        HTTPException(422): If the URL format is invalid or suspicious.
    """
    parsed = urlparse(url)

    # 1. Verify scheme
    if parsed.scheme not in {"https", "http"}:
        raise HTTPException(
            status_code=422,
            detail="Use a valid GitHub repository URL (http or https).",
        )

    # 2. Restrict to github.com only
    if parsed.hostname != "github.com":
        raise HTTPException(
            status_code=422,
            detail="Only github.com repository URLs are supported.",
        )

    # 3. Disallow port numbers or embedded user credentials
    if parsed.port is not None or parsed.username or parsed.password:
        raise HTTPException(
            status_code=422,
            detail="Invalid GitHub repository URL format.",
        )

    # 4. Extract path segments: '/owner/repo' -> ['owner', 'repo']
    parts = parsed.path.strip("/").split("/")
    if len(parts) != 2:
        raise HTTPException(
            status_code=422,
            detail="URL must look like https://github.com/owner/repo",
        )

    owner, repo = parts
    # Clean up optional '.git' suffix (e.g. 'repo.git' -> 'repo')
    repo = repo.removesuffix(".git")

    # 5. Validate characters: alphanumeric, dashes, dots, underscores
    valid_name = re.compile(r"^[A-Za-z0-9_.-]+$")
    if not valid_name.fullmatch(owner) or not valid_name.fullmatch(repo):
        raise HTTPException(
            status_code=422,
            detail="Invalid repository owner or repository name.",
        )

    # Prevent directory traversal patterns
    if owner in {".", ".."} or repo in {".", ".."}:
        raise HTTPException(
            status_code=422,
            detail="Invalid repository owner or repository name.",
        )

    return owner, repo


async def fetch_repository(url: str) -> dict[str, Any]:
    """Fetch repository metadata from GitHub's REST API asynchronously.

    Calls: GET https://api.github.com/repos/{owner}/{repo}

    Args:
        url: Validated GitHub repository URL.

    Returns:
        dict: Normalized repository metadata dictionary ready for database insertion:
              - github_id (int)
              - full_name (str)
              - url (str)
              - description (str | None)
              - stars (int)
              - language (str | None)

    Raises:
        HTTPException:
            - 404: Repository does not exist or is private.
            - 403: GitHub API rate limit exceeded.
            - 502: GitHub unreachable or bad gateway.
            - 504: GitHub API timed out.
    """
    owner, repo = parse_github_url(url)
    api_url = f"{GITHUB_API}/repos/{owner}/{repo}"

    try:
        # Using async httpx client with follow_redirects=True so GitHub 301/302 redirects
        # (e.g. for renamed repos or organization slug changes) are followed automatically.
        # Timeout set to 25.0s to allow for DNS resolution and TLS handshakes.
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            response = await client.get(api_url, headers=_build_headers())

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="GitHub API request timed out. Please try again later.",
        )
    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Could not connect to GitHub API ({type(exc).__name__}). Please check network connectivity.",
        )

    # Handle GitHub API specific HTTP response statuses
    if response.status_code == 404:
        raise HTTPException(
            status_code=404,
            detail="Repository not found or is not publicly accessible.",
        )
    if response.status_code == 403:
        raise HTTPException(
            status_code=503,
            detail="GitHub API rate limit or access restriction reached. Add a GITHUB_TOKEN to .env to raise limits.",
        )
    if response.status_code != 200:
        error_detail = "Unexpected response from GitHub"
        try:
            github_error = response.json()
            if isinstance(github_error, dict) and "message" in github_error:
                error_detail = github_error["message"]
        except Exception:
            pass

        raise HTTPException(
            status_code=502,
            detail=f"GitHub returned HTTP {response.status_code}: {error_detail}",
        )

    data = response.json()

    # Extract and normalize only the fields required by RepoPilot
    return {
        "github_id": data["id"],
        "full_name": data["full_name"],
        "url": data["html_url"],
        "description": data.get("description"),
        "stars": data.get("stargazers_count", 0),
        "language": data.get("language"),
    }


# -----------------------------------------------------------------------------
# Phase 2 — File Tree & Content Fetching
# -----------------------------------------------------------------------------

async def fetch_repo_tree(owner: str, repo: str) -> list[dict[str, Any]]:
    """Fetch the full recursive file tree of a GitHub repository.

    Uses the Git Trees API with `recursive=1` to get all file paths in a single
    request instead of walking directory by directory.

    Calls: GET /repos/{owner}/{repo}/git/trees/HEAD?recursive=1

    Args:
        owner: Repository owner/organization name.
        repo: Repository name.

    Returns:
        list[dict]: Filtered list of file entries, each containing:
                    - 'path' (str): relative file path (e.g. 'src/utils/parser.py')
                    - 'size' (int): file size in bytes
                    Only blobs (files) with supported extensions are included.

    Raises:
        HTTPException: On network errors, rate limits, or missing default branch.
    """
    api_url = f"{GITHUB_API}/repos/{owner}/{repo}/git/trees/HEAD?recursive=1"

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(api_url, headers=_build_headers())
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="GitHub API timed out fetching file tree.")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"Network error fetching file tree: {exc}")

    if response.status_code == 404:
        raise HTTPException(status_code=404, detail="Repository not found or has no commits.")
    if response.status_code == 403:
        raise HTTPException(status_code=503, detail="GitHub rate limit reached. Set GITHUB_TOKEN in .env.")
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail=f"GitHub returned HTTP {response.status_code} for file tree.")

    data = response.json()

    # Filter: only include blob (file) entries with supported extensions
    # and within the file size limit
    supported_files: list[dict[str, Any]] = []
    for item in data.get("tree", []):
        if item.get("type") != "blob":
            continue  # skip directories and submodules

        path: str = item.get("path", "")
        size: int = item.get("size", 0)

        # Check extension using shared helper
        ext = _get_file_extension(path)
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        # Skip files that are too large
        if size > MAX_FILE_BYTES:
            continue

        supported_files.append({"path": path, "size": size})

    return supported_files


async def fetch_file_content(owner: str, repo: str, path: str) -> str | None:
    """Download and decode the content of a single file from a GitHub repository.

    Uses the Contents API which returns base64-encoded file content.

    Calls: GET /repos/{owner}/{repo}/contents/{path}

    Args:
        owner: Repository owner/organization name.
        repo: Repository name.
        path: Relative file path within the repository.

    Returns:
        str | None: Decoded UTF-8 text content of the file, or None if the
                    file cannot be decoded (e.g., it is actually a binary file).

    Raises:
        HTTPException: On network errors or rate limits.
    """
    api_url = f"{GITHUB_API}/repos/{owner}/{repo}/contents/{path}"

    try:
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
            response = await client.get(api_url, headers=_build_headers())
    except httpx.TimeoutException:
        # Non-fatal: skip this file and continue indexing
        return None
    except httpx.RequestError:
        return None

    if response.status_code == 403:
        raise HTTPException(status_code=503, detail="GitHub rate limit reached. Set GITHUB_TOKEN in .env.")

    if response.status_code != 200:
        # Non-fatal: some files may have been moved or deleted since tree was fetched
        return None

    data = response.json()

    # GitHub returns content as base64-encoded string with possible newlines
    encoded = data.get("content", "")
    if not encoded:
        return None

    try:
        decoded_bytes = base64.b64decode(encoded)
        return decoded_bytes.decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        # Skip binary files that slipped through extension filtering
        return None
