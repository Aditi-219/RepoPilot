"""GitHub integration service for validating and fetching repository metadata.

This service interacts with the public GitHub REST API to verify repository
existence and retrieve core details (stars, language, description, IDs).
"""

import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException

# GitHub REST API v3 base endpoint
GITHUB_API = "https://api.github.com"


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

    # Required and recommended GitHub headers
    headers: dict[str, str] = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "RepoPilot-Agent",
    }

    # Optional: If GITHUB_TOKEN is set in .env, use it to increase rate limit from 60/hr to 5,000/hr
    github_token = os.getenv("GITHUB_TOKEN")
    if github_token:
        headers["Authorization"] = f"Bearer {github_token}"

    try:
        # Using async httpx client with follow_redirects=True so GitHub 301/302 redirects
        # (e.g. for renamed repos or organization slug changes) are followed automatically.
        # Timeout set to 25.0s to allow for DNS resolution and TLS handshakes.
        async with httpx.AsyncClient(timeout=25.0, follow_redirects=True) as client:
            response = await client.get(api_url, headers=headers)

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