"""Unit tests for GitHub service URL validation and path handling."""

import pytest
from app.github_service import _get_file_extension, parse_github_url
from fastapi import HTTPException


def test_get_file_extension():
    assert _get_file_extension("src/app.py") == ".py"
    assert _get_file_extension("README.MD") == ".md"
    assert _get_file_extension("Dockerfile") == ""
    assert _get_file_extension("component.test.tsx") == ".tsx"


def test_parse_valid_github_urls():
    owner, repo = parse_github_url("https://github.com/fastapi/fastapi")
    assert owner == "fastapi"
    assert repo == "fastapi"

    owner, repo = parse_github_url("http://github.com/psf/requests.git")
    assert owner == "psf"
    assert repo == "requests"

    owner, repo = parse_github_url("https://github.com/facebook/react/")
    assert owner == "facebook"
    assert repo == "react"


def test_parse_invalid_github_urls_rejected():
    # Non-GitHub domain (SSRF protection)
    with pytest.raises(HTTPException) as exc_info:
        parse_github_url("https://gitlab.com/owner/repo")
    assert exc_info.value.status_code == 422

    # Suspicious IP or local URL
    with pytest.raises(HTTPException) as exc_info:
        parse_github_url("https://127.0.0.1/owner/repo")
    assert exc_info.value.status_code == 422

    # Invalid scheme
    with pytest.raises(HTTPException) as exc_info:
        parse_github_url("ftp://github.com/owner/repo")
    assert exc_info.value.status_code == 422

    # Custom port
    with pytest.raises(HTTPException) as exc_info:
        parse_github_url("https://github.com:8080/owner/repo")
    assert exc_info.value.status_code == 422

    # Directory traversal
    with pytest.raises(HTTPException) as exc_info:
        parse_github_url("https://github.com/../repo")
    assert exc_info.value.status_code == 422
