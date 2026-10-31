"""Unit tests for Pydantic request/response schemas."""

import pytest
from pydantic import ValidationError

from app.schemas import AskQuery, RepositoryCreate, SearchQuery, StatusUpdate


def test_repository_create_valid():
    repo = RepositoryCreate(url="https://github.com/tiangolo/fastapi")
    assert str(repo.url) == "https://github.com/tiangolo/fastapi"


def test_repository_create_invalid_url():
    with pytest.raises(ValidationError):
        RepositoryCreate(url="not-a-valid-url")


def test_search_query_validation():
    # Valid query
    query = SearchQuery(query="authentication logic", limit=10, rerank=True)
    assert query.query == "authentication logic"
    assert query.limit == 10

    # Limit out of bounds (< 1)
    with pytest.raises(ValidationError):
        SearchQuery(query="test", limit=0)

    # Limit out of bounds (> 50)
    with pytest.raises(ValidationError):
        SearchQuery(query="test", limit=100)


def test_ask_query_validation():
    query = AskQuery(question="Where is DB session created?", top_k=5)
    assert query.question == "Where is DB session created?"
    assert query.top_k == 5

    with pytest.raises(ValidationError):
        AskQuery(question="test", top_k=0)


def test_status_update_validation():
    valid = StatusUpdate(status="completed")
    assert valid.status == "completed"

    with pytest.raises(ValidationError):
        StatusUpdate(status="invalid_status")
