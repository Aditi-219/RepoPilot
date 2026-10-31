"""Unit and integration tests for the LangGraph chat & streaming router."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.checkpoint import set_checkpointer
from app.database import Base, SessionLocal, engine
from app.main import app
from app.models import Repository


@pytest.fixture(scope="session", autouse=True)
def ensure_db_tables():
    """Ensure all SQLAlchemy tables exist in PostgreSQL for tests."""
    Base.metadata.create_all(bind=engine)


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Test client running within FastAPI lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_create_and_list_sessions(client):
    """Test creating a chat session and listing it under the repository."""
    # Ensure a test repository exists in the DB
    from app.database import SessionLocal

    with SessionLocal() as db:
        repo = db.query(Repository).filter_by(full_name="test/repo").first()
        if not repo:
            repo = Repository(
                github_id=99999,
                full_name="test/repo",
                url="https://github.com/test/repo",
                status="completed",
                stars=10,
            )
            db.add(repo)
            db.commit()
            db.refresh(repo)
        repo_id = repo.id

    # 1. Create a new session
    res = client.post(
        f"/api/chat/{repo_id}/sessions",
        json={"title": "Test Chat Session"},
    )
    assert res.status_code == 201
    data = res.json()
    assert data["repository_id"] == repo_id
    assert data["title"] == "Test Chat Session"
    session_id = data["id"]
    assert len(session_id) == 36  # UUID length

    # 2. List sessions for the repository
    list_res = client.get(f"/api/chat/{repo_id}/sessions")
    assert list_res.status_code == 200
    sessions = list_res.json()
    assert any(s["id"] == session_id for s in sessions)

    # 3. Retrieve history (initially empty)
    hist_res = client.get(f"/api/chat/sessions/{session_id}/history")
    assert hist_res.status_code == 200
    hist = hist_res.json()
    assert hist["session_id"] == session_id
    assert hist["messages"] == []

    # 4. Delete the session
    del_res = client.delete(f"/api/chat/sessions/{session_id}")
    assert del_res.status_code == 204

    # 5. Verify deleted session returns 404
    hist_after = client.get(f"/api/chat/sessions/{session_id}/history")
    assert hist_after.status_code == 404


def test_get_session_history_with_checkpointed_messages(client):
    """Verify history endpoint returns messages formatted from checkpointer state."""
    from app.database import SessionLocal

    with SessionLocal() as db:
        repo = db.query(Repository).filter_by(full_name="test/repo").first()
        if not repo:
            repo = Repository(
                github_id=99999,
                full_name="test/repo",
                url="https://github.com/test/repo",
                status="completed",
                stars=10,
            )
            db.add(repo)
            db.commit()
            db.refresh(repo)
        repo_id = repo.id

    # Create session
    create_res = client.post(f"/api/chat/{repo_id}/sessions")
    assert create_res.status_code == 201
    session_id = create_res.json()["id"]

    # Use MemorySaver to simulate stored checkpoint state
    mem_checkpointer = MemorySaver()
    set_checkpointer(mem_checkpointer)

    from app.agent.graph import create_agent_graph

    class StaticModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, config=None):
            return AIMessage(content="Here is the architecture overview.")

    graph = create_agent_graph(model=StaticModel(), checkpointer=mem_checkpointer)

    import asyncio

    asyncio.run(
        graph.ainvoke(
            {"messages": [HumanMessage(content="Explain the architecture")]},
            config={"configurable": {"thread_id": session_id}},
        )
    )

    # Fetch history via API
    hist_res = client.get(f"/api/chat/sessions/{session_id}/history")
    assert hist_res.status_code == 200
    messages = hist_res.json()["messages"]

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Explain the architecture"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Here is the architecture overview."

    # Clean up
    client.delete(f"/api/chat/sessions/{session_id}")
    set_checkpointer(None)


def test_stream_chat_sse_endpoint(client):
    """Test SSE streaming response from the chat streaming endpoint."""
    from unittest.mock import patch

    from app.agent.graph import create_agent_graph as real_create_graph

    with SessionLocal() as db:
        repo = db.query(Repository).filter_by(full_name="test/repo").first()
        if not repo:
            repo = Repository(
                github_id=99999,
                full_name="test/repo",
                url="https://github.com/test/repo",
                status="completed",
                stars=10,
            )
            db.add(repo)
            db.commit()
            db.refresh(repo)
        repo_id = repo.id

    # Create session
    create_res = client.post(f"/api/chat/{repo_id}/sessions")
    session_id = create_res.json()["id"]

    class StreamMockModel:
        def bind_tools(self, tools):
            return self

        async def ainvoke(self, messages, config=None):
            return AIMessage(content="Streaming copilot response.")

    mock_graph = real_create_graph(model=StreamMockModel(), checkpointer=MemorySaver())

    # Stream response with mocked graph
    with patch("app.routers.chat.create_agent_graph", return_value=mock_graph):
        stream_res = client.post(
            f"/api/chat/{repo_id}/sessions/{session_id}/stream",
            json={"message": "Hello copilot!"},
        )
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]

        body = stream_res.text
        assert "data: " in body
        assert '{"type": "done"}' in body

    # Clean up
    client.delete(f"/api/chat/sessions/{session_id}")


def test_stream_chat_handles_recursion_error_gracefully(client):
    """Verify that GraphRecursionError yields a friendly error event instead of a raw crash."""
    from unittest.mock import MagicMock, patch

    from langgraph.errors import GraphRecursionError

    with SessionLocal() as db:
        repo = db.query(Repository).filter_by(full_name="test/repo").first()
        repo_id = repo.id

    create_res = client.post(f"/api/chat/{repo_id}/sessions")
    session_id = create_res.json()["id"]

    mock_graph = MagicMock()

    async def mock_astream_events(*args, **kwargs):
        raise GraphRecursionError("Recursion limit of 15 reached without hitting a stop condition.")
        yield  # Make it an async generator

    mock_graph.astream_events = mock_astream_events

    with patch("app.routers.chat.create_agent_graph", return_value=mock_graph):
        stream_res = client.post(
            f"/api/chat/{repo_id}/sessions/{session_id}/stream",
            json={"message": "Deep search query"},
        )
        assert stream_res.status_code == 200
        assert "text/event-stream" in stream_res.headers["content-type"]

        body = stream_res.text
        assert '{"type": "error"' in body
        assert "maximum step limit" in body
        assert "AGENT_RECURSION_LIMIT" in body

    client.delete(f"/api/chat/sessions/{session_id}")
