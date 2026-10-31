"""Streaming chat & durable session management router for RepoPilot.

This router provides endpoints for:
1. POST   /api/chat/{repository_id}/sessions                   — Create a new chat session
2. GET    /api/chat/{repository_id}/sessions                   — List past sessions for a repo
3. GET    /api/chat/sessions/{session_id}/history              — Retrieve thread history (resume chat)
4. DELETE /api/chat/sessions/{session_id}                      — Delete a chat session
5. POST   /api/chat/{repository_id}/sessions/{session_id}/stream — Stream agent execution via SSE
"""

import json
import logging
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.agent.checkpoint import get_checkpointer
from app.agent.graph import create_agent_graph
from app.database import get_db
from app.github_service import parse_github_url
from app.models import ChatSession
from app.routers.helpers import get_repository_or_404, require_indexed
from app.schemas import (
    ChatHistoryResponse,
    ChatMessageItem,
    ChatSessionCreate,
    ChatSessionResponse,
    ChatStreamRequest,
)

logger = logging.getLogger(__name__)

# Default LangGraph execution recursion limit per turn (can be overridden via AGENT_RECURSION_LIMIT)
DEFAULT_RECURSION_LIMIT: int = int(os.getenv("AGENT_RECURSION_LIMIT", "50"))

router = APIRouter(
    prefix="/api/chat",
    tags=["Chat & Streaming"],
)


def _extract_text_content(content: Any) -> str:
    """Extract clean string text from string or structured content block list."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and "text" in p:
                parts.append(str(p["text"]))
        return "\n".join(parts)
    return str(content) if content is not None else ""


def _get_session_or_404(db: Session, session_id: str) -> ChatSession:
    """Fetch a chat session by ID or raise HTTP 404."""
    session = db.get(ChatSession, session_id)
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chat session '{session_id}' not found.",
        )
    return session


@router.post(
    "/{repository_id}/sessions",
    response_model=ChatSessionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new conversation session",
)
def create_session(
    repository_id: int,
    payload: ChatSessionCreate = ChatSessionCreate(),
    db: Session = Depends(get_db),
) -> ChatSession:
    """Create a new persistent conversation thread for a repository."""
    repo = get_repository_or_404(db, repository_id)

    title = payload.title or "New Conversation"
    session_id = str(uuid.uuid4())
    chat_session = ChatSession(
        id=session_id,
        repository_id=repo.id,
        title=title,
    )
    db.add(chat_session)
    db.commit()
    db.refresh(chat_session)

    logger.info("Created chat session '%s' for repo %d", session_id, repo.id)
    return chat_session


@router.get(
    "/{repository_id}/sessions",
    response_model=list[ChatSessionResponse],
    summary="List past chat sessions for a repository",
)
def list_sessions(
    repository_id: int,
    db: Session = Depends(get_db),
) -> list[ChatSession]:
    """Retrieve all chat sessions for a repository, ordered by newest activity first."""
    get_repository_or_404(db, repository_id)
    statement = (
        select(ChatSession).where(ChatSession.repository_id == repository_id).order_by(ChatSession.updated_at.desc())
    )
    return list(db.scalars(statement).all())


@router.get(
    "/sessions/{session_id}/history",
    response_model=ChatHistoryResponse,
    summary="Retrieve full conversation history to resume a chat",
)
async def get_session_history(
    session_id: str,
    db: Session = Depends(get_db),
) -> ChatHistoryResponse:
    """Retrieve the stored message history for a conversation session from checkpoints."""
    session = _get_session_or_404(db, session_id)

    # Initialize graph with active checkpointer to inspect stored state
    checkpointer = get_checkpointer()
    graph = create_agent_graph(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": session_id}}

    state = await graph.aget_state(config)
    raw_messages = state.values.get("messages", []) if state else []

    formatted_messages: list[ChatMessageItem] = []
    for msg in raw_messages:
        # Filter out system messages from client display
        if isinstance(msg, SystemMessage):
            continue

        if isinstance(msg, HumanMessage):
            formatted_messages.append(
                ChatMessageItem(
                    id=getattr(msg, "id", None),
                    role="user",
                    content=_extract_text_content(msg.content),
                )
            )
        elif isinstance(msg, AIMessage):
            tool_calls = getattr(msg, "tool_calls", None) or None
            formatted_messages.append(
                ChatMessageItem(
                    id=getattr(msg, "id", None),
                    role="assistant",
                    content=_extract_text_content(msg.content),
                    tool_calls=tool_calls,
                )
            )
        elif isinstance(msg, ToolMessage):
            formatted_messages.append(
                ChatMessageItem(
                    id=getattr(msg, "id", None),
                    role="tool",
                    content=_extract_text_content(msg.content),
                )
            )

    return ChatHistoryResponse(
        session_id=session.id,
        repository_id=session.repository_id,
        title=session.title,
        messages=formatted_messages,
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a conversation session",
)
def delete_session(
    session_id: str,
    db: Session = Depends(get_db),
) -> None:
    """Remove a conversation session and its association."""
    session = _get_session_or_404(db, session_id)
    db.delete(session)
    db.commit()
    logger.info("Deleted chat session '%s'", session_id)


@router.post(
    "/{repository_id}/sessions/{session_id}/stream",
    summary="Stream conversation events token-by-token (SSE)",
)
async def stream_chat(
    repository_id: int,
    session_id: str,
    payload: ChatStreamRequest,
    db: Session = Depends(get_db),
) -> StreamingResponse:
    """Stream real-time agent generation and tool execution events using Server-Sent Events (SSE)."""
    repo = get_repository_or_404(db, repository_id)
    require_indexed(repo)
    chat_session = _get_session_or_404(db, session_id)

    # Parse repo owner and name for state context
    owner, repo_name = parse_github_url(repo.url)

    # Auto-update session title on first turn if it still has default title
    if chat_session.title == "New Conversation":
        preview = payload.message.strip()
        chat_session.title = preview[:40] + ("…" if len(preview) > 40 else "")
    chat_session.updated_at = func.now()
    db.commit()

    checkpointer = get_checkpointer()
    graph = create_agent_graph(checkpointer=checkpointer)

    recursion_limit = int(os.getenv("AGENT_RECURSION_LIMIT", str(DEFAULT_RECURSION_LIMIT)))

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": recursion_limit,
        "run_name": f"repopilot-chat:{owner}/{repo_name}",
        "tags": ["repopilot", "chat", f"repo:{repo.full_name}", f"repo_id:{repository_id}"],
        "metadata": {
            "session_id": session_id,
            "repository_id": repository_id,
            "repository_name": repo.full_name,
            "repository_url": repo.url,
            "repo_owner": owner,
            "repo_name": repo_name,
        },
    }

    initial_input = {
        "messages": [HumanMessage(content=payload.message)],
        "repo_id": repository_id,
        "repo_owner": owner,
        "repo_name": repo_name,
    }

    async def event_generator() -> AsyncIterator[str]:
        """Generate formatted SSE event chunks."""
        try:
            async for event in graph.astream_events(initial_input, config=config, version="v2"):
                event_type = event.get("event")

                # 1. LLM token delta streaming
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and chunk.content:
                        delta = _extract_text_content(chunk.content)
                        if delta:
                            data = json.dumps({"type": "token", "delta": delta})
                            yield f"data: {data}\n\n"

                # 2. Tool invocation initiated
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    data = json.dumps({"type": "tool_start", "tool": tool_name, "args": tool_input})
                    yield f"data: {data}\n\n"

                # 3. Tool execution completed
                elif event_type == "on_tool_end":
                    tool_name = event.get("name", "")
                    raw_output = event.get("data", {}).get("output", "")
                    # Limit output length sent in stream payload for bandwidth efficiency
                    output_str = str(raw_output)[:400]
                    data = json.dumps({"type": "tool_end", "tool": tool_name, "output": output_str})
                    yield f"data: {data}\n\n"

            # Terminal completion event
            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except GraphRecursionError as exc:
            logger.warning(
                "LangGraph recursion limit (%d) reached for session %s: %s",
                recursion_limit,
                session_id,
                exc,
            )
            friendly_err = (
                f"The agent reached its maximum step limit ({recursion_limit}) while exploring the repository. "
                "You can increase AGENT_RECURSION_LIMIT in backend/.env, or ask a more specific question."
            )
            err_data = json.dumps({"type": "error", "error": friendly_err})
            yield f"data: {err_data}\n\n"
        except Exception as exc:
            logger.exception("Error during chat stream for session %s: %s", session_id, exc)
            err_data = json.dumps({"type": "error", "error": str(exc)})
            yield f"data: {err_data}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
