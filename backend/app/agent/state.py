"""Typed state schema for the RepoPilot LangGraph agent."""

from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Represents the complete state of a conversation in RepoPilot.

    Attributes:
        messages: Chronological sequence of chat messages (user, assistant, tool calls, and tool outputs).
                  Uses add_messages reducer to handle append and in-place message update semantics.
        repo_id: Internal database ID of the repository being inspected (if registered in RepoPilot).
        repo_owner: GitHub owner or organization username (e.g. 'tiangolo').
        repo_name: GitHub repository name (e.g. 'fastapi').
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    repo_id: int | None
    repo_owner: str | None
    repo_name: str | None
