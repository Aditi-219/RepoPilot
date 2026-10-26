"""Unit tests for RepoPilot LangGraph agent workflow."""

from unittest.mock import AsyncMock, patch

import pytest
from app.agent.graph import create_agent_graph
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver


class MockChatModel:
    """Mock chat model for deterministic graph testing without live LLM calls."""

    def __init__(self, responses: list[AIMessage]):
        self.responses = list(responses)
        self.idx = 0
        self.recorded_messages: list[list] = []

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, config=None):
        self.recorded_messages.append(list(messages))
        if self.idx < len(self.responses):
            resp = self.responses[self.idx]
            self.idx += 1
            return resp
        return AIMessage(content="Finished")


@pytest.mark.asyncio
async def test_agent_graph_direct_answer():
    """Verify single-turn agent execution without tool calls."""
    model = MockChatModel([AIMessage(content="RepoPilot copilot is ready.")])
    graph = create_agent_graph(model=model)

    initial_state = {
        "messages": [HumanMessage(content="Hello!")],
        "repo_id": 1,
        "repo_owner": "tiangolo",
        "repo_name": "fastapi",
    }
    result = await graph.ainvoke(initial_state)

    assert len(result["messages"]) >= 2
    assert result["messages"][-1].content == "RepoPilot copilot is ready."
    assert isinstance(result["messages"][-1], AIMessage)


@pytest.mark.asyncio
async def test_agent_graph_injects_repo_context():
    """Verify repo_owner and repo_name are injected into the prepended SystemMessage."""
    model = MockChatModel([AIMessage(content="Answer")])
    graph = create_agent_graph(model=model)

    initial_state = {
        "messages": [HumanMessage(content="What is this repo?")],
        "repo_id": 42,
        "repo_owner": "pallets",
        "repo_name": "flask",
    }
    await graph.ainvoke(initial_state)

    # Inspect messages received by the model on its first invocation
    first_call_messages = model.recorded_messages[0]
    system_msg = first_call_messages[0]
    assert isinstance(system_msg, SystemMessage)
    assert "pallets/flask" in system_msg.content


@pytest.mark.asyncio
async def test_agent_graph_tool_call_loop():
    """Verify multi-step tool execution loop: agent -> tools -> agent -> END."""
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "inspect_file_tree",
                "args": {"owner": "pallets", "repo": "flask", "prefix": "src"},
                "id": "call_inspect_1",
            }
        ],
    )
    final_response = AIMessage(content="Found src/flask/app.py in the repository.")

    model = MockChatModel([tool_call_response, final_response])

    mock_tree = [{"path": "src/flask/app.py", "size": 1024}]
    with patch("app.tools.github_tools.fetch_repo_tree", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = mock_tree

        graph = create_agent_graph(model=model)
        result = await graph.ainvoke({
            "messages": [HumanMessage(content="List files in src")],
            "repo_id": 1,
            "repo_owner": "pallets",
            "repo_name": "flask",
        })

        assert mock_fetch.awaited
        # Verify the final message is the assistant's concluding response
        assert result["messages"][-1].content == "Found src/flask/app.py in the repository."
        # Verify total messages in thread (Human -> ToolCall AIMessage -> ToolMessage -> Final AIMessage)
        assert len(result["messages"]) == 4


@pytest.mark.asyncio
async def test_agent_graph_durable_checkpoint_memory():
    """Verify multi-turn memory persistence using thread_id checkpointer."""
    model = MockChatModel([
        AIMessage(content="Your project uses FastAPI."),
        AIMessage(content="Yes, you previously asked about FastAPI."),
    ])

    checkpointer = MemorySaver()
    graph = create_agent_graph(model=model, checkpointer=checkpointer)
    config = {"configurable": {"thread_id": "test_session_1"}}

    # Turn 1
    await graph.ainvoke(
        {"messages": [HumanMessage(content="What framework does this project use?")]},
        config=config,
    )

    # Turn 2 with same thread_id
    res2 = await graph.ainvoke(
        {"messages": [HumanMessage(content="Do you remember what I just asked?")]},
        config=config,
    )

    # Verify state accumulated messages from both turns
    messages = res2["messages"]
    assert len(messages) == 4  # Human1, AI1, Human2, AI2
    assert messages[-1].content == "Yes, you previously asked about FastAPI."
