"""Unit tests for RepoPilot LangGraph agent workflow."""

from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from app.agent.graph import create_agent_graph


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
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="List files in src")],
                "repo_id": 1,
                "repo_owner": "pallets",
                "repo_name": "flask",
            }
        )

        assert mock_fetch.awaited
        # Verify the final message is the assistant's concluding response
        assert result["messages"][-1].content == "Found src/flask/app.py in the repository."
        # Verify total messages in thread (Human -> ToolCall AIMessage -> ToolMessage -> Final AIMessage)
        assert len(result["messages"]) == 4


@pytest.mark.asyncio
async def test_agent_graph_durable_checkpoint_memory():
    """Verify multi-turn memory persistence using thread_id checkpointer."""
    model = MockChatModel(
        [
            AIMessage(content="Your project uses FastAPI."),
            AIMessage(content="Yes, you previously asked about FastAPI."),
        ]
    )

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


@pytest.mark.asyncio
async def test_agent_graph_concludes_near_recursion_limit():
    """Verify agent terminates gracefully with a summary when approaching recursion limit."""

    class LoopingModel:
        def __init__(self, with_tools: bool = False):
            self.with_tools = with_tools
            self.invocations = 0

        def bind_tools(self, tools):
            return LoopingModel(with_tools=True)

        async def ainvoke(self, messages, config=None):
            self.invocations += 1
            # Check if concluding instruction was passed
            for m in messages:
                if "maximum tool exploration step limit" in str(getattr(m, "content", "")):
                    return AIMessage(content="Final summary: gathered all available dependency details.")
            if self.with_tools:
                return AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "inspect_file_tree",
                            "args": {"owner": "pallets", "repo": "flask", "prefix": "src"},
                            "id": f"call_{self.invocations}",
                        }
                    ],
                )
            return AIMessage(content="Final response.")

    with patch("app.tools.github_tools.fetch_repo_tree", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = [{"path": "src/app.py", "size": 100}]
        model = LoopingModel(with_tools=False)
        graph = create_agent_graph(model=model)

        # Set a tight recursion limit of 6
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What dependencies are used?")],
                "repo_id": 1,
                "repo_owner": "pallets",
                "repo_name": "flask",
            },
            config={"recursion_limit": 6},
        )

        assert mock_fetch.awaited
        # Graph must finish cleanly without raising GraphRecursionError
        assert "Final summary" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_agent_graph_recovers_dangling_tool_calls():
    """Verify that unfulfilled tool calls from an interrupted turn are resolved safely."""
    model = MockChatModel([AIMessage(content="Recovered and answered.")])
    graph = create_agent_graph(model=model)

    # State containing a dangling AIMessage with a tool_call and no ToolMessage response
    dangling_tool_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "inspect_file_tree",
                "args": {"owner": "pallets", "repo": "flask"},
                "id": "interrupted_call_1",
            }
        ],
    )
    initial_state = {
        "messages": [
            HumanMessage(content="Previous prompt"),
            dangling_tool_call,
        ],
        "repo_id": 1,
        "repo_owner": "pallets",
        "repo_name": "flask",
    }

    result = await graph.ainvoke(initial_state)
    assert result["messages"][-1].content == "Recovered and answered."
    # Check that model received a ToolMessage resolving the interrupted call
    recorded = model.recorded_messages[0]
    has_resolved_tool_msg = any(getattr(m, "tool_call_id", None) == "interrupted_call_1" for m in recorded)
    assert has_resolved_tool_msg
