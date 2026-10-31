"""Core LangGraph workflow for the RepoPilot chatbot agent.

Defines the typed state graph, model binding, tool integration, and conditional
routing edges for codebase intelligence interactions.
"""

import os
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import AgentState
from app.tools.github_tools import fetch_repo_metadata, inspect_file_tree, read_source_file

# Default toolset available to the agent
AGENT_TOOLS = [
    fetch_repo_metadata,
    inspect_file_tree,
    read_source_file,
]


def get_default_model(model_name: str | None = None) -> BaseChatModel:
    """Initialize the default Gemini chat model.

    Args:
        model_name: Optional override for the Gemini model identifier.

    Returns:
        BaseChatModel: Configured ChatGoogleGenerativeAI instance.

    Raises:
        RuntimeError: If GEMINI_API_KEY is not set in environment.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured in backend/.env. Please provide a valid Google AI API key."
        )

    resolved_model = model_name or os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    primary_model = ChatGoogleGenerativeAI(
        model=resolved_model,
        temperature=0.2,
        google_api_key=api_key,
        max_retries=3,
    )
    # Configure fallback model to survive 503 high-demand temporary outages on primary models
    fallback_model = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        temperature=0.2,
        google_api_key=api_key,
        max_retries=3,
    )
    if resolved_model != "gemini-3.5-flash-lite":
        return primary_model.with_fallbacks([fallback_model])
    return primary_model


def create_agent_graph(
    model: BaseChatModel | None = None,
    tools: list[Any] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    """Build and compile the RepoPilot LangGraph agent.

    The workflow topology:
    1. START -> 'agent' node
    2. 'agent' node calls the LLM with bound tools
    3. Conditional edge 'tools_condition':
       - If LLM produced tool_calls -> route to 'tools' node
       - Else -> route to END
    4. 'tools' node executes tool functions and routes back to 'agent' node

    Args:
        model: Custom or mocked BaseChatModel instance (defaults to Gemini if None).
        tools: Tool functions available to the agent (defaults to AGENT_TOOLS).
        checkpointer: Checkpoint persistence mechanism (e.g. MemorySaver, AsyncPostgresSaver).

    Returns:
        CompiledStateGraph: Executable and streamable LangGraph agent.
    """
    active_tools = tools if tools is not None else AGENT_TOOLS
    active_model = model if model is not None else get_default_model()
    model_with_tools = active_model.bind_tools(active_tools)

    async def agent_node(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        """Agent node: injects system instructions and invokes the tool-bound model."""
        messages = list(state.get("messages", []))

        # Prepend system prompt if not present at start of conversation
        if not messages or not isinstance(messages[0], SystemMessage):
            # Inject repository context if present in state
            context_header = ""
            if state.get("repo_owner") and state.get("repo_name"):
                context_header = (
                    f"\n\nCurrent Active Repository: {state['repo_owner']}/{state['repo_name']}\n"
                    "Use this as default owner/repo for tools when not specified by user.\n"
                )
            full_system_prompt = SYSTEM_PROMPT + context_header
            messages = [SystemMessage(content=full_system_prompt), *messages]

        # Defensive check: if the previous turn was interrupted leaving an unfulfilled tool_call,
        # inject placeholder tool responses so the LLM doesn't reject malformed message sequences.
        if messages and isinstance(messages[-1], AIMessage):
            last_ai = messages[-1]
            if getattr(last_ai, "tool_calls", None):
                for tc in last_ai.tool_calls:
                    tc_id = tc.get("id") or "call_recovered"
                    tc_name = tc.get("name", "unknown_tool")
                    messages.append(
                        ToolMessage(
                            content=f"Tool call '{tc_name}' was interrupted or cancelled.",
                            tool_call_id=tc_id,
                        )
                    )

        # Inspect current recursion step budget
        runnable_config = config or {}
        step = runnable_config.get("metadata", {}).get("langgraph_step", 1)
        limit = runnable_config.get("recursion_limit")

        # When within 2 steps of the limit, conclude without tools to route to END gracefully
        if limit is not None and step >= limit - 2:
            conclude_instruction = SystemMessage(
                content=(
                    "You have reached the maximum tool exploration step limit for this query. "
                    "Do NOT attempt to call any more tools. Summarize your findings based strictly "
                    "on the information gathered so far and present your answer to the user now."
                )
            )
            response = await active_model.ainvoke([*messages, conclude_instruction], config)
            # Ensure no pending tool calls remain on this response
            if hasattr(response, "tool_calls") and response.tool_calls:
                response = AIMessage(
                    content=response.content or "Reached exploration step limit. Summary of findings provided above.",
                    id=getattr(response, "id", None),
                )
        else:
            response = await model_with_tools.ainvoke(messages, config)

        return {"messages": [response]}

    workflow = StateGraph(AgentState)

    # 1. Register nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(active_tools))

    # 2. Define edge connections
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile(checkpointer=checkpointer)
