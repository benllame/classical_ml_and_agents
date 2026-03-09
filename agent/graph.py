"""ReAct Agent Graph — LangGraph-based orchestration.

We use LangGraph's StateGraph to implement the Thought → Action →
Observation loop. The graph has two nodes: an agent node where the LLM
reasons and decides what tool to call, and a tools node that executes
the selected tool.

Usage:
    from agent.graph import create_agent, run_agent

    agent = create_agent()
    response = run_agent(agent, "What is the churn risk of customer X?")
"""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from loguru import logger

from agent.prompts import SYSTEM_PROMPT
from agent.tools import ALL_TOOLS
from src.config import (
    GOOGLE_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_TEMPERATURE,
)

# ── State Definition ─────────────────────────────────────────────────────────


class AgentState(TypedDict):
    """State schema for the ReAct agent graph.

    Holds the conversation history. The add_messages reducer handles
    deduplication and proper message ordering as messages flow between nodes.

    messages: The conversation history (user + AI + tool messages).
    """

    messages: Annotated[list[BaseMessage], add_messages]


# ── LLM Factory ──────────────────────────────────────────────────────────────


def create_llm(provider: str | None = None, model: str | None = None, temperature: float | None = None):
    """Create the LLM instance based on configuration.

    We use temperature=0.0 for deterministic tool-calling so the agent
    gives consistent answers.

    Parameters
    ----------
    provider : str
        LLM provider. Defaults to config.
    model : str
        Model name. Defaults to config.
    temperature : float
        LLM temperature. Defaults to config.

    Returns
    -------
    ChatModel
        LangChain-compatible chat model.
    """
    provider = provider or LLM_PROVIDER
    model = model or LLM_MODEL
    temperature = temperature if temperature is not None else LLM_TEMPERATURE

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=GOOGLE_API_KEY or os.getenv("GOOGLE_API_KEY"),
    )


# ── Graph Construction ───────────────────────────────────────────────────────


def create_agent(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> StateGraph:
    """Create the ReAct agent graph.

    The graph has two nodes: 'agent' (LLM reasoning) and 'tools' (tool
    execution). The conditional edge from 'agent' routes to either 'tools'
    (if tool_calls present) or END (final answer). This two-node topology
    is the minimal ReAct implementation.

    The graph follows the pattern:
    1. User message → agent node (LLM reasons + may call tools)
    2. If tool call → tools node (executes tool) → back to agent
    3. If no tool call → END (final answer)

    Parameters
    ----------
    provider, model, temperature : optional
        LLM configuration overrides.

    Returns
    -------
    Compiled StateGraph
        The runnable agent graph.
    """
    llm = create_llm(provider, model, temperature)

    # bind_tools gives the LLM awareness of available tools — their names,
    # descriptions, and parameter schemas.
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # ── Define nodes ──

    def agent_node(state: AgentState) -> dict:
        """The reasoning node — LLM decides what to do next."""
        messages = state["messages"]

        # Prepend system prompt if not already there
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Tool node — executes whichever tool the LLM selected
    tool_node = ToolNode(ALL_TOOLS)

    # ── Define routing ──

    def should_continue(state: AgentState) -> str:
        """Determine if the agent should call a tool or finish.

        Routing function: if the last message contains tool_calls, route
        to 'tools' node. Otherwise, the agent has produced its final
        answer → END.
        """
        last_message = state["messages"][-1]

        # If the LLM made tool calls, route to tools node
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"

        # Otherwise, we're done
        return END

    # ── Build the graph ──

    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    # Set entry point
    graph.set_entry_point("agent")

    # Add conditional edge from agent
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})

    # Tools always route back to agent for the next reasoning step.
    graph.add_edge("tools", "agent")

    # Compile
    compiled = graph.compile()
    logger.info("ReAct agent graph compiled successfully")

    return compiled


# ── Runner ───────────────────────────────────────────────────────────────────


def run_agent(
    agent,
    query: str,
    conversation_history: list[BaseMessage] | None = None,
    stream: bool = False,
) -> dict[str, Any]:
    """Run the agent with a user query.

    Returns the final answer, all tool calls made, and the number of
    reasoning steps.

    Parameters
    ----------
    agent : CompiledGraph
        The compiled agent graph.
    query : str
        The user's question in natural language.
    conversation_history : list | None
        Previous messages for multi-turn conversations.
    stream : bool
        If True, return a generator that yields intermediate steps.

    Returns
    -------
    dict
        {
            "answer": str,          # Final agent response
            "messages": list,       # Full message history
            "tool_calls": list,     # Tools that were called
            "steps": int,           # Number of reasoning steps
        }
    """
    # Build initial state
    messages = []
    if conversation_history:
        messages.extend(conversation_history)
    messages.append(HumanMessage(content=query))

    initial_state = {"messages": messages}

    if stream:
        return _stream_agent(agent, initial_state)

    # Run the full graph
    result = agent.invoke(initial_state)

    # Extract results
    all_messages = result["messages"]
    final_message = all_messages[-1]

    # Collect tool calls made during execution
    tool_calls = []
    steps = 0
    for msg in all_messages:
        if isinstance(msg, AIMessage):
            steps += 1
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append(
                        {
                            "tool": tc["name"],
                            "args": tc["args"],
                        }
                    )

    # Gemini can return content as a list of blocks: [{'type':'text','text':'...'}]
    raw_content = final_message.content if hasattr(final_message, "content") else str(final_message)
    if isinstance(raw_content, list):
        answer = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        ).strip()
    else:
        answer = str(raw_content)

    return {
        "answer": answer,
        "messages": all_messages,
        "tool_calls": tool_calls,
        "steps": steps,
    }


def _stream_agent(agent, initial_state):
    """Generator that yields intermediate agent steps."""
    for event in agent.stream(initial_state):
        for node_name, node_output in event.items():
            yield {
                "node": node_name,
                "output": node_output,
            }


# ── Convenience: Quick Agent ─────────────────────────────────────────────────


def quick_ask(query: str, **kwargs) -> str:
    """One-liner to ask the agent a question and get the answer.

    Parameters
    ----------
    query : str
        Natural language question.
    **kwargs
        Passed to create_agent (provider, model, temperature).

    Returns
    -------
    str
        The agent's final answer.
    """
    agent = create_agent(**kwargs)
    result = run_agent(agent, query)
    return result["answer"]


# Alias so README examples (from agent.graph import create_graph) import cleanly
create_graph = create_agent
