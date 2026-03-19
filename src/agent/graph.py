"""ThinkBack agent graph assembly.

This module wires together all the nodes into a LangGraph workflow.
Per AGENTS.md convention: graph.py is assembly only — no business logic here.
"""

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agent.nodes.generate_answer import generate_answer
from src.agent.nodes.intent_router import intent_router
from src.agent.nodes.retrieve_memories import retrieve_memories
from src.agent.nodes.save_memory import save_memory
from src.agent.state import AgentState


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[Any, Any]:
    """Build and compile the ThinkBack agent graph.

    The graph follows this flow:
    1. START -> intent_router
    2. intent_router -> save_memory (if intent is "save")
    3. intent_router -> retrieve_memories (if intent is "query")
    4. retrieve_memories -> generate_answer -> END

    Returns:
        Compiled StateGraph ready for invocation.

    Example:
        >>> graph = build_graph()
        >>> result = graph.invoke({
        ...     "user_input": "/save test memory",
        ...     "cleaned_input": "",
        ...     "intent": None,
        ...     "memories": [],
        ...     "response": "",
        ...     "error": None,
        ... })
    """
    # Create the graph with AgentState
    graph = StateGraph(AgentState)

    # Add all nodes
    graph.add_node("intent_router", intent_router)
    graph.add_node("save_memory", save_memory)
    graph.add_node("retrieve_memories", retrieve_memories)
    graph.add_node("generate_answer", generate_answer)

    # Set entry point
    graph.set_entry_point("intent_router")

    # Add edge from retrieve_memories to generate_answer
    graph.add_edge("retrieve_memories", "generate_answer")

    # Add edges to END
    graph.add_edge("save_memory", END)
    graph.add_edge("generate_answer", END)

    if checkpointer is None or not isinstance(checkpointer, BaseCheckpointSaver):
        return graph.compile(checkpointer=InMemorySaver())
    return graph.compile(checkpointer=checkpointer)
