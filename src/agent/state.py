"""Agent state definition for LangGraph.

The AgentState is a TypedDict that represents the state of the agent
throughout the graph execution. It uses Annotated fields with operator.add
for list fields that accumulate across steps.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from langgraph.graph import MessagesState


def add_memories(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add two lists of memories together.

    Used as a reducer for the memories field in AgentState.

    Args:
        left: The left list of memories.
        right: The right list of memories.

    Returns:
        The concatenated list of memories.
    """
    return left + right


class AgentState(MessagesState):
    """State of the ThinkBack agent.

    This TypedDict defines the structure of the agent's state throughout
    the LangGraph execution. Fields annotated with a reducer function
    will accumulate values across graph steps.

    Attributes:
        user_input: The raw user input from Telegram (including command prefix).
        cleaned_input: The user input with command prefix stripped.
        intent: The detected intent ('save', 'query', or None).
        memories: List of retrieved or saved memory records. Accumulates across steps.
        response: The final response to send to the user.
        error: Any error message that occurred during processing.
    """

    user_input: str
    cleaned_input: str
    intent: Literal["save", "query"] | None
    memories: Annotated[list[dict[str, Any]], add_memories]
    response: str
    error: str | None
