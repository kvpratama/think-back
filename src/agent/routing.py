"""Intent-based routing for the ThinkBack agent graph.

Contains routing functions used by conditional edges in the graph.
"""

from __future__ import annotations

from src.agent.state import AgentState


def route_by_intent(state: AgentState) -> str:
    """Route based on the detected intent.

    Args:
        state: The current agent state.

    Returns:
        The next node to route to: "save", "query", or "error".
    """

    error = state.get("error")
    if error is not None:
        return "error"
    if state["intent"] == "save":
        return "save"
    if state["intent"] == "query":
        return "query"
    return "error"
