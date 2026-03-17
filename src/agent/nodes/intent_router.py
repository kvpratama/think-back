"""Intent router node for the ThinkBack agent.

This node routes user input to the appropriate flow based on the command.
"""

from __future__ import annotations

from src.agent.state import AgentState


async def intent_router(state: AgentState) -> AgentState:
    """Route the user input to the appropriate intent.

    Detects whether the user wants to save a memory or query knowledge
    based on the command prefix (/save or /ask).

    Args:
        state: The current agent state containing user_input.

    Returns:
        Updated agent state with the detected intent.

    Example:
        >>> state = {
        ...     "user_input": "/save Consistency beats intensity",
        ...     "intent": None,
        ...     "memories": [],
        ...     "response": "",
        ...     "error": None,
        ... }
        >>> result = await intent_router(state)
        >>> result["intent"]
        'save'
    """
    user_input = state["user_input"].strip().lower()

    if user_input.startswith("/save"):
        return {
            **state,
            "intent": "save",
        }
    elif user_input.startswith("/ask") or user_input.startswith("/query"):
        return {
            **state,
            "intent": "query",
        }
    else:
        return {
            **state,
            "intent": None,
            "error": "Unknown command. Use /save to save knowledge or /ask to query.",
            "response": "Unknown command. Use /save to save knowledge or /ask to query.",
        }
