"""Save memory node for the ThinkBack agent.

This node handles saving user-provided knowledge to the vector database.
"""

from __future__ import annotations

from typing import Any

from src.agent.state import AgentState
from src.db.vector_store import save_memory as db_save_memory


async def save_memory(state: AgentState) -> dict[str, Any]:
    """Save the user's input as a memory to the vector database.

    Uses cleaned_input (with command prefix already stripped by intent_router).

    Args:
        state: The current agent state containing cleaned_input.

    Returns:
        Partial state update with response message and saved memory.

    Example:
        >>> state = {
        ...     "user_input": "/save Consistency beats intensity",
        ...     "cleaned_input": "Consistency beats intensity",
        ...     "intent": "save",
        ...     "memories": [],
        ...     "response": "",
        ...     "error": None,
        ... }
        >>> result = await save_memory(state)
        >>> result["response"]
        'Memory saved.'
    """
    try:
        result = await db_save_memory(state["cleaned_input"])
        return {
            "response": "Memory saved.",
            "memories": [result],
        }
    except Exception as e:
        return {
            "error": f"Failed to save memory: {e!s}",
            "response": "Failed to save memory. Please try again.",
        }
