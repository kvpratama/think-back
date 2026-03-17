"""Save memory node for the ThinkBack agent.

This node handles saving user-provided knowledge to the vector database.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.db.vector_store import save_memory as db_save_memory


async def save_memory(state: AgentState) -> AgentState:
    """Save the user's input as a memory to the vector database.

    Args:
        state: The current agent state containing user_input.

    Returns:
        Updated agent state with response message and saved memory.

    Example:
        >>> state = {
        ...     "user_input": "Consistency beats intensity",
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
        result = await db_save_memory(state["user_input"])
        return {
            **state,
            "response": "Memory saved.",
            "memories": [result],
        }
    except Exception as e:
        return {
            **state,
            "error": f"Failed to save memory: {e!s}",
            "response": "Failed to save memory. Please try again.",
        }
