"""Retrieve memories node for the ThinkBack agent.

This node handles searching for relevant memories using vector similarity.
"""

from __future__ import annotations

from src.agent.state import AgentState
from src.db.vector_store import search_memories as db_search_memories


async def retrieve_memories(state: AgentState) -> AgentState:
    """Retrieve relevant memories from the vector database.

    Args:
        state: The current agent state containing user_input as the query.

    Returns:
        Updated agent state with retrieved memories.

    Example:
        >>> state = {
        ...     "user_input": "What do I know about habits?",
        ...     "intent": "query",
        ...     "memories": [],
        ...     "response": "",
        ...     "error": None,
        ... }
        >>> result = await retrieve_memories(state)
        >>> len(result["memories"]) >= 0
        True
    """
    try:
        memories = await db_search_memories(state["user_input"], top_k=3)
        return {
            **state,
            "memories": memories,
        }
    except Exception as e:
        return {
            **state,
            "error": f"Failed to retrieve memories: {e!s}",
            "memories": [],
        }
