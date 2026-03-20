"""Retrieve memories node for the ThinkBack agent.

This node handles searching for relevant memories using vector similarity.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agent.state import AgentState
from src.db.vector_store import search_memories as db_search_memories

logger = logging.getLogger(__name__)


async def retrieve_memories(state: AgentState) -> dict[str, Any]:
    """Retrieve relevant memories from the vector database.

    Uses cleaned_input (with command prefix already stripped by intent_router).

    Args:
        state: The current agent state containing cleaned_input as the query.

    Returns:
        Partial state update with retrieved memories.

    Example:
        >>> state = {
        ...     "user_input": "/ask What do I know about habits?",
        ...     "cleaned_input": "What do I know about habits?",
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
        memories = await db_search_memories(state["cleaned_input"], top_k=3)
        return {
            "memories": memories,
        }
    except Exception as e:
        logger.exception("Failed to retrieve memories: %s", e)
        raise e
