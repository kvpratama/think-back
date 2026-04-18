"""Tool definitions for the ThinkBack agent.

These tools are used by `create_agent` to give the LLM the ability
to save and search memories in the vector database.
"""

from __future__ import annotations

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.types import interrupt

from src.db.vector_store import find_duplicates as db_find_duplicates
from src.db.vector_store import save_memory as db_save_memory
from src.db.vector_store import search_memories as db_search_memories


def _get_user_settings_id(config: RunnableConfig) -> str:
    """Extract user_settings_id from RunnableConfig.

    Args:
        config: The LangGraph runtime config.

    Returns:
        The user_settings_id string.

    Raises:
        ValueError: If user_settings_id is missing from config.
    """
    user_settings_id = config.get("configurable", {}).get("user_settings_id")
    if not user_settings_id:
        raise ValueError("user_settings_id is required in config['configurable']")
    return user_settings_id


@tool
async def search_memories_tool(
    query: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """Search saved knowledge for memories related to the query.

    Use this when the user asks about their saved knowledge or wants
    to recall something they previously saved.

    Args:
        query: The search query describing what to look for.
    """
    if not query:
        return "Please provide a search query."

    user_settings_id = _get_user_settings_id(config)
    results = await db_search_memories(query, user_settings_id=user_settings_id, top_k=5)

    if not results:
        return "No saved memories found for this topic."

    lines = [f"• {m['content']} (similarity: {m['similarity']:.2f})" for m in results]
    return "Here are the relevant memories:\n" + "\n".join(lines)


@tool
async def save_memory_tool(
    content: str = "",
    insight: str = "",
    *,
    config: RunnableConfig,
) -> str:
    """Save a piece of knowledge or insight to memory.

    Use this when the user shares something they want to remember —
    a lesson, realization, fact, or piece of wisdom. Extract the core
    insight from their message.

    Args:
        content: The user's original message (verbatim).
        insight: The extracted core insight or lesson.
    """
    if not content or not insight:
        return "Please provide both content and insight."

    user_settings_id = _get_user_settings_id(config)

    duplicates = await db_find_duplicates(content, user_settings_id=user_settings_id)

    confirmation = interrupt({"content": content, "insight": insight, "duplicates": duplicates})

    if not confirmation.get("approved", False):
        return "Save cancelled by user."

    await db_save_memory(content, summary=insight, user_settings_id=user_settings_id)
    return f"Memory saved: {insight}"
