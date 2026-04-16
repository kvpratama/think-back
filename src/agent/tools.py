"""Tool definitions for the ThinkBack agent.

These tools are used by `create_agent` to give the LLM the ability
to save and search memories in the vector database.
"""

from __future__ import annotations

from langchain_core.tools import tool
from langgraph.types import interrupt

from src.db.vector_store import save_memory as db_save_memory
from src.db.vector_store import search_memories as db_search_memories


@tool
async def search_memories_tool(query: str = "") -> str:
    """Search saved knowledge for memories related to the query.

    Use this when the user asks about their saved knowledge or wants
    to recall something they previously saved.

    Args:
        query: The search query describing what to look for.
    """
    if not query:
        return "Please provide a search query."

    results = await db_search_memories(query, top_k=3)

    if not results:
        return "No saved memories found for this topic."

    lines = [f"• {m['content']}" for m in results]
    return "Here are the relevant memories:\n" + "\n".join(lines)


@tool
async def save_memory_tool(content: str = "", insight: str = "") -> str:
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

    confirmation = interrupt({"content": content, "insight": insight})

    if not confirmation.get("approved", False):
        return "Save cancelled."

    await db_save_memory(content, summary=insight)
    return f"Memory saved: {insight}"
