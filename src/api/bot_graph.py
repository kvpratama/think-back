"""Graph initialization for the Telegram bot."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph
    from telegram.ext import ContextTypes


def get_graph(context: ContextTypes.DEFAULT_TYPE) -> CompiledStateGraph:
    """Lazily build and cache the agent graph in bot_data.

    Args:
        context: The Telegram context whose bot_data stores the graph.

    Returns:
        Compiled LangGraph instance.
    """
    if "graph" not in context.bot_data:
        from src.agent.graph import build_graph
        from src.db.checkpointer import get_checkpointer

        context.bot_data["graph"] = build_graph(
            checkpointer=get_checkpointer(),
        )
    return context.bot_data["graph"]
