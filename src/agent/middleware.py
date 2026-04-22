"""Middleware for the ThinkBack agent.

Provides a turn-based message trimming middleware that keeps
the last N conversation turns to prevent context window bloat.
"""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware import before_model
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    RemoveMessage,
    trim_messages,
)

from src.core.config import get_settings


def _trim_messages_by_turns_impl(state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
    """Implementation of turn-based message trimming logic using trim_messages.

    A "turn" is counted as one HumanMessage and all subsequent non-Human
    messages (AIMessages, ToolMessages, etc.) until the next HumanMessage.
    Uses a custom turn counter that counts HumanMessages to keep only the
    last ``max_turns`` turns. Preserves the initial SystemMessage.

    Args:
        state: The current agent state containing messages.
        runtime: The LangGraph runtime (unused).

    Returns:
        Update dictionary with RemoveMessage objects for dropped messages,
        or None if no trimming needed.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    settings = get_settings()

    def turn_counter(msgs: list[BaseMessage]) -> int:
        return sum(1 for m in msgs if isinstance(m, HumanMessage))

    kept = trim_messages(
        messages,
        max_tokens=settings.max_turns,
        token_counter=turn_counter,
        strategy="last",
        start_on="human",
        include_system=True,
        allow_partial=False,
    )

    if len(kept) == len(messages):
        return None

    # Compute difference to generate RemoveMessage objects for LangGraph state
    kept_ids = {m.id for m in kept if m.id}
    remove_msgs = [RemoveMessage(id=m.id) for m in messages if m.id and m.id not in kept_ids]

    return {"messages": remove_msgs} if remove_msgs else None


@before_model
def trim_messages_by_turns(state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
    """Trim conversation history to the last N turns before each LLM call.

    A "turn" is defined as one HumanMessage plus all subsequent messages
    (AI responses, tool calls, tool results) until the next HumanMessage.
    Preserves the initial SystemMessage and keeps only the last
    ``max_turns`` turns.

    Args:
        state: The current agent state containing messages.
        runtime: The LangGraph runtime (unused).

    Returns:
        Updated state with trimmed messages, or None if no trimming needed.
    """
    return _trim_messages_by_turns_impl(state, runtime)
