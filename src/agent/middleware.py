"""Middleware for the ThinkBack agent.

Provides a turn-based message trimming middleware that keeps
the last N conversation turns to prevent context window bloat.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from langchain.agents.middleware import before_model
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from src.core.config import get_settings


def split_into_turns(messages: Sequence[BaseMessage]) -> list[list[BaseMessage]]:
    """Split a flat message list into turn groups.

    A turn starts with a HumanMessage and includes all subsequent
    messages until the next HumanMessage. Messages before the first
    HumanMessage are prepended to the first turn.

    Args:
        messages: Flat list of conversation messages (no SystemMessages expected).

    Returns:
        List of turn groups, where each group is a list of messages.
    """
    if not messages:
        return []

    turns: list[list[BaseMessage]] = []
    current: list[BaseMessage] = []
    seen_human = False

    for msg in messages:
        if isinstance(msg, HumanMessage):
            if seen_human and current:
                turns.append(current)
                current = []
            seen_human = True
        current.append(msg)

    if current:
        turns.append(current)

    return turns


def _trim_messages_by_turns_impl(state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
    """Implementation of turn-based message trimming logic.

    Preserves all SystemMessages at the start. Groups remaining messages
    into turns (each starting with a HumanMessage) and keeps only the
    last ``max_turns`` turns.

    Args:
        state: The current agent state containing messages.
        runtime: The LangGraph runtime (unused).

    Returns:
        Updated state with trimmed messages, or None if no trimming needed.
    """
    messages = state.get("messages", [])
    if not messages:
        return None

    system_msgs = []
    conversation = []
    for msg in messages:
        if isinstance(msg, SystemMessage) and not conversation:
            system_msgs.append(msg)
        else:
            conversation.append(msg)

    turns = split_into_turns(conversation)

    settings = get_settings()
    if len(turns) <= settings.max_turns:
        return None

    kept_turns = turns[-settings.max_turns :]
    kept_messages = [msg for turn in kept_turns for msg in turn]

    return {"messages": system_msgs + kept_messages}


@before_model
def trim_messages_by_turns(state: dict[str, Any], runtime: Any) -> dict[str, Any] | None:
    """Trim conversation history to the last N turns before each LLM call.

    Preserves all SystemMessages at the start. Groups remaining messages
    into turns (each starting with a HumanMessage) and keeps only the
    last ``max_turns`` turns.

    Args:
        state: The current agent state containing messages.
        runtime: The LangGraph runtime (unused).

    Returns:
        Updated state with trimmed messages, or None if no trimming needed.
    """
    return _trim_messages_by_turns_impl(state, runtime)
