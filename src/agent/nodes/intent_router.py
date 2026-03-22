"""Intent router node for the ThinkBack agent.

This node routes user input to the appropriate flow based on the command.
It also strips command prefixes and sets cleaned_input for downstream nodes.
"""

from __future__ import annotations

from typing import Literal

from langgraph.types import Command

from src.agent.state import AgentState


async def intent_router(
    state: AgentState,
) -> Command[Literal["save_memory", "retrieve_memories", "__end__"]]:
    """Route the user input to the appropriate intent.

    Detects whether the user wants to save a memory or query knowledge
    based on the command prefix (/save or /ask). Strips the command
    prefix and stores the cleaned text in cleaned_input.

    Args:
        state: The current agent state containing user_input.

    Returns:
        Partial state update with intent and cleaned_input.

    Example:
        >>> state = {
        ...     "user_input": "/save Consistency beats intensity",
        ...     "cleaned_input": "",
        ...     "intent": None,
        ...     "memories": [],
        ...     "response": "",
        ...     "error": None,
        ... }
        >>> result = await intent_router(state)
        >>> result["intent"]
        'save'
        >>> result["cleaned_input"]
        'Consistency beats intensity'
    """
    user_input = state["user_input"].strip()
    user_input_lower = user_input.lower()

    if user_input_lower == "/save" or user_input_lower.startswith("/save "):
        prefix = "/save"
    elif user_input_lower == "/ask" or user_input_lower.startswith("/ask "):
        prefix = "/ask"
    else:
        return Command(
            update={
                "intent": None,
                "cleaned_input": user_input,
                "error": "Unknown command. Use /save or /ask.",
                "response": "Unknown command. Use /save or /ask.",
            },
            goto="__end__",
        )

    cleaned = user_input[len(prefix) :].strip()
    if not cleaned:
        return Command(
            update={
                "intent": None,
                "cleaned_input": "",
                "error": f"No content provided after {prefix}.",
                "response": f"No content provided after {prefix}.",
            },
            goto="__end__",
        )

    if prefix == "/save":
        return Command(update={"intent": "save", "cleaned_input": cleaned}, goto="save_memory")
    return Command(update={"intent": "query", "cleaned_input": cleaned}, goto="retrieve_memories")
