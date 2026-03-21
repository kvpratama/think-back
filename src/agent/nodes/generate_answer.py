"""Generate answer node for the ThinkBack agent.

This node handles generating responses using the LLM based on retrieved memories.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.state import AgentState

logger = logging.getLogger(__name__)


@lru_cache
def _get_llm() -> BaseChatModel:
    """Create and return the LLM instance (cached singleton).

    Returns:
        The configured LLM instance.
    """
    from src.core.config import get_settings

    settings = get_settings()
    return init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.llm_provider_base_url,
        temperature=0,
        streaming=True,
    )


SYSTEM_MESSAGE = (
    "You are a personal knowledge assistant.\n\n"
    "Answer questions ONLY using the memories provided by the user.\n\n"
    "If the memories do not contain the answer, say that the knowledge has not been saved yet.\n\n"
    "If there are no memories, "
    "say you don't have any saved knowledge about this topic yet."
)


async def generate_answer(state: AgentState) -> dict[str, Any]:
    """Generate an answer using the LLM based on retrieved memories.

    Args:
        state: The current agent state containing memories and user_input.

    Returns:
        Partial state update with the generated response.

    Example:
        >>> state = {
        ...     "user_input": "/ask What do I know about habits?",
        ...     "cleaned_input": "What do I know about habits?",
        ...     "intent": "query",
        ...     "memories": [{"content": "Consistency beats intensity"}],
        ...     "messages": [],
        ...     "response": "",
        ...     "error": None,
        ... }
        >>> result = await generate_answer(state)
        >>> "response" in result and "messages" in result
        True
    """

    # Clean user message saved to state history (before try so it's always available)
    user_msg = HumanMessage(content=state.get("cleaned_input", ""))

    try:
        # Format memories for the prompt
        memories_text = "\n".join([f"• {m['content']}" for m in state["memories"]])

        # Get cached LLM instance
        llm = _get_llm()

        # Temporary context message for LLM only (not persisted)
        context_msg = HumanMessage(
            content=f"Memories:\n{memories_text}\n\nQuestion:\n{state['cleaned_input']}"
        )

        # Always prepend system message dynamically
        messages_for_llm = (
            [SystemMessage(content=SYSTEM_MESSAGE)] + state.get("messages", []) + [context_msg]
        )

        response = await llm.ainvoke(messages_for_llm)

        response_text = (
            response.content if isinstance(response.content, str) else str(response.content)
        )

        return {
            "messages": [user_msg, AIMessage(content=response_text)],
            "response": response_text,
        }
    except Exception as e:
        error_response = "Sorry, I encountered an error while generating a response."
        return {
            "messages": [user_msg, AIMessage(content=error_response)],
            "error": f"Failed to generate answer: {e!s}",
            "response": error_response,
        }
