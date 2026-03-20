"""Generate answer node for the ThinkBack agent.

This node handles generating responses using the LLM based on retrieved memories.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

from langchain.chat_models import init_chat_model

from src.agent.state import AgentState

logger = logging.getLogger(__name__)


@lru_cache
def _get_llm() -> Any:
    """Create and return the LLM instance (cached singleton).

    Returns:
        The configured LLM instance.
    """
    from src.core.config import Settings

    settings = Settings()
    return init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.openai_api_key,
        base_url=settings.llm_provider_base_url,
        temperature=0,
        streaming=False,
    )


# RAG prompt template for answering questions using memories
RAG_PROMPT = """You are a personal knowledge assistant.

Answer questions ONLY using the memories provided below.

If the memories do not contain the answer, say that the knowledge has not been saved yet.

User Question:
{question}

Memories:
{memories}

Answer using only these memories. If there are no memories, \
say you don't have any saved knowledge about this topic yet."""


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
        ...     "response": "",
        ...     "error": None,
        ... }
        >>> result = await generate_answer(state)
        >>> "From your saved memories" in result["response"]
        True
    """
    # Handle case with no memories
    if not state["memories"]:
        return {
            "response": (
                "I don't have any saved memories about this topic yet. Use /save to add knowledge."
            ),
        }

    try:
        # Format memories for the prompt
        memories_text = "\n".join([f"• {m['content']}" for m in state["memories"]])

        # Get cached LLM instance
        llm = _get_llm()

        prompt = RAG_PROMPT.format(
            question=state["user_input"],
            memories=memories_text,
        )

        response = llm.invoke(prompt)

        # Extract text content from the response
        response_text = (
            response.content if isinstance(response.content, str) else str(response.content)
        )

        return {
            "response": response_text,
        }
    except Exception as e:
        return {
            "error": f"Failed to generate answer: {e!s}",
            "response": "Sorry, I encountered an error while generating a response.",
        }
