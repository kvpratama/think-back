"""Generate answer node for the ThinkBack agent.

This node handles generating responses using the LLM based on retrieved memories.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from src.agent.state import AgentState

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


async def generate_answer(state: AgentState) -> AgentState:
    """Generate an answer using the LLM based on retrieved memories.

    Args:
        state: The current agent state containing memories and user_input.

    Returns:
        Updated agent state with the generated response.

    Example:
        >>> state = {
        ...     "user_input": "What do I know about habits?",
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
            **state,
            "response": (
                "I don't have any saved memories about this topic yet. Use /save to add knowledge."
            ),
        }

    try:
        # Format memories for the prompt
        memories_text = "\n".join([f"• {m['content']}" for m in state["memories"]])

        # Create and invoke the LLM
        from src.core.config import Settings

        settings = Settings()
        llm = ChatOpenAI(model=settings.llm_model, api_key=settings.openai_api_key)

        prompt = RAG_PROMPT.format(
            question=state["user_input"],
            memories=memories_text,
        )

        response = llm.invoke(prompt)

        return {
            **state,
            "response": response.content,  # type: ignore[attr-defined]
        }
    except Exception as e:
        return {
            **state,
            "error": f"Failed to generate answer: {e!s}",
            "response": "Sorry, I encountered an error while generating a response.",
        }
