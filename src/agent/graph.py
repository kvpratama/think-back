"""ThinkBack agent graph assembly.

This module wires together the agent using create_agent.
Per AGENTS.md convention: graph.py is assembly only — no business logic here.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ToolCallLimitMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph

from src.agent.tools import save_memory_tool, search_memories_tool

SYSTEM_PROMPT = (
    "You are ThinkBack, a personal knowledge assistant on Telegram.\n\n"
    "IMPORTANT — always search first:\n"
    "For ANY message with topical or knowledge content, ALWAYS use "
    "search_memories_tool BEFORE responding. Your first instinct must be to "
    "check the user's saved knowledge — never rely on your own knowledge "
    "without searching first. If no saved memories are found, tell the user "
    "you don't have any saved knowledge on that topic.\n\n"
    "The only exception is simple greetings, thanks, or pleasantries "
    "(e.g. 'hi', 'thanks', 'good morning') — respond to those naturally "
    "without searching.\n\n"
    "Your capabilities:\n"
    "1. SAVE: When the user shares an insight, lesson, or piece of knowledge "
    "they want to remember, use the save_memory_tool. Extract the core insight "
    "from their message for the `insight` parameter, and pass the original "
    "message as `content`.\n"
    "2. QUERY: For any topical message, search saved knowledge first using "
    "search_memories_tool, then answer based on the results. If nothing is "
    "found, let the user know.\n\n"
    "The user may use /save or /ask commands, or just type naturally. "
    "Detect the intent from context.\n\n"
    "When saving, always extract a concise insight from the user's message "
    "for the `insight` parameter. The `content` parameter should be the "
    "user's original message verbatim.\n\n"
    "FORMATTING:\n"
    "You are responding via Telegram, which supports HTML formatting. "
    "Use these tags to make your responses clear and readable:\n"
    "• <b>bold</b> for emphasis or key terms\n"
    "• <i>italic</i> for titles, quotes, or subtle emphasis\n"
    "• <code>code</code> for inline code or technical terms\n"
    "• <pre>code block</pre> for multi-line code\n"
    "• <blockquote>text</blockquote> for quoting memories or insights\n"
    "Do NOT use Markdown syntax (**, *, `, ```) — only HTML tags."
)


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
    )


def build_graph(
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph:
    """Build and compile the ThinkBack agent.

    Args:
        checkpointer: Optional checkpoint saver for state persistence.
            Defaults to InMemorySaver if None.

    Returns:
        Compiled agent ready for invocation.
    """
    llm = _get_llm()

    if checkpointer is None:
        checkpointer = InMemorySaver()

    agent = create_agent(
        model=llm,
        tools=[save_memory_tool, search_memories_tool],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
        middleware=[ToolCallLimitMiddleware(run_limit=5)],
    )

    return agent
