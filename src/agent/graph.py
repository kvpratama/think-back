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
from langgraph.graph.state import CompiledStateGraph

from src.agent.middleware import trim_messages_by_turns
from src.agent.tools import save_memory_tool, search_memories_tool


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
    checkpointer: BaseCheckpointSaver[Any],
) -> CompiledStateGraph:
    """Build and compile the ThinkBack agent.

    Args:
        checkpointer: Checkpoint saver for state persistence.

    Returns:
        Compiled agent ready for invocation.
    """
    from src.core.prompts import get_prompt

    llm = _get_llm()

    prompt = get_prompt("thinkback-agent")
    messages = prompt.format_messages()
    system_msg = str(messages[0].content) if messages else ""

    agent = create_agent(
        model=llm,
        tools=[save_memory_tool, search_memories_tool],
        system_prompt=system_msg,
        checkpointer=checkpointer,
        middleware=[trim_messages_by_turns, ToolCallLimitMiddleware(run_limit=5)],  # type: ignore[arg-type]
    )

    return agent
