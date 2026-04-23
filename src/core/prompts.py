"""Prompt retrieval with LangSmith Hub integration.

Pulls prompts from LangSmith Hub at runtime, falling back to hardcoded
defaults in ``prompt_defaults.py`` when the hub is unavailable.
"""

import logging
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.prompt_cache import configure_global_prompt_cache

from src.core.prompt_defaults import DEFAULTS as _DEFAULTS

logger = logging.getLogger(__name__)


@lru_cache
def _get_ls_client() -> Client:
    """Return a lazily-initialised LangSmith client (cached singleton)."""
    configure_global_prompt_cache()
    return Client()


def get_prompt(name: str, *, tag: str = "prod") -> ChatPromptTemplate:
    """Pull a prompt from LangSmith Hub, falling back to hardcoded default.

    Args:
        name: The LangSmith prompt name (e.g. "thinkback-agent").
        tag: Commit tag to pull. Defaults to "prod".

    Returns:
        The prompt template from LangSmith, or the hardcoded fallback.
    """
    try:
        result = _get_ls_client().pull_prompt(f"{name}:{tag}")
        if isinstance(result, ChatPromptTemplate):
            return result
        logger.warning(
            "Non-ChatPromptTemplate returned from LangSmith for '%s', using default", name
        )
        return _DEFAULTS[name]
    except Exception as e:
        logger.warning("LangSmith unavailable for '%s': %s", name, e)
        if name in _DEFAULTS:
            return _DEFAULTS[name]
        raise ValueError(f"Prompt '{name}' not found in LangSmith or defaults") from e
