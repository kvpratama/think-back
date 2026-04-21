"""PostgresSaver checkpointer singleton.

All LangGraph checkpointer access goes through this module.
"""

from functools import lru_cache

from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool


@lru_cache
def get_checkpointer() -> PostgresSaver:
    """Get or create the PostgresSaver checkpointer instance.

    Returns a singleton checkpointer using LRU cache.
    Calls setup() on first use to create checkpoint tables (idempotent).

    Returns:
        PostgresSaver: The checkpointer instance.
    """
    from src.core.config import get_settings

    settings = get_settings()
    pool = ConnectionPool(
        settings.database_url.get_secret_value(),
        kwargs={"row_factory": dict_row},
    )
    saver = PostgresSaver(pool)  # type: ignore[arg-type]
    saver.setup()
    return saver
