"""PostgresSaver checkpointer singleton.

All LangGraph checkpointer access goes through this module.
"""

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

_checkpointer_instance: AsyncPostgresSaver | None = None


async def aget_checkpointer() -> AsyncPostgresSaver:
    """Get or create the AsyncPostgresSaver checkpointer instance.

    Returns a singleton checkpointer.
    Calls setup() on first use to create checkpoint tables (idempotent).

    Returns:
        AsyncPostgresSaver: The checkpointer instance.
    """
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        return _checkpointer_instance

    from psycopg.rows import dict_row

    from src.core.config import get_settings

    settings = get_settings()
    pool = AsyncConnectionPool(
        settings.database_url.get_secret_value(),
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=False,
    )
    await pool.open()
    saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
    await saver.setup()
    _checkpointer_instance = saver
    return saver


async def aclose_checkpointer() -> None:
    """Close the underlying connection pool if it exists."""
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        if isinstance(_checkpointer_instance.conn, AsyncConnectionPool):
            await _checkpointer_instance.conn.close()
        _checkpointer_instance = None
