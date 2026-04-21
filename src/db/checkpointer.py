"""PostgresSaver checkpointer singleton.

All LangGraph checkpointer access goes through this module.
"""

import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

_checkpointer_instance: AsyncPostgresSaver | None = None
_checkpointer_lock: asyncio.Lock = asyncio.Lock()


async def aget_checkpointer() -> AsyncPostgresSaver:
    """Get or create the AsyncPostgresSaver checkpointer instance.

    Returns a singleton checkpointer.
    Calls setup() on first use to create checkpoint tables (idempotent).
    Uses double-checked locking to prevent concurrent callers from
    creating duplicate connection pools.

    Returns:
        AsyncPostgresSaver: The checkpointer instance.
    """
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        return _checkpointer_instance

    async with _checkpointer_lock:
        # Re-check after acquiring the lock — another caller may have
        # completed initialization while we were waiting.
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
        try:
            saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
            await saver.setup()
        except BaseException:
            await pool.close()
            raise
        _checkpointer_instance = saver
        return saver


async def aclose_checkpointer() -> None:
    """Close the underlying connection pool if it exists."""
    global _checkpointer_instance
    if _checkpointer_instance is not None:
        if isinstance(_checkpointer_instance.conn, AsyncConnectionPool):
            await _checkpointer_instance.conn.close()
        _checkpointer_instance = None
