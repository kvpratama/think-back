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
            kwargs={
                "row_factory": dict_row,
                "autocommit": True,
                # Disable auto-prepared statements. Required for Supabase's
                # transaction pooler (port 6543) and harmless on session
                # connections; without this, recycled backends raise
                # "DbHandler exited" on the next aget_tuple call.
                "prepare_threshold": None,
            },
            # Validate a connection before handing it out so stale/closed
            # connections (e.g. killed by Supavisor) get discarded instead
            # of being returned to the caller.
            check=AsyncConnectionPool.check_connection,
            # Rotate connections proactively so we never hold one long
            # enough for the upstream pooler to drop it from under us.
            max_idle=300,  # 5 min
            max_lifetime=1800,  # 30 min
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
    async with _checkpointer_lock:
        instance = _checkpointer_instance
        _checkpointer_instance = None
    if instance is not None and isinstance(instance.conn, AsyncConnectionPool):
        await instance.conn.close()
