"""Tests for the PostgresSaver checkpointer module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def reset_checkpointer_state():
    """Reset checkpointer singleton state before and after each test."""
    import src.db.checkpointer as checkpointer_module

    checkpointer_module._checkpointer_instance = None
    yield
    checkpointer_module._checkpointer_instance = None


@pytest.mark.asyncio
async def test_get_checkpointer_returns_postgres_saver() -> None:
    """Test that aget_checkpointer returns an AsyncPostgresSaver instance."""
    from src.db.checkpointer import aget_checkpointer

    mock_settings = MagicMock()
    mock_settings.database_url.get_secret_value.return_value = (
        "postgresql://user:pass@localhost:5432/db"
    )

    mock_pool = MagicMock()
    mock_pool.open = AsyncMock()
    mock_saver = MagicMock()
    mock_saver.setup = AsyncMock()

    with (
        patch("src.core.config.get_settings", return_value=mock_settings),
        patch("src.db.checkpointer.AsyncConnectionPool", return_value=mock_pool),
        patch("src.db.checkpointer.AsyncPostgresSaver", return_value=mock_saver),
    ):
        result = await aget_checkpointer()

    assert result is mock_saver
    mock_pool.open.assert_awaited_once()
    mock_saver.setup.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_checkpointer_is_singleton() -> None:
    """Test that aget_checkpointer returns the same instance on repeated calls."""
    from src.db.checkpointer import aget_checkpointer

    mock_settings = MagicMock()
    mock_settings.database_url.get_secret_value.return_value = (
        "postgresql://user:pass@localhost:5432/db"
    )

    mock_pool = MagicMock()
    mock_pool.open = AsyncMock()
    mock_saver = MagicMock()
    mock_saver.setup = AsyncMock()

    with (
        patch("src.core.config.get_settings", return_value=mock_settings),
        patch("src.db.checkpointer.AsyncConnectionPool", return_value=mock_pool),
        patch("src.db.checkpointer.AsyncPostgresSaver", return_value=mock_saver),
    ):
        first = await aget_checkpointer()
        second = await aget_checkpointer()

    assert first is second


@pytest.mark.asyncio
async def test_aclose_checkpointer_closes_pool() -> None:
    """Test that aclose_checkpointer closes the pool and resets singleton."""
    from psycopg_pool import AsyncConnectionPool

    import src.db.checkpointer as checkpointer_module
    from src.db.checkpointer import aclose_checkpointer

    mock_pool = MagicMock(spec=AsyncConnectionPool)
    mock_pool.close = AsyncMock()
    mock_saver = MagicMock()
    mock_saver.conn = mock_pool

    checkpointer_module._checkpointer_instance = mock_saver

    await aclose_checkpointer()

    mock_pool.close.assert_awaited_once()
    assert checkpointer_module._checkpointer_instance is None


@pytest.mark.asyncio
async def test_aclose_checkpointer_noop_when_none() -> None:
    """Test that aclose_checkpointer no-ops when instance is None."""
    from src.db.checkpointer import aclose_checkpointer

    await aclose_checkpointer()


@pytest.mark.asyncio
async def test_get_checkpointer_closes_pool_on_setup_failure() -> None:
    """Test that pool is closed when saver.setup() fails."""
    from src.db.checkpointer import aget_checkpointer

    mock_settings = MagicMock()
    mock_settings.database_url.get_secret_value.return_value = (
        "postgresql://user:pass@localhost:5432/db"
    )

    mock_pool = MagicMock()
    mock_pool.open = AsyncMock()
    mock_pool.close = AsyncMock()
    mock_saver = MagicMock()
    mock_saver.setup = AsyncMock(side_effect=RuntimeError("setup failed"))

    with (
        patch("src.core.config.get_settings", return_value=mock_settings),
        patch("src.db.checkpointer.AsyncConnectionPool", return_value=mock_pool),
        patch("src.db.checkpointer.AsyncPostgresSaver", return_value=mock_saver),
        pytest.raises(RuntimeError, match="setup failed"),
    ):
        await aget_checkpointer()

    mock_pool.close.assert_awaited_once()
