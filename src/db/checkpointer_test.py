"""Tests for the PostgresSaver checkpointer module."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_get_checkpointer_returns_postgres_saver() -> None:
    """Test that aget_checkpointer returns an AsyncPostgresSaver instance."""
    import src.db.checkpointer as checkpointer_module
    from src.db.checkpointer import aget_checkpointer

    # Reset singleton state
    checkpointer_module._checkpointer_instance = None

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
    mock_saver.setup.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_checkpointer_is_singleton() -> None:
    """Test that aget_checkpointer returns the same instance on repeated calls."""
    import src.db.checkpointer as checkpointer_module
    from src.db.checkpointer import aget_checkpointer

    # Reset singleton state
    checkpointer_module._checkpointer_instance = None

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
