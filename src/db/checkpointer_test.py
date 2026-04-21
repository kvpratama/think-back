"""Tests for the PostgresSaver checkpointer module."""

from unittest.mock import MagicMock, patch


def test_get_checkpointer_returns_postgres_saver() -> None:
    """Test that get_checkpointer returns a PostgresSaver instance."""
    from src.db.checkpointer import get_checkpointer

    mock_settings = MagicMock()
    mock_settings.database_url.get_secret_value.return_value = (
        "postgresql://user:pass@localhost:5432/db"
    )

    mock_pool = MagicMock()
    mock_saver = MagicMock()

    with (
        patch("src.core.config.get_settings", return_value=mock_settings),
        patch("src.db.checkpointer.ConnectionPool", return_value=mock_pool),
        patch("src.db.checkpointer.PostgresSaver", return_value=mock_saver),
    ):
        get_checkpointer.cache_clear()
        result = get_checkpointer()

    assert result is mock_saver
    mock_saver.setup.assert_called_once()


def test_get_checkpointer_is_singleton() -> None:
    """Test that get_checkpointer returns the same instance on repeated calls."""
    from src.db.checkpointer import get_checkpointer

    mock_settings = MagicMock()
    mock_settings.database_url.get_secret_value.return_value = (
        "postgresql://user:pass@localhost:5432/db"
    )

    mock_pool = MagicMock()
    mock_saver = MagicMock()

    with (
        patch("src.core.config.get_settings", return_value=mock_settings),
        patch("src.db.checkpointer.ConnectionPool", return_value=mock_pool),
        patch("src.db.checkpointer.PostgresSaver", return_value=mock_saver),
    ):
        get_checkpointer.cache_clear()
        first = get_checkpointer()
        second = get_checkpointer()

    assert first is second
