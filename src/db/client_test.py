"""Tests for the Supabase database client module."""

from unittest.mock import MagicMock, patch


def test_get_supabase_client_returns_client() -> None:
    """Test that get_supabase_client returns a Supabase client instance."""
    from src.db.client import get_supabase_client

    mock_settings = MagicMock()
    mock_settings.supabase_url = "https://test.supabase.co"
    mock_settings.supabase_key = "test-key"

    with patch("src.db.client.create_client") as mock_create:
        with patch("src.core.config.get_settings", return_value=mock_settings):
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            # Clear the lru_cache to ensure fresh test
            get_supabase_client.cache_clear()

            client = get_supabase_client()

            assert client is mock_client
            mock_create.assert_called_once_with("https://test.supabase.co", "test-key")


def test_get_supabase_client_is_singleton() -> None:
    """Test that get_supabase_client returns the same client instance."""
    from src.db.client import get_supabase_client

    mock_settings = MagicMock()
    mock_settings.supabase_url = "https://test.supabase.co"
    mock_settings.supabase_key = "test-key"

    with patch("src.db.client.create_client") as mock_create:
        with patch("src.core.config.get_settings", return_value=mock_settings):
            mock_client = MagicMock()
            mock_create.return_value = mock_client

            # Clear the lru_cache to ensure fresh test
            get_supabase_client.cache_clear()

            client1 = get_supabase_client()
            client2 = get_supabase_client()

            assert client1 is client2
            mock_create.assert_called_once()
