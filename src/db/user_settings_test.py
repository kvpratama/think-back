"""Tests for user settings DB operations."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_supabase() -> MagicMock:
    """Create a mock Supabase client."""
    return MagicMock()


class TestUpsertUserSettings:
    """Tests for upsert_user_settings()."""

    def test_inserts_new_user_and_returns_true(self, mock_supabase: MagicMock) -> None:
        # Simulate upsert returning a row with created_at == updated_at (new insert)
        mock_supabase.table.return_value.upsert.return_value.execute.return_value.data = [
            {
                "id": "aaa",
                "telegram_chat_id": "123",
                "timezone": "UTC",
                "created_at": "2026-04-17T00:00:00+00:00",
                "updated_at": "2026-04-17T00:00:00+00:00",
            }
        ]

        with patch("src.db.user_settings.get_supabase_client", return_value=mock_supabase):
            from src.db.user_settings import upsert_user_settings

            is_new = upsert_user_settings("123")

        assert is_new is True
        mock_supabase.table.assert_called_with("user_settings")

    def test_returns_false_for_existing_user(self, mock_supabase: MagicMock) -> None:
        # Simulate upsert returning a row with updated_at > created_at (existing)
        mock_supabase.table.return_value.upsert.return_value.execute.return_value.data = [
            {
                "id": "aaa",
                "telegram_chat_id": "123",
                "timezone": "Etc/GMT-7",
                "created_at": "2026-04-10T00:00:00+00:00",
                "updated_at": "2026-04-17T00:00:00+00:00",
            }
        ]

        with patch("src.db.user_settings.get_supabase_client", return_value=mock_supabase):
            from src.db.user_settings import upsert_user_settings

            is_new = upsert_user_settings("123")

        assert is_new is False


class TestInsertDefaultReminders:
    """Tests for insert_default_reminders()."""

    def test_inserts_two_default_reminder_times(self, mock_supabase: MagicMock) -> None:
        mock_supabase.table.return_value.insert.return_value.execute.return_value.data = [
            {"id": "r1", "user_settings_id": "aaa", "time": "08:00:00"},
            {"id": "r2", "user_settings_id": "aaa", "time": "20:00:00"},
        ]

        with patch("src.db.user_settings.get_supabase_client", return_value=mock_supabase):
            from src.db.user_settings import insert_default_reminders

            insert_default_reminders("aaa")

        insert_call = mock_supabase.table.return_value.insert.call_args[0][0]
        assert len(insert_call) == 2
        times = {row["time"] for row in insert_call}
        assert times == {"08:00", "20:00"}


class TestUpdateTimezone:
    """Tests for update_timezone()."""

    def test_updates_timezone_for_chat_id(self, mock_supabase: MagicMock) -> None:
        mock_response = mock_supabase.table.return_value.update.return_value.eq.return_value
        mock_response.execute.return_value.data = [
            {"id": "aaa", "telegram_chat_id": "123", "timezone": "Etc/GMT-7"}
        ]

        with patch("src.db.user_settings.get_supabase_client", return_value=mock_supabase):
            from src.db.user_settings import update_timezone

            update_timezone("123", "Etc/GMT-7")

        mock_supabase.table.assert_called_with("user_settings")
        update_call = mock_supabase.table.return_value.update.call_args[0][0]
        assert update_call == {"timezone": "Etc/GMT-7"}


class TestGetUserSettingsId:
    """Tests for get_user_settings_id()."""

    def test_returns_id_for_existing_user(self, mock_supabase: MagicMock) -> None:
        mock_response = mock_supabase.table.return_value.select.return_value.eq.return_value
        mock_response.execute.return_value.data = [{"id": "aaa-bbb"}]

        with patch("src.db.user_settings.get_supabase_client", return_value=mock_supabase):
            from src.db.user_settings import get_user_settings_id

            result = get_user_settings_id("123")

        assert result == "aaa-bbb"

    def test_returns_none_for_missing_user(self, mock_supabase: MagicMock) -> None:
        mock_response = mock_supabase.table.return_value.select.return_value.eq.return_value
        mock_response.execute.return_value.data = []

        with patch("src.db.user_settings.get_supabase_client", return_value=mock_supabase):
            from src.db.user_settings import get_user_settings_id

            result = get_user_settings_id("999")

        assert result is None
