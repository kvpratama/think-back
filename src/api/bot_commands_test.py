"""Tests for src/api/bot_commands.py."""

# mypy: disable-error-code="union-attr"

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Update, User


@pytest.fixture
def mock_user() -> MagicMock:
    """Create a mock Telegram User."""
    mock_user = MagicMock(spec=User)
    mock_user.id = 12345
    return mock_user


@pytest.fixture
def mock_update(mock_user: MagicMock) -> Update:
    """Create a mock Telegram Update."""
    mock_message = MagicMock()
    mock_message.from_user = mock_user
    mock_message.text = "I realized that motivation follows action"
    mock_message.reply_text = AsyncMock()
    mock_message.chat.id = 67890

    mock_update = MagicMock(spec=Update)
    mock_update.message = mock_message
    mock_update.effective_message = mock_message
    mock_update.effective_user = mock_user

    return mock_update


@pytest.fixture
def mock_context() -> MagicMock:
    """Create a mock Telegram Context."""
    context = MagicMock()
    context.bot = AsyncMock()
    return context


async def test_start_command_upserts_settings_and_shows_timezone_picker(
    mock_update: Update, mock_context: MagicMock
) -> None:
    """Test that /start upserts user settings and shows timezone keyboard for new users."""
    from src.api.bot_commands import start_command

    with (
        patch("src.api.bot_commands.upsert_user_settings", return_value=True) as mock_upsert,
        patch("src.api.bot_commands.get_user_settings_id", return_value="aaa-bbb"),
        patch("src.api.bot_commands.insert_default_reminders") as mock_reminders,
    ):
        await start_command(mock_update, mock_context)

    mock_upsert.assert_called_once_with("67890")
    mock_reminders.assert_called_once_with("aaa-bbb")

    assert mock_update.message is not None
    calls = cast(Any, mock_update.message.reply_text).call_args_list
    assert len(calls) == 2

    tz_call_kwargs = calls[1].kwargs if calls[1].kwargs else {}
    assert "reply_markup" in tz_call_kwargs


async def test_start_command_existing_user_shows_welcome_back_only(
    mock_update: Update, mock_context: MagicMock
) -> None:
    """Test that /start for existing user shows welcome back without timezone picker."""
    from src.api.bot_commands import start_command

    with (
        patch("src.api.bot_commands.upsert_user_settings", return_value=False),
        patch("src.api.bot_commands.get_user_settings_id", return_value="aaa-bbb"),
        patch("src.api.bot_commands.insert_default_reminders") as mock_reminders,
    ):
        await start_command(mock_update, mock_context)

    mock_reminders.assert_not_called()

    assert mock_update.message is not None
    calls = cast(Any, mock_update.message.reply_text).call_args_list
    assert len(calls) == 1

    welcome_text = calls[0].args[0] if calls[0].args else calls[0].kwargs.get("text", "")
    assert "Welcome to ThinkBack" in welcome_text


async def test_timezone_command_shows_offset_picker(
    mock_update: Update, mock_context: MagicMock
) -> None:
    """Test that /timezone shows the UTC offset keyboard."""
    from src.api.bot_commands import timezone_command

    await timezone_command(mock_update, mock_context)

    assert mock_update.message is not None
    call = cast(Any, mock_update.message.reply_text).call_args
    assert "time zone" in call.args[0] or "time zone" in call.kwargs.get("text", "")
    assert "reply_markup" in call.kwargs


async def test_reminders_command_shows_current_reminders(
    mock_update: Update, mock_context: MagicMock
) -> None:
    """Test that /reminders shows current reminder times with remove buttons."""
    from src.api.bot_commands import reminders_command

    with (
        patch("src.api.bot_commands.get_user_settings_id", return_value="aaa"),
        patch(
            "src.api.bot_commands.get_reminders",
            return_value=[
                {"id": "r1", "user_settings_id": "aaa", "time": "08:00:00"},
                {"id": "r2", "user_settings_id": "aaa", "time": "20:00:00"},
            ],
        ),
    ):
        await reminders_command(mock_update, mock_context)

    assert mock_update.message is not None
    call = cast(Any, mock_update.message.reply_text).call_args
    text = call.args[0] if call.args else call.kwargs.get("text", "")
    assert "08:00" in text
    assert "20:00" in text
    assert "reply_markup" in call.kwargs


async def test_reminders_command_no_settings(mock_update: Update, mock_context: MagicMock) -> None:
    """Test that /reminders handles missing user settings gracefully."""
    from src.api.bot_commands import reminders_command

    with patch("src.api.bot_commands.get_user_settings_id", return_value=None):
        await reminders_command(mock_update, mock_context)

    assert mock_update.message is not None
    call = cast(Any, mock_update.message.reply_text).call_args
    text = call.args[0] if call.args else call.kwargs.get("text", "")
    assert "/start" in text


@pytest.fixture
def mock_chat_member_update() -> Update:
    """Create a mock ChatMemberUpdated update for a new user joining."""
    mock_chat = MagicMock()
    mock_chat.id = 67890

    mock_new_member = MagicMock()
    mock_new_member.status = "member"

    mock_chat_member = MagicMock()
    mock_chat_member.chat = mock_chat
    mock_chat_member.new_chat_member = mock_new_member

    mock_update = MagicMock(spec=Update)
    mock_update.my_chat_member = mock_chat_member
    mock_update.effective_chat = mock_chat

    return mock_update


async def test_chat_member_update_new_user_sends_welcome(
    mock_chat_member_update: Update, mock_context: MagicMock
) -> None:
    """Test that ChatMemberUpdated sends welcome + timezone for new users."""
    from src.api.bot_commands import chat_member_update

    with (
        patch("src.api.bot_commands.upsert_user_settings", return_value=True) as mock_upsert,
        patch("src.api.bot_commands.get_user_settings_id", return_value="aaa-bbb"),
        patch("src.api.bot_commands.insert_default_reminders") as mock_reminders,
    ):
        await chat_member_update(mock_chat_member_update, mock_context)

    mock_upsert.assert_called_once_with("67890")
    mock_reminders.assert_called_once_with("aaa-bbb")

    calls = mock_context.bot.send_message.call_args_list
    assert len(calls) == 2

    welcome_text = calls[0].kwargs.get("text", "")
    assert "Welcome to ThinkBack" in welcome_text

    tz_call_kwargs = calls[1].kwargs
    assert "reply_markup" in tz_call_kwargs


async def test_chat_member_update_existing_user_no_welcome(
    mock_chat_member_update: Update, mock_context: MagicMock
) -> None:
    """Test that ChatMemberUpdated does nothing for existing users."""
    from src.api.bot_commands import chat_member_update

    with (
        patch("src.api.bot_commands.upsert_user_settings", return_value=False),
        patch("src.api.bot_commands.insert_default_reminders") as mock_reminders,
    ):
        await chat_member_update(mock_chat_member_update, mock_context)

    mock_reminders.assert_not_called()
    mock_context.bot.send_message.assert_not_called()


async def test_chat_member_update_non_member_status_ignored(
    mock_context: MagicMock,
) -> None:
    """Test that ChatMemberUpdated ignores non-member statuses (e.g., kicked)."""
    from src.api.bot_commands import chat_member_update

    mock_chat = MagicMock()
    mock_chat.id = 67890

    mock_new_member = MagicMock()
    mock_new_member.status = "kicked"

    mock_chat_member = MagicMock()
    mock_chat_member.chat = mock_chat
    mock_chat_member.new_chat_member = mock_new_member

    mock_update = MagicMock(spec=Update)
    mock_update.my_chat_member = mock_chat_member

    await chat_member_update(mock_update, mock_context)

    mock_context.bot.send_message.assert_not_called()


async def test_help_command_lists_commands(mock_update: Update, mock_context: MagicMock) -> None:
    """Test that /help lists all available commands."""
    from src.api.bot_commands import help_command

    await help_command(mock_update, mock_context)

    assert mock_update.message is not None
    call = cast(Any, mock_update.message.reply_text).call_args
    text = call.args[0] if call.args else call.kwargs.get("text", "")
    assert "/start" in text
    assert "/timezone" in text
    assert "/reminders" in text
    assert "/help" in text
