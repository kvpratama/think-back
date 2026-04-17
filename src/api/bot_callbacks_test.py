"""Tests for src/api/bot_callbacks.py."""

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
def mock_context() -> MagicMock:
    """Create a mock Telegram Context."""
    context = MagicMock()
    context.bot = AsyncMock()
    context.bot_data = {}
    return context


@pytest.fixture
def mock_callback_update(mock_user: MagicMock) -> Update:
    """Create a mock Telegram Update with callback query."""
    mock_query = MagicMock()
    mock_query.data = "save_yes|67890_12345"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.from_user = mock_user

    mock_update = MagicMock(spec=Update)
    mock_update.callback_query = mock_query
    mock_update.message = None

    return mock_update


async def test_handle_timezone_callback(
    mock_context: MagicMock,
) -> None:
    """Test that timezone callback updates the user's timezone."""
    from src.api.bot_callbacks import handle_callback

    mock_query = MagicMock()
    mock_query.data = "tz|7|67890"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.callback_query = mock_query
    mock_update.message = None

    with patch("src.api.bot_callbacks.update_timezone") as mock_update_tz:
        await handle_callback(mock_update, mock_context)

    mock_update_tz.assert_called_once_with("67890", "Etc/GMT-7")
    mock_query.edit_message_text.assert_called_once()


async def test_handle_callback_approved(
    mock_callback_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that approved callback resumes graph and saves memory."""
    from src.api.bot_callbacks import handle_callback

    mock_graph = MagicMock()
    mock_result_msg = MagicMock()
    mock_result_msg.content = "Memory saved: Motivation follows action"
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [mock_result_msg]})

    mock_context.bot_data["graph"] = mock_graph

    await handle_callback(mock_callback_update, mock_context)

    mock_graph.ainvoke.assert_called_once()
    call_args = mock_graph.ainvoke.call_args
    command = call_args[0][0]
    assert command.resume == {"approved": True}

    assert mock_callback_update.callback_query is not None
    cast(Any, mock_callback_update.callback_query.edit_message_text).assert_called_once()


async def test_handle_callback_cancelled(
    mock_callback_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that cancelled callback resumes graph without saving."""
    from src.api.bot_callbacks import handle_callback

    assert mock_callback_update.callback_query is not None
    cast(Any, mock_callback_update.callback_query).data = "save_no|67890_12345"

    mock_graph = MagicMock()
    mock_result_msg = MagicMock()
    mock_result_msg.content = "Save cancelled."
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [mock_result_msg]})

    mock_context.bot_data["graph"] = mock_graph

    await handle_callback(mock_callback_update, mock_context)

    call_args = mock_graph.ainvoke.call_args
    command = call_args[0][0]
    assert command.resume == {"approved": False}


async def test_handle_callback_remove_reminder(
    mock_context: MagicMock,
) -> None:
    """Test that remove reminder callback removes and refreshes the list."""
    from src.api.bot_callbacks import handle_callback

    mock_query = MagicMock()
    mock_query.data = "rm_rem|0|aaa"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()
    mock_query.message.chat.id = 67890

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with (
        patch("src.api.bot_callbacks.remove_reminder") as mock_remove,
        patch(
            "src.api.bot_callbacks.get_reminders",
            side_effect=[
                [
                    {"id": "r1", "user_settings_id": "aaa", "time": "08:00:00"},
                    {"id": "r2", "user_settings_id": "aaa", "time": "20:00:00"},
                ],
                [{"id": "r2", "user_settings_id": "aaa", "time": "20:00:00"}],
            ],
        ),
    ):
        await handle_callback(mock_upd, mock_context)

    mock_remove.assert_called_once_with("r1")
    mock_query.edit_message_text.assert_called_once()


async def test_handle_callback_add_reminder_shows_hour_picker(
    mock_context: MagicMock,
) -> None:
    """Test that add reminder callback shows hour picker keyboard."""
    from src.api.bot_callbacks import handle_callback

    mock_query = MagicMock()
    mock_query.data = "add_rem|aaa"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()
    mock_query.message.chat.id = 67890

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    await handle_callback(mock_upd, mock_context)

    mock_query.edit_message_text.assert_called_once()
    call_kwargs = mock_query.edit_message_text.call_args.kwargs
    assert "reply_markup" in call_kwargs


async def test_handle_callback_add_reminder_hour_saves(
    mock_context: MagicMock,
) -> None:
    """Test that selecting an hour from the picker adds the reminder."""
    from src.api.bot_callbacks import handle_callback

    mock_query = MagicMock()
    mock_query.data = "add_hr|14|aaa"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()
    mock_query.message.chat.id = 67890

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with (
        patch("src.api.bot_callbacks.add_reminder", return_value=True),
        patch(
            "src.api.bot_callbacks.get_reminders",
            return_value=[
                {"id": "r1", "user_settings_id": "aaa", "time": "08:00:00"},
                {"id": "r2", "user_settings_id": "aaa", "time": "14:00:00"},
                {"id": "r3", "user_settings_id": "aaa", "time": "20:00:00"},
            ],
        ),
    ):
        await handle_callback(mock_upd, mock_context)

    mock_query.edit_message_text.assert_called_once()
    text = mock_query.edit_message_text.call_args.kwargs.get(
        "text",
        (
            mock_query.edit_message_text.call_args.args[0]
            if mock_query.edit_message_text.call_args.args
            else ""
        ),
    )
    assert "14:00" in text
