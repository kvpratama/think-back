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
    mock_query.data = "save_yes|67890_12345|67890"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.from_user = mock_user

    mock_message = MagicMock()
    mock_message.chat.id = 67890
    mock_query.message = mock_message

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

    with patch(
        "src.api.bot_callbacks.get_user_settings_id", return_value="settings-1"
    ) as mock_get_id:
        await handle_callback(mock_callback_update, mock_context)

    mock_get_id.assert_called_once_with("67890")

    mock_graph.ainvoke.assert_called_once()
    call_args = mock_graph.ainvoke.call_args
    command = call_args[0][0]
    config = call_args.kwargs.get("config", {})

    assert command.resume == {"approved": True}
    assert config["configurable"]["thread_id"] == "67890_12345"
    assert config["configurable"]["user_settings_id"] == "settings-1"

    assert mock_callback_update.callback_query is not None
    cast(Any, mock_callback_update.callback_query.edit_message_text).assert_called_once()


async def test_handle_callback_cancelled(
    mock_callback_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that cancelled callback resumes graph without saving."""
    from src.api.bot_callbacks import handle_callback

    assert mock_callback_update.callback_query is not None
    cast(Any, mock_callback_update.callback_query).data = "save_no|67890_12345|67890"

    mock_graph = MagicMock()
    mock_result_msg = MagicMock()
    mock_result_msg.content = "Save cancelled."
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [mock_result_msg]})

    mock_context.bot_data["graph"] = mock_graph

    with patch(
        "src.api.bot_callbacks.get_user_settings_id", return_value="settings-1"
    ) as mock_get_id:
        await handle_callback(mock_callback_update, mock_context)

    mock_get_id.assert_called_once_with("67890")

    call_args = mock_graph.ainvoke.call_args
    command = call_args[0][0]
    config = call_args.kwargs.get("config", {})

    assert command.resume == {"approved": False}
    assert config["configurable"]["thread_id"] == "67890_12345"
    assert config["configurable"]["user_settings_id"] == "settings-1"


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


async def test_handle_callback_add_reminder_shows_limit_reached_message(
    mock_context: MagicMock,
) -> None:
    """Test that limit reached shows specific message."""
    from src.api.bot_callbacks import handle_callback
    from src.db.user_settings import AddReminderResult

    mock_query = MagicMock()
    mock_query.data = "add_hr|14|aaa"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with patch("src.api.bot_callbacks.add_reminder", return_value=AddReminderResult.LIMIT_REACHED):
        await handle_callback(mock_upd, mock_context)

    mock_query.edit_message_text.assert_called_once()
    text = mock_query.edit_message_text.call_args.kwargs.get("text", "")
    assert "Maximum 5 reminders reached" in text


async def test_handle_callback_add_reminder_shows_db_error_message(
    mock_context: MagicMock,
) -> None:
    """Test that DB error shows generic error message."""
    from src.api.bot_callbacks import handle_callback
    from src.db.user_settings import AddReminderResult

    mock_query = MagicMock()
    mock_query.data = "add_hr|14|aaa"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with patch("src.api.bot_callbacks.add_reminder", return_value=AddReminderResult.DB_ERROR):
        await handle_callback(mock_upd, mock_context)

    mock_query.edit_message_text.assert_called_once()
    text = mock_query.edit_message_text.call_args.kwargs.get("text", "")
    assert "Couldn't add reminder" in text


async def test_handle_timezone_onboarding_sends_reminders_followup(
    mock_context: MagicMock,
) -> None:
    """Test that timezone callback with onboarding flag sends reminders panel."""
    from src.api.bot_callbacks import handle_callback

    mock_query = MagicMock()
    mock_query.data = "tz|7|67890|1"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()
    mock_query.message.chat.send_message = AsyncMock()

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with (
        patch("src.api.bot_callbacks.update_timezone"),
        patch("src.api.bot_callbacks.get_user_settings_id", return_value="settings-1"),
        patch(
            "src.api.bot_callbacks.get_reminders",
            return_value=[{"id": "r1", "time": "12:00:00"}],
        ),
    ):
        await handle_callback(mock_upd, mock_context)

    mock_query.message.chat.send_message.assert_called_once()
    call_kwargs = mock_query.message.chat.send_message.call_args.kwargs
    assert "reply_markup" in call_kwargs


async def test_handle_timezone_no_onboarding_skips_reminders_followup(
    mock_context: MagicMock,
) -> None:
    """Test that timezone callback without onboarding flag skips reminders panel."""
    from src.api.bot_callbacks import handle_callback

    mock_query = MagicMock()
    mock_query.data = "tz|7|67890|0"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()
    mock_query.message.chat.send_message = AsyncMock()

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with patch("src.api.bot_callbacks.update_timezone"):
        await handle_callback(mock_upd, mock_context)

    mock_query.message.chat.send_message.assert_not_called()


async def test_handle_rem_done_onboarding_sends_final_message(
    mock_context: MagicMock,
) -> None:
    """Test that rem_done with onboarding flag sends the final onboarding message."""
    from src.api.bot_callbacks import handle_callback

    mock_query = MagicMock()
    mock_query.data = "rem_done|aaa|1"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()
    mock_query.message.chat.send_message = AsyncMock()

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with patch(
        "src.api.bot_callbacks.get_reminders",
        return_value=[{"id": "r1", "time": "12:00:00"}],
    ):
        await handle_callback(mock_upd, mock_context)

    mock_query.edit_message_text.assert_called_once()
    edit_text = mock_query.edit_message_text.call_args.kwargs.get("text", "")
    assert "12:00" in edit_text

    mock_query.message.chat.send_message.assert_called_once()
    final_text = mock_query.message.chat.send_message.call_args.kwargs.get("text", "")
    assert "first thought" in final_text


async def test_handle_rem_done_no_onboarding_skips_final_message(
    mock_context: MagicMock,
) -> None:
    """Test that rem_done without onboarding flag does not send the final message."""
    from src.api.bot_callbacks import handle_callback

    mock_query = MagicMock()
    mock_query.data = "rem_done|aaa|0"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()
    mock_query.message.chat.send_message = AsyncMock()

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with patch(
        "src.api.bot_callbacks.get_reminders",
        return_value=[{"id": "r1", "time": "12:00:00"}],
    ):
        await handle_callback(mock_upd, mock_context)

    mock_query.edit_message_text.assert_called_once()
    mock_query.message.chat.send_message.assert_not_called()


async def test_handle_callback_unknown_action_does_not_invoke_graph(
    mock_context: MagicMock,
) -> None:
    """Test that unknown callback actions do not trigger graph.ainvoke."""
    from src.api.bot_callbacks import handle_callback

    mock_query = MagicMock()
    mock_query.data = "unknown_action|thread123"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock()
    mock_context.bot_data["graph"] = mock_graph

    await handle_callback(mock_upd, mock_context)

    # Unknown actions should NOT trigger graph.ainvoke
    mock_graph.ainvoke.assert_not_called()
