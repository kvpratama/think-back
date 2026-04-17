"""Tests for the Telegram bot interface."""

# mypy: disable-error-code="union-attr"

from collections.abc import AsyncGenerator
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


async def test_handle_message_natural_language(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that handle_message processes natural language input."""
    from src.api.bot import handle_message

    async def mock_astream(*args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"messages": [MagicMock(content="I've noted your insight.")]}},
        }

    mock_state = MagicMock()
    mock_state.next = []
    mock_state.values = {"messages": [MagicMock(content="I've noted your insight.")]}

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    with patch("src.api.bot._get_graph", return_value=mock_graph):
        await handle_message(mock_update, mock_context)

        assert mock_update.message is not None
        cast(Any, mock_update.message.reply_text).assert_called_once()
        mock_context.bot.edit_message_text.assert_called()


async def test_start_command(mock_update: Update, mock_context: MagicMock) -> None:
    """Test that start_command sends welcome message."""
    from src.api.bot import start_command

    with (
        patch("src.api.bot.upsert_user_settings", return_value=False),
        patch("src.api.bot.get_user_settings_id", return_value="aaa"),
        patch("src.api.bot.insert_default_reminders"),
    ):
        await start_command(mock_update, mock_context)

    assert mock_update.message is not None
    cast(Any, mock_update.message.reply_text).assert_called_once()


async def test_create_application_registers_handlers() -> None:
    """Test that create_application registers the expected handlers."""
    from telegram.ext import CallbackQueryHandler, CommandHandler, MessageHandler

    from src.api.bot import create_application

    with patch("src.api.bot.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.telegram_bot_token.get_secret_value.return_value = "test-token"

        app = create_application()

        handler_types = [type(h) for group in app.handlers.values() for h in group]
        assert CommandHandler in handler_types
        assert MessageHandler in handler_types
        assert CallbackQueryHandler in handler_types


async def test_handle_message_interrupt_not_overwritten(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that the finally block does not overwrite the save-confirmation UI."""
    from src.api.bot import handle_message

    async def mock_astream(*args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": MagicMock(content="Let me save that")},
        }

    mock_interrupt = MagicMock()
    mock_interrupt.value = {"insight": "Test insight", "content": "Test content"}

    mock_task = MagicMock()
    mock_task.interrupts = [mock_interrupt]

    mock_state = MagicMock()
    mock_state.next = ["some_node"]
    mock_state.tasks = [mock_task]

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    with patch("src.api.bot._get_graph", return_value=mock_graph):
        await handle_message(mock_update, mock_context)

        # The last edit_message_text call should be the confirmation UI, not the accumulated text.
        last_call = mock_context.bot.edit_message_text.call_args
        assert "Save this insight?" in last_call.kwargs.get("text", last_call[1].get("text", ""))


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


async def test_handle_callback_approved(
    mock_callback_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that approved callback resumes graph and saves memory."""
    from src.api.bot import handle_callback

    mock_graph = MagicMock()
    mock_result_msg = MagicMock()
    mock_result_msg.content = "Memory saved: Motivation follows action"
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [mock_result_msg]})

    with patch("src.api.bot._get_graph", return_value=mock_graph):
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
    from src.api.bot import handle_callback

    assert mock_callback_update.callback_query is not None
    cast(Any, mock_callback_update.callback_query).data = "save_no|67890_12345"

    mock_graph = MagicMock()
    mock_result_msg = MagicMock()
    mock_result_msg.content = "Save cancelled."
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [mock_result_msg]})

    with patch("src.api.bot._get_graph", return_value=mock_graph):
        await handle_callback(mock_callback_update, mock_context)

        call_args = mock_graph.ainvoke.call_args
        command = call_args[0][0]
        assert command.resume == {"approved": False}


async def test_handle_message_interrupt_shows_duplicates(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that the save confirmation UI surfaces duplicate memories."""
    from src.api.bot import handle_message

    async def mock_astream(*args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": MagicMock(content="Let me save that")},
        }

    mock_interrupt = MagicMock()
    mock_interrupt.value = {
        "insight": "Exercise matters",
        "content": "Exercise is good",
        "duplicates": [
            {"content": "Exercise is good", "similarity": 1.0, "match_type": "exact"},
            {
                "content": "Working out is healthy",
                "similarity": 0.91,
                "match_type": "semantic",
            },
        ],
    }

    mock_task = MagicMock()
    mock_task.interrupts = [mock_interrupt]

    mock_state = MagicMock()
    mock_state.next = ["some_node"]
    mock_state.tasks = [mock_task]

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    with patch("src.api.bot._get_graph", return_value=mock_graph):
        await handle_message(mock_update, mock_context)

        last_call = mock_context.bot.edit_message_text.call_args
        text = last_call.kwargs.get("text", last_call[1].get("text", ""))
        assert "Exercise is good" in text
        assert "Working out is healthy" in text
        assert "Similar" in text or "Duplicate" in text or "duplicate" in text


async def test_handle_message_interrupt_no_duplicates(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that the save confirmation UI works when no duplicates are found."""
    from src.api.bot import handle_message

    async def mock_astream(*args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": MagicMock(content="Let me save that")},
        }

    mock_interrupt = MagicMock()
    mock_interrupt.value = {
        "insight": "Brand new insight",
        "content": "Something new",
        "duplicates": [],
    }

    mock_task = MagicMock()
    mock_task.interrupts = [mock_interrupt]

    mock_state = MagicMock()
    mock_state.next = ["some_node"]
    mock_state.tasks = [mock_task]

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    with patch("src.api.bot._get_graph", return_value=mock_graph):
        await handle_message(mock_update, mock_context)

        last_call = mock_context.bot.edit_message_text.call_args
        text = last_call.kwargs.get("text", last_call[1].get("text", ""))
        assert "Save this insight?" in text
        assert "duplicate" not in text.lower()


async def test_start_command_upserts_settings_and_shows_timezone_picker(
    mock_update: Update, mock_context: MagicMock
) -> None:
    """Test that /start upserts user settings and shows timezone keyboard for new users."""
    from src.api.bot import start_command

    with (
        patch("src.api.bot.upsert_user_settings", return_value=True) as mock_upsert,
        patch("src.api.bot.get_user_settings_id", return_value="aaa-bbb"),
        patch("src.api.bot.insert_default_reminders") as mock_reminders,
    ):
        await start_command(mock_update, mock_context)

    mock_upsert.assert_called_once_with("67890")
    mock_reminders.assert_called_once_with("aaa-bbb")

    # Should have two reply_text calls: welcome message + timezone picker
    assert mock_update.message is not None
    calls = cast(Any, mock_update.message.reply_text).call_args_list
    assert len(calls) == 2

    # Second call should have inline keyboard with UTC offsets
    tz_call_kwargs = calls[1].kwargs if calls[1].kwargs else {}
    assert "reply_markup" in tz_call_kwargs


async def test_start_command_existing_user_no_timezone_picker(
    mock_update: Update, mock_context: MagicMock
) -> None:
    """Test that /start for existing user does not show timezone picker or insert reminders."""
    from src.api.bot import start_command

    with (
        patch("src.api.bot.upsert_user_settings", return_value=False),
        patch("src.api.bot.get_user_settings_id", return_value="aaa-bbb"),
        patch("src.api.bot.insert_default_reminders") as mock_reminders,
    ):
        await start_command(mock_update, mock_context)

    mock_reminders.assert_not_called()

    assert mock_update.message is not None
    calls = cast(Any, mock_update.message.reply_text).call_args_list
    assert len(calls) == 1  # Only welcome message


async def test_handle_timezone_callback(
    mock_context: MagicMock,
) -> None:
    """Test that timezone callback updates the user's timezone."""
    from src.api.bot import handle_callback

    mock_query = MagicMock()
    mock_query.data = "tz|7|67890"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()

    mock_update = MagicMock(spec=Update)
    mock_update.callback_query = mock_query
    mock_update.message = None

    with patch("src.api.bot.update_timezone") as mock_update_tz:
        await handle_callback(mock_update, mock_context)

    # UTC+7 → Etc/GMT-7 (POSIX inverts the sign)
    mock_update_tz.assert_called_once_with("67890", "Etc/GMT-7")
    mock_query.edit_message_text.assert_called_once()


async def test_timezone_command_shows_offset_picker(
    mock_update: Update, mock_context: MagicMock
) -> None:
    """Test that /timezone shows the UTC offset keyboard."""
    from src.api.bot import timezone_command

    await timezone_command(mock_update, mock_context)

    assert mock_update.message is not None
    call = cast(Any, mock_update.message.reply_text).call_args
    assert "UTC offset" in call.args[0] or "UTC offset" in call.kwargs.get("text", "")
    assert "reply_markup" in call.kwargs


async def test_reminders_command_shows_current_reminders(
    mock_update: Update, mock_context: MagicMock
) -> None:
    """Test that /reminders shows current reminder times with remove buttons."""
    from src.api.bot import reminders_command

    with (
        patch("src.api.bot.get_user_settings_id", return_value="aaa"),
        patch(
            "src.api.bot.get_reminders",
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
    from src.api.bot import reminders_command

    with patch("src.api.bot.get_user_settings_id", return_value=None):
        await reminders_command(mock_update, mock_context)

    assert mock_update.message is not None
    call = cast(Any, mock_update.message.reply_text).call_args
    text = call.args[0] if call.args else call.kwargs.get("text", "")
    assert "/start" in text


async def test_handle_callback_remove_reminder(
    mock_context: MagicMock,
) -> None:
    """Test that remove reminder callback removes and refreshes the list."""
    from src.api.bot import handle_callback

    mock_query = MagicMock()
    mock_query.data = "rm_rem|0"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()
    mock_query.message.chat.id = 67890

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with (
        patch("src.api.bot.get_user_settings_id", return_value="aaa"),
        patch("src.api.bot.remove_reminder") as mock_remove,
        patch(
            "src.api.bot.get_reminders",
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
    from src.api.bot import handle_callback

    mock_query = MagicMock()
    mock_query.data = "add_rem"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()
    mock_query.message.chat.id = 67890

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with patch("src.api.bot.get_user_settings_id", return_value="aaa"):
        await handle_callback(mock_upd, mock_context)

    mock_query.edit_message_text.assert_called_once()
    call_kwargs = mock_query.edit_message_text.call_args.kwargs
    assert "reply_markup" in call_kwargs


async def test_handle_callback_add_reminder_hour_saves(
    mock_context: MagicMock,
) -> None:
    """Test that selecting an hour from the picker adds the reminder."""
    from src.api.bot import handle_callback

    mock_query = MagicMock()
    mock_query.data = "add_hr|14"
    mock_query.answer = AsyncMock()
    mock_query.edit_message_text = AsyncMock()
    mock_query.message = MagicMock()
    mock_query.message.chat.id = 67890

    mock_upd = MagicMock(spec=Update)
    mock_upd.callback_query = mock_query
    mock_upd.message = None

    with (
        patch("src.api.bot.get_user_settings_id", return_value="aaa"),
        patch("src.api.bot.add_reminder", return_value=True),
        patch(
            "src.api.bot.get_reminders",
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


async def test_help_command_lists_commands(mock_update: Update, mock_context: MagicMock) -> None:
    """Test that /help lists all available commands."""
    from src.api.bot import help_command

    await help_command(mock_update, mock_context)

    assert mock_update.message is not None
    call = cast(Any, mock_update.message.reply_text).call_args
    text = call.args[0] if call.args else call.kwargs.get("text", "")
    assert "/start" in text
    assert "/timezone" in text
    assert "/reminders" in text
    assert "/help" in text
