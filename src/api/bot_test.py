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
    context.bot_data = {}
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

    mock_context.bot_data["graph"] = mock_graph

    await handle_message(mock_update, mock_context)

    assert mock_update.message is not None
    cast(Any, mock_update.message.reply_text).assert_called_once()
    mock_context.bot.edit_message_text.assert_called()


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

    mock_context.bot_data["graph"] = mock_graph

    await handle_message(mock_update, mock_context)

    # The last edit_message_text call should be the confirmation UI, not the accumulated text.
    last_call = mock_context.bot.edit_message_text.call_args
    assert "Save this insight?" in last_call.kwargs.get("text", last_call[1].get("text", ""))


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

    mock_context.bot_data["graph"] = mock_graph

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

    mock_context.bot_data["graph"] = mock_graph

    await handle_message(mock_update, mock_context)

    last_call = mock_context.bot.edit_message_text.call_args
    text = last_call.kwargs.get("text", last_call[1].get("text", ""))
    assert "Save this insight?" in text
    assert "duplicate" not in text.lower()


async def test_unknown_command(mock_update: Update, mock_context: MagicMock) -> None:
    """Test that unknown commands receive a helpful error message."""
    from src.api.bot import unknown_command

    await unknown_command(mock_update, mock_context)

    assert mock_update.message is not None
    cast(Any, mock_update.message.reply_text).assert_called_once_with(
        "Unknown command. Use /help to see available commands."
    )
