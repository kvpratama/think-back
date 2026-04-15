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

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream

    with patch("src.api.bot._get_graph", return_value=mock_graph):
        await handle_message(mock_update, mock_context)

        assert mock_update.message is not None
        cast(Any, mock_update.message.reply_text).assert_called_once()
        mock_context.bot.edit_message_text.assert_called()


async def test_start_command(mock_update: Update, mock_context: MagicMock) -> None:
    """Test that start_command sends welcome message."""
    from src.api.bot import start_command

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
