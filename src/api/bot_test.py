"""Tests for the Telegram bot interface."""

# mypy: disable-error-code="union-attr"

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Update, User


@pytest.fixture
def mock_update() -> Update:
    """Create a mock Telegram Update."""
    mock_user = MagicMock(spec=User)
    mock_user.id = 12345

    mock_message = MagicMock()
    mock_message.from_user = mock_user
    mock_message.text = "/save test memory"
    mock_message.reply_text = AsyncMock()

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


async def test_handle_message_save_command(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that handle_message processes save command."""
    from src.api.bot import handle_message

    async def mock_astream(*args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"response": "Memory saved."}},
        }

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream

    with patch("src.api.bot._get_graph", return_value=mock_graph):
        await handle_message(mock_update, mock_context)

        # Should call reply_text for "Thinking..." and then edit_message_text for the result
        assert mock_update.effective_message.reply_text.call_count == 1
        mock_context.bot.edit_message_text.assert_called()


async def test_handle_message_ask_command(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that handle_message processes ask command."""
    from src.api.bot import handle_message

    mock_update.effective_message.text = "/ask What do I know about habits?"
    mock_update.message.text = "/ask What do I know about habits?"

    async def mock_astream(*args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": MagicMock(content="test ")},
        }
        yield {
            "event": "on_chat_model_stream",
            "data": {"chunk": MagicMock(content="memory")},
        }
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"response": "From your saved memories: test memory"}},
        }

    mock_graph = MagicMock()
    mock_graph.astream_events = mock_astream

    with patch("src.api.bot._get_graph", return_value=mock_graph):
        await handle_message(mock_update, mock_context)

        assert mock_update.effective_message.reply_text.call_count == 1
        mock_context.bot.edit_message_text.assert_called()


async def test_handle_message_unknown_command(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that handle_message handles unknown commands."""
    from src.api.bot import handle_message

    mock_update.effective_message.text = "/unknown command"
    mock_update.message.text = "/unknown command"

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "user_input": "/unknown command",
            "cleaned_input": "/unknown command",
            "intent": None,
            "memories": [],
            "response": "Unknown command. Use /save to save knowledge or /ask to query.",
            "error": "Unknown command.",
        }
    )

    with patch("src.api.bot._get_graph", return_value=mock_graph):
        await handle_message(mock_update, mock_context)

        mock_update.effective_message.reply_text.assert_called_once()


async def test_start_command(mock_update: Update, mock_context: MagicMock) -> None:
    """Test that start_command sends welcome message."""
    from src.api.bot import start_command

    await start_command(mock_update, mock_context)

    mock_update.effective_message.reply_text.assert_called_once()
