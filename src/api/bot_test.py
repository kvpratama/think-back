"""Tests for the Telegram bot interface."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Message, Update, User


@pytest.fixture
def mock_update() -> Update:
    """Create a mock Telegram Update."""
    mock_user = MagicMock(spec=User)
    mock_user.id = 12345

    mock_message = MagicMock(spec=Message)
    mock_message.from_user = mock_user
    mock_message.text = "/save test memory"
    mock_message.chat_id = 12345
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


@pytest.mark.asyncio
async def test_handle_message_save_command(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that handle_message processes save command."""
    from src.api.bot import handle_message

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "user_input": "/save test memory",
            "intent": "save",
            "memories": [],
            "response": "Memory saved.",
            "error": None,
        }
    )

    with patch("src.api.bot.graph", mock_graph):
        await handle_message(mock_update, mock_context)

        mock_update.effective_message.reply_text.assert_called_once()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_handle_message_ask_command(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that handle_message processes ask command."""
    from src.api.bot import handle_message

    mock_update.effective_message.text = "/ask What do I know about habits?"  # type: ignore[union-attr]
    mock_update.message.text = "/ask What do I know about habits?"

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "user_input": "/ask What do I know about habits?",
            "intent": "query",
            "memories": [{"content": "test memory"}],
            "response": "From your saved memories: test memory",
            "error": None,
        }
    )

    with patch("src.api.bot.graph", mock_graph):
        await handle_message(mock_update, mock_context)

        mock_update.effective_message.reply_text.assert_called_once()  # type: ignore[union-attr]


@pytest.mark.asyncio
async def test_handle_message_unknown_command(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that handle_message handles unknown commands."""
    from src.api.bot import handle_message

    mock_update.effective_message.text = "/unknown command"  # type: ignore[union-attr]
    mock_update.message.text = "/unknown command"

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(
        return_value={
            "user_input": "/unknown command",
            "intent": None,
            "memories": [],
            "response": "Unknown command. Use /save to save knowledge or /ask to query.",
            "error": "Unknown command.",
        }
    )

    with patch("src.api.bot.graph", mock_graph):
        await handle_message(mock_update, mock_context)

        mock_update.effective_message.reply_text.assert_called_once()  # type: ignore[union-attr]


def test_start_command(mock_update: Update, mock_context: MagicMock) -> None:
    """Test that start_command sends welcome message."""
    import asyncio

    from src.api.bot import start_command

    asyncio.run(start_command(mock_update, mock_context))

    mock_update.effective_message.reply_text.assert_called_once()  # type: ignore[union-attr]
