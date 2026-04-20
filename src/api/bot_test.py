"""Tests for the Telegram bot interface."""

# mypy: disable-error-code="union-attr"

from collections.abc import AsyncGenerator
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Update, User

from src.api.bot_batcher import MessageBatcher


@pytest.fixture(autouse=True)
async def reset_message_batcher(mock_context: MagicMock) -> AsyncGenerator[None, None]:
    """Reset the message_batcher in mock_context before and after each test."""
    batcher = mock_context.bot_data["message_batcher"]

    await batcher.shutdown()
    yield
    await batcher.shutdown()


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
    # Initialize a real batcher for tests that need it
    from src.api.bot import process_batch

    batcher = MessageBatcher(timeout=1.0, process_callback=process_batch)
    context.bot_data = {"message_batcher": batcher}
    return context


async def test_handle_message_natural_language(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """Test that handle_message processes natural language input through batching."""

    from src.api.bot import handle_message

    message_batcher = mock_context.bot_data["message_batcher"]

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
    mock_graph.astream_events = MagicMock(side_effect=mock_astream)
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    mock_context.bot_data["graph"] = mock_graph

    with patch(
        "src.api.bot.get_user_settings_id", return_value="settings-1"
    ) as mock_get_user_settings_id:
        await handle_message(mock_update, mock_context)

        # Process batch immediately instead of waiting for timeout
        assert mock_update.message is not None
        assert mock_update.message.from_user is not None
        message_batcher = mock_context.bot_data["message_batcher"]
        await message_batcher.flush(
            str(mock_update.message.chat.id),
            mock_update.message.from_user.id,
        )

    assert mock_update.message is not None
    mock_get_user_settings_id.assert_called_once_with(str(mock_update.message.chat.id))
    mock_graph.astream_events.assert_called_once()
    config = mock_graph.astream_events.call_args.kwargs.get("config")
    assert config is not None
    assert config["configurable"]["user_settings_id"] == "settings-1"

    cast(Any, mock_update.message.reply_text).assert_called_once()
    mock_context.bot.edit_message_text.assert_called()


async def test_split_messages_combined(mock_context: MagicMock) -> None:
    """Test that rapid successive messages are combined before processing."""

    from src.api.bot import handle_message

    message_batcher = mock_context.bot_data["message_batcher"]

    update1 = MagicMock(spec=Update)
    update1.message = MagicMock()
    update1.message.text = "First part of long message"
    update1.message.chat.id = 123
    update1.message.from_user = MagicMock(spec=User)
    update1.message.from_user.id = 1
    update1.message.reply_text = AsyncMock(return_value=MagicMock(message_id=1))

    update2 = MagicMock(spec=Update)
    update2.message = MagicMock()
    update2.message.text = "Second part of long message"
    update2.message.chat.id = 123
    update2.message.from_user = MagicMock(spec=User)
    update2.message.from_user.id = 1
    update2.message.reply_text = AsyncMock(return_value=MagicMock(message_id=2))

    async def mock_astream(*args: Any, **kwargs: Any) -> AsyncGenerator[dict[str, Any], None]:
        yield {
            "event": "on_chain_end",
            "name": "LangGraph",
            "data": {"output": {"messages": [MagicMock(content="Combined response")]}},
        }

    mock_state = MagicMock()
    mock_state.next = []
    mock_state.values = {"messages": [MagicMock(content="Combined response")]}

    mock_graph = MagicMock()
    mock_graph.astream_events = MagicMock(side_effect=mock_astream)
    mock_graph.aget_state = AsyncMock(return_value=mock_state)
    mock_context.bot_data["graph"] = mock_graph

    with patch("src.api.bot.get_user_settings_id", return_value="settings-1"):
        await handle_message(update1, mock_context)
        await handle_message(update2, mock_context)

        # Process batch immediately instead of waiting for timeout
        message_batcher = mock_context.bot_data["message_batcher"]
        assert update1.message is not None
        assert update1.message.from_user is not None
        await message_batcher.flush(
            str(update1.message.chat.id),
            update1.message.from_user.id,
        )

    mock_graph.astream_events.assert_called_once()
    messages = mock_graph.astream_events.call_args.args[0]["messages"]
    assert messages[0]["content"] == "First part of long message\n\nSecond part of long message"


async def test_create_application_sets_post_init() -> None:
    """Test that create_application sets a post_init callback."""
    from src.api.bot import create_application

    with patch("src.api.bot.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.telegram_bot_token.get_secret_value.return_value = "test-token"

        app = create_application()

    assert app.post_init is not None


async def test_post_init_sets_bot_commands() -> None:
    """Test that post_init registers the four bot commands."""
    from src.api.bot import create_application

    with patch("src.api.bot.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.telegram_bot_token.get_secret_value.return_value = "test-token"

        app = create_application()

    mock_bot = AsyncMock()
    mock_app = MagicMock()
    mock_app.bot = mock_bot

    assert app.post_init is not None
    await app.post_init(mock_app)

    mock_bot.set_my_commands.assert_called_once()
    commands = mock_bot.set_my_commands.call_args[0][0]
    command_names = [c.command for c in commands]
    assert command_names == ["start", "timezone", "reminders", "help"]


async def test_create_application_registers_handlers() -> None:
    """Test that create_application registers the expected handlers."""
    from telegram.ext import (
        CallbackQueryHandler,
        ChatMemberHandler,
        CommandHandler,
        MessageHandler,
    )

    from src.api.bot import create_application

    with patch("src.api.bot.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.telegram_bot_token.get_secret_value.return_value = "test-token"

        app = create_application()

        handler_types = [type(h) for group in app.handlers.values() for h in group]
        assert CommandHandler in handler_types
        assert MessageHandler in handler_types
        assert CallbackQueryHandler in handler_types
        assert ChatMemberHandler in handler_types


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
    mock_interrupt.value = {"insight": "Test insight", "content": "Test content", "duplicates": []}

    mock_task = MagicMock()
    mock_task.interrupts = [mock_interrupt]

    mock_state = MagicMock()
    mock_state.next = ["some_node"]
    mock_state.tasks = [mock_task]

    mock_graph = MagicMock()
    mock_graph.astream_events = MagicMock(side_effect=mock_astream)
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    mock_context.bot_data["graph"] = mock_graph

    with patch(
        "src.api.bot.get_user_settings_id", return_value="settings-1"
    ) as mock_get_user_settings_id:
        await handle_message(mock_update, mock_context)

        # Process batch immediately instead of waiting for timeout
        message_batcher = mock_context.bot_data["message_batcher"]
        assert mock_update.message is not None
        assert mock_update.message.from_user is not None
        await message_batcher.flush(
            str(mock_update.message.chat.id),
            mock_update.message.from_user.id,
        )

    assert mock_update.message is not None
    mock_get_user_settings_id.assert_called_once_with(str(mock_update.message.chat.id))
    mock_graph.astream_events.assert_called_once()
    config = mock_graph.astream_events.call_args.kwargs.get("config")
    assert config is not None
    assert config["configurable"]["user_settings_id"] == "settings-1"

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
    mock_graph.astream_events = MagicMock(side_effect=mock_astream)
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    mock_context.bot_data["graph"] = mock_graph

    with patch(
        "src.api.bot.get_user_settings_id", return_value="settings-1"
    ) as mock_get_user_settings_id:
        await handle_message(mock_update, mock_context)

        # Process batch immediately instead of waiting for timeout
        message_batcher = mock_context.bot_data["message_batcher"]
        assert mock_update.message is not None
        assert mock_update.message.from_user is not None
        await message_batcher.flush(
            str(mock_update.message.chat.id),
            mock_update.message.from_user.id,
        )

    assert mock_update.message is not None
    mock_get_user_settings_id.assert_called_once_with(str(mock_update.message.chat.id))
    mock_graph.astream_events.assert_called_once()
    config = mock_graph.astream_events.call_args.kwargs.get("config")
    assert config is not None
    assert config["configurable"]["user_settings_id"] == "settings-1"

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
    mock_graph.astream_events = MagicMock(side_effect=mock_astream)
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    mock_context.bot_data["graph"] = mock_graph

    with patch(
        "src.api.bot.get_user_settings_id", return_value="settings-1"
    ) as mock_get_user_settings_id:
        await handle_message(mock_update, mock_context)

        # Process batch immediately instead of waiting for timeout
        message_batcher = mock_context.bot_data["message_batcher"]
        assert mock_update.message is not None
        assert mock_update.message.from_user is not None
        await message_batcher.flush(
            str(mock_update.message.chat.id),
            mock_update.message.from_user.id,
        )

    assert mock_update.message is not None
    mock_get_user_settings_id.assert_called_once_with(str(mock_update.message.chat.id))
    mock_graph.astream_events.assert_called_once()
    config = mock_graph.astream_events.call_args.kwargs.get("config")
    assert config is not None
    assert config["configurable"]["user_settings_id"] == "settings-1"

    last_call = mock_context.bot.edit_message_text.call_args
    text = last_call.kwargs.get("text", last_call[1].get("text", ""))
    assert "Save this insight?" in text
    assert "duplicate" not in text.lower()


def test_main_uses_polling_when_no_webhook_url() -> None:
    """Test that main() uses run_polling when WEBHOOK_URL is not set."""
    from src.api.bot import main

    mock_app = MagicMock()
    mock_settings = MagicMock()
    mock_settings.webhook_url = ""

    with (
        patch("src.api.bot.create_application", return_value=mock_app),
        patch("src.api.bot.get_settings", return_value=mock_settings),
    ):
        main()

    mock_app.run_polling.assert_called_once_with(allowed_updates=Update.ALL_TYPES)
    mock_app.run_webhook.assert_not_called()


def test_main_uses_webhook_when_webhook_url_set() -> None:
    """Test that main() uses run_webhook when WEBHOOK_URL is set."""
    from src.api.bot import main

    mock_app = MagicMock()
    mock_settings = MagicMock()
    mock_settings.webhook_url = "https://my-app.up.railway.app"
    mock_settings.webhook_secret.get_secret_value.return_value = "test-secret"
    mock_settings.port = 8000

    with (
        patch("src.api.bot.create_application", return_value=mock_app),
        patch("src.api.bot.get_settings", return_value=mock_settings),
    ):
        main()

    mock_app.run_webhook.assert_called_once_with(
        listen="0.0.0.0",
        port=8000,
        url_path="/webhook",
        webhook_url="https://my-app.up.railway.app/webhook",
        secret_token="test-secret",
        allowed_updates=Update.ALL_TYPES,
    )
    mock_app.run_polling.assert_not_called()


def test_main_raises_when_webhook_url_set_but_secret_empty() -> None:
    """Test that main() raises ValueError when WEBHOOK_URL is set but WEBHOOK_SECRET is empty."""
    from src.api.bot import main

    mock_app = MagicMock()
    mock_settings = MagicMock()
    mock_settings.webhook_url = "https://my-app.up.railway.app"
    mock_settings.webhook_secret.get_secret_value.return_value = ""
    mock_settings.port = 8000

    with (
        patch("src.api.bot.create_application", return_value=mock_app),
        patch("src.api.bot.get_settings", return_value=mock_settings),
        pytest.raises(
            ValueError, match="WEBHOOK_SECRET must be set when WEBHOOK_URL is configured"
        ),
    ):
        main()


async def test_unknown_command(mock_update: Update, mock_context: MagicMock) -> None:
    """Test that unknown commands receive a helpful error message."""
    from src.api.bot import unknown_command

    await unknown_command(mock_update, mock_context)

    assert mock_update.message is not None
    cast(Any, mock_update.message.reply_text).assert_called_once_with(
        "Unknown command. Use /help to see available commands."
    )


async def test_commands_bypass_batching(mock_update: Update, mock_context: MagicMock) -> None:
    """Test that commands are not batched."""
    from src.api.bot import handle_message

    message_batcher = mock_context.bot_data["message_batcher"]

    # Clear batcher to ensure clean state
    await message_batcher.shutdown()

    # Set command text and unique chat ID to avoid shared state flakiness
    assert mock_update.message is not None
    mock_update.message.text = "/help"
    mock_update.message.chat.id = 99999

    # Call handle_message
    await handle_message(mock_update, mock_context)

    # Verify message was not added to batcher
    assert mock_update.message is not None
    assert mock_update.message.from_user is not None
    key = (str(mock_update.message.chat.id), mock_update.message.from_user.id)
    assert key not in message_batcher.buffers
