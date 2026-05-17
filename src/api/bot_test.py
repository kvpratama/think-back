"""Tests for the Telegram bot interface."""

# mypy: disable-error-code="union-attr"

from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from telegram import Update, User
from telegram.constants import ChatType
from telegram.ext import CommandHandler, MessageHandler


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
    """handle_message should invoke the graph once and send one reply."""

    from src.api.bot import handle_message

    final_ai = MagicMock()
    final_ai.content = "I've noted your insight."

    mock_state = MagicMock()
    mock_state.next = []
    mock_state.values = {"messages": [final_ai]}

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"messages": [final_ai]})
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    with (
        patch("src.api.bot.get_user_settings_id", return_value="settings-1") as mock_uid,
        patch("src.api.bot.aget_graph", AsyncMock(return_value=mock_graph)),
    ):
        await handle_message(mock_update, mock_context)

    assert mock_update.message is not None
    mock_uid.assert_called_once_with(str(mock_update.message.chat.id))
    mock_graph.ainvoke.assert_awaited_once()
    config = mock_graph.ainvoke.call_args.kwargs.get("config")
    assert config is not None
    assert config["configurable"]["user_settings_id"] == "settings-1"
    assert config["configurable"]["thread_id"] == str(mock_update.message.chat.id)

    # Final response sent via reply_text exactly once
    assert cast(Any, mock_update.message.reply_text).call_count == 1


async def test_process_message_handles_save_interrupt(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """When the graph interrupts with a save prompt, reply with the Save/Cancel keyboard."""

    from src.api.bot import process_message

    mock_state = MagicMock()
    mock_state.next = ("save_node",)
    interrupt = MagicMock()
    interrupt.value = {
        "insight": "Motivation follows action",
        "content": "I noticed motivation follows action",
        "duplicates": [],
    }
    mock_state.tasks = [MagicMock(interrupts=[interrupt])]

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"messages": []})
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    assert mock_update.message is not None
    assert mock_update.message.from_user is not None
    with (
        patch("src.api.bot.get_user_settings_id", return_value="settings-1"),
        patch("src.api.bot.aget_graph", AsyncMock(return_value=mock_graph)),
    ):
        chat_id = str(mock_update.message.chat.id)
        await process_message(
            chat_id=chat_id,
            user_id=mock_update.message.from_user.id,
            text="anything",
            update=mock_update,
            context=mock_context,
        )

    last_call = cast(Any, mock_update.message.reply_text).call_args_list[-1]
    text_arg = last_call.args[0] if last_call.args else last_call.kwargs.get("text", "")
    assert "Save this insight?" in text_arg


async def test_process_message_interrupt_shows_duplicates(
    mock_update: Update,
    mock_context: MagicMock,
) -> None:
    """The save confirmation UI should surface duplicate memories."""

    from src.api.bot import process_message

    interrupt = MagicMock()
    interrupt.value = {
        "insight": "Exercise matters",
        "content": "Exercise is good",
        "duplicates": [
            {"content": "Exercise is good", "similarity": 1.0, "match_type": "exact"},
            {"content": "Working out is healthy", "similarity": 0.91, "match_type": "semantic"},
        ],
    }
    mock_state = MagicMock()
    mock_state.next = ("save_node",)
    mock_state.tasks = [MagicMock(interrupts=[interrupt])]

    mock_graph = MagicMock()
    mock_graph.ainvoke = AsyncMock(return_value={"messages": []})
    mock_graph.aget_state = AsyncMock(return_value=mock_state)

    assert mock_update.message is not None
    assert mock_update.message.from_user is not None
    with (
        patch("src.api.bot.get_user_settings_id", return_value="settings-1"),
        patch("src.api.bot.aget_graph", AsyncMock(return_value=mock_graph)),
    ):
        chat_id = str(mock_update.message.chat.id)
        await process_message(
            chat_id=chat_id,
            user_id=mock_update.message.from_user.id,
            text="anything",
            update=mock_update,
            context=mock_context,
        )

    last_call = cast(Any, mock_update.message.reply_text).call_args_list[-1]
    text_arg = last_call.args[0] if last_call.args else last_call.kwargs.get("text", "")
    assert "Exercise is good" in text_arg
    assert "Working out is healthy" in text_arg
    assert "Similar" in text_arg or "duplicate" in text_arg.lower()


async def test_create_application_sets_post_init() -> None:
    """create_application should set a post_init callback (used in polling mode)."""
    from src.api.bot import create_application

    with patch("src.api.bot.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        mock_settings.return_value.telegram_bot_token.get_secret_value.return_value = "test-token"

        app = create_application()

    assert app.post_init is not None


async def test_post_init_sets_bot_commands() -> None:
    """post_init should register the four bot commands when polling."""
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
    """create_application should register the expected handler types."""
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


async def test_post_shutdown_closes_checkpointer() -> None:
    """_post_shutdown calls aclose_checkpointer (used by polling mode)."""
    from src.api.bot import create_application

    mock_settings = MagicMock()
    mock_settings.telegram_bot_token.get_secret_value.return_value = "test-token"

    with patch("src.api.bot.get_settings", return_value=mock_settings):
        app = create_application()

    mock_app = MagicMock()
    mock_app.bot_data = {}

    with patch("src.db.checkpointer.aclose_checkpointer", new_callable=AsyncMock) as mock_close:
        assert app.post_shutdown is not None
        await app.post_shutdown(mock_app)

    mock_close.assert_awaited_once()


def test_main_uses_polling_when_no_webhook_url() -> None:
    """main() uses run_polling when WEBHOOK_URL is not set."""
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
    """main() uses run_webhook when WEBHOOK_URL is set (self-hosted fallback)."""
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
    """main() raises ValueError when WEBHOOK_URL is set but WEBHOOK_SECRET is empty."""
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
    """Unknown commands receive a helpful error message."""
    from src.api.bot import unknown_command

    await unknown_command(mock_update, mock_context)

    assert mock_update.message is not None
    cast(Any, mock_update.message.reply_text).assert_called_once_with(
        "Unknown command. Use /help to see available commands."
    )


@pytest.mark.asyncio
async def test_handlers_reject_group_chats() -> None:
    """Handlers with ChatType.PRIVATE filter reject group messages."""
    from src.api.bot import create_application

    mock_settings = MagicMock()
    mock_settings.telegram_bot_token.get_secret_value.return_value = "test-token"

    with patch("src.api.bot.get_settings", return_value=mock_settings):
        app = create_application()

    for handler in app.handlers[0]:
        if not isinstance(handler, (CommandHandler, MessageHandler)):
            continue
        filter_str = str(handler.filters)
        if isinstance(handler, CommandHandler):
            assert "PRIVATE" in filter_str, (
                f"CommandHandler for {handler.commands} missing ChatType.PRIVATE filter"
            )
        elif isinstance(handler, MessageHandler):
            if "NOT" not in filter_str:
                assert "PRIVATE" in filter_str, (
                    f"MessageHandler with filters={filter_str} missing ChatType.PRIVATE"
                )


@pytest.mark.asyncio
async def test_fallback_handler_for_non_private_chats() -> None:
    """Fallback handler sends an error message for non-private chats."""
    from src.api.bot import non_private_chat_handler

    user = MagicMock()
    user.id = 123

    group_chat = MagicMock()
    group_chat.id = 456
    group_chat.type = ChatType.GROUP

    message = MagicMock()
    message.chat = group_chat
    message.from_user = user
    message.text = "Hello bot"
    message.reply_text = AsyncMock()

    update = MagicMock()
    update.message = message

    context = MagicMock()

    await non_private_chat_handler(update, context)

    message.reply_text.assert_called_once()
    call_args = message.reply_text.call_args[0][0]
    assert "private chats" in call_args.lower()
    assert "/start" in call_args
