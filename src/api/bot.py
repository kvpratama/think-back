"""Telegram bot interface for ThinkBack.

This module handles the Telegram bot commands and message routing.
"""

from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update, constants
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.api.bot_callbacks import handle_callback
from src.api.bot_commands import (
    chat_member_update,
    help_command,
    reminders_command,
    start_command,
    timezone_command,
)
from src.api.bot_graph import aget_graph
from src.api.bot_helpers import truncate_for_telegram
from src.core.config import get_settings
from src.db.user_settings import get_user_settings_id

logger = logging.getLogger(__name__)

load_dotenv()


async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown commands.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    if not update.message or not update.message.from_user:
        logger.debug("unknown_command invoked without message/from_user; skipping")
        return

    await update.message.reply_text("Unknown command. Use /help to see available commands.")


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle incoming text messages by invoking the agent graph directly.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    if not update.message or not update.message.from_user:
        logger.debug("handle_message invoked without message/from_user; skipping")
        return

    user_input = update.message.text
    chat_id = str(update.message.chat.id)

    # Commands are routed by their own CommandHandlers
    if user_input and user_input.startswith("/"):
        logger.debug("Command detected, bypassing message handler: %s", user_input)
        return

    if not user_input:
        return

    await process_message(
        chat_id=chat_id,
        user_id=update.message.from_user.id,
        text=user_input,
        update=update,
        context=context,
    )


async def non_private_chat_handler(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Inform users that the bot only works in private chats.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    if not update.message:
        return

    await update.message.reply_text(
        "👋 ThinkBack only works in private chats.\n\n"
        "Please message me directly to get started: /start"
    )


async def process_message(
    chat_id: str,
    user_id: int,
    text: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Process a single user message through the agent graph and reply once.

    Args:
        chat_id: Chat identifier (also used as LangGraph thread_id in private chats).
        user_id: Telegram user identifier.
        text: User's text content.
        update: Telegram update from this message.
        context: Telegram context.
    """
    try:
        user_settings_id = await asyncio.to_thread(get_user_settings_id, chat_id)
    except Exception:
        logger.exception("Failed to resolve user settings")
        if update.message:
            await update.message.reply_text(
                "Sorry, I couldn't load your account settings. Please try again."
            )
        return
    if not user_settings_id:
        if update.message:
            await update.message.reply_text("Please run /start first to set up your account.")
        return

    if not update.message:
        return

    await context.bot.send_chat_action(
        chat_id=int(chat_id),
        action=constants.ChatAction.TYPING,
    )

    telegram_char_limit = 4000
    final_response = ""

    try:
        graph = await aget_graph(context)
        # In private chats, chat_id equals user_id, so this uniquely identifies
        # the user's thread. ThinkBack is restricted to private chats only
        # (enforced via filters.ChatType.PRIVATE).
        thread_id = str(chat_id)
        config = RunnableConfig(
            {"configurable": {"thread_id": thread_id, "user_settings_id": user_settings_id}}
        )

        await graph.ainvoke(
            {"messages": [{"role": "user", "content": text}]},
            config=config,
        )

        state = await graph.aget_state(config)

        # Interrupt path: graph paused awaiting Save/Cancel confirmation
        if state.next:
            interrupt_value = state.tasks[0].interrupts[0].value
            insight = interrupt_value.get("insight", "")
            content = interrupt_value.get("content", "")
            duplicates = interrupt_value.get("duplicates", [])

            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "✅ Save",
                            callback_data=f"save_yes|{thread_id}|{chat_id}",
                        ),
                        InlineKeyboardButton(
                            "❌ Cancel",
                            callback_data=f"save_no|{thread_id}|{chat_id}",
                        ),
                    ]
                ]
            )

            confirm_parts = [
                f"💡 Save this insight?\n\n"
                f"<blockquote>{insight}</blockquote>\n\n"
                f'<i>Original: "{content}"</i>',
            ]
            if duplicates:
                confirm_parts.append("\n\n⚠️ <b>Similar memories found:</b>")
                for dup in duplicates:
                    label = "exact" if dup["match_type"] == "exact" else "similar"
                    confirm_parts.append(f"• [{label}] {dup['content']}")

            confirm_text = truncate_for_telegram(
                "\n".join(confirm_parts),
                max_len=telegram_char_limit,
            )
            try:
                await update.message.reply_text(
                    confirm_text,
                    reply_markup=keyboard,
                    parse_mode=ParseMode.HTML,
                )
            except BadRequest:
                await update.message.reply_text(
                    confirm_text,
                    reply_markup=keyboard,
                )
            return

        # No interrupt — send final assistant message
        messages = state.values.get("messages", [])
        if messages:
            final_response = messages[-1].content if hasattr(messages[-1], "content") else ""

    except Exception:
        logger.exception("Error processing message in chat %s", chat_id)
        final_response = final_response or "Sorry, something went wrong. Please try again."

    if final_response:
        safe_response = truncate_for_telegram(
            final_response,
            max_len=telegram_char_limit,
        )
        try:
            await update.message.reply_text(
                safe_response,
                parse_mode=ParseMode.HTML,
            )
        except BadRequest:
            await update.message.reply_text(
                safe_response,
            )


async def _post_init(application: Application) -> None:
    """Set bot commands after the application initializes.

    Args:
        application: The Telegram Application instance.
    """
    await application.bot.set_my_commands(
        [
            BotCommand("start", "Get started or reset your intro"),
            BotCommand("timezone", "Set your local timezone"),
            BotCommand("reminders", "Customize when insights resurface"),
            BotCommand("help", "Show available commands"),
        ]
    )


async def _post_shutdown(application: Application) -> None:
    """Clean up resources after the application shuts down (polling mode only).

    Args:
        application: The Telegram Application instance.
    """
    from src.db.checkpointer import aclose_checkpointer

    await aclose_checkpointer()


def create_application() -> Application:
    """Create and configure the Telegram bot application.

    Returns:
        Configured Telegram Application.
    """
    settings = get_settings()

    application = (
        Application.builder()
        .token(settings.telegram_bot_token.get_secret_value())
        .post_init(_post_init)
        .post_shutdown(_post_shutdown)
        .build()
    )

    application.add_handler(
        CommandHandler("start", start_command, filters=filters.ChatType.PRIVATE)
    )
    application.add_handler(
        CommandHandler("timezone", timezone_command, filters=filters.ChatType.PRIVATE)
    )
    application.add_handler(
        CommandHandler("reminders", reminders_command, filters=filters.ChatType.PRIVATE)
    )
    application.add_handler(CommandHandler("help", help_command, filters=filters.ChatType.PRIVATE))
    application.add_handler(
        MessageHandler(filters.COMMAND & filters.ChatType.PRIVATE, unknown_command)
    )
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND & filters.ChatType.PRIVATE, handle_message)
    )
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(ChatMemberHandler(chat_member_update, ChatMemberHandler.MY_CHAT_MEMBER))

    # Fallback handler for non-private chats (must be last)
    application.add_handler(MessageHandler(~filters.ChatType.PRIVATE, non_private_chat_handler))

    return application


def main() -> None:
    """Run the Telegram bot.

    Uses webhook mode when WEBHOOK_URL is set (production),
    otherwise falls back to polling (local development).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    settings = get_settings()
    app = create_application()

    if settings.webhook_url:
        if not settings.webhook_secret.get_secret_value():
            raise ValueError("WEBHOOK_SECRET must be set when WEBHOOK_URL is configured")
        logger.info("Starting webhook mode at %s/webhook", settings.webhook_url)
        app.run_webhook(
            listen="0.0.0.0",
            port=settings.port,
            url_path="/webhook",
            webhook_url=f"{settings.webhook_url}/webhook",
            secret_token=settings.webhook_secret.get_secret_value(),
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        logger.info("Starting polling mode")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
