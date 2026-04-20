"""Telegram bot interface for ThinkBack.

This module handles the Telegram bot commands and message routing.
"""

from __future__ import annotations

import asyncio
import logging
import time

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from telegram import BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    ChatMemberHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.api.bot_batcher import MessageBatcher
from src.api.bot_callbacks import handle_callback
from src.api.bot_commands import (
    chat_member_update,
    help_command,
    reminders_command,
    start_command,
    timezone_command,
)
from src.api.bot_graph import get_graph
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
    """Handle incoming messages and route to the message batcher.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    if not update.message or not update.message.from_user:
        logger.debug("handle_message invoked without message/from_user; skipping")
        return

    user_input = update.message.text
    chat_id = str(update.message.chat.id)

    # Commands bypass batching for immediate response
    if user_input and user_input.startswith("/"):
        logger.debug("Command detected, bypassing batching: %s", user_input)
        return

    # Skip if no text
    if not user_input:
        return

    # Add message to batcher
    message_batcher = context.bot_data["message_batcher"]
    await message_batcher.add_message(
        chat_id=chat_id,
        text=user_input,
        update=update,
        context=context,
    )


async def process_batch(
    chat_id: str,
    user_id: int,
    combined_text: str,
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Process a batch of combined messages through the agent graph.

    Args:
        chat_id: Chat identifier.
        user_id: User identifier.
        combined_text: Combined text from all batched messages.
        update: Telegram update from first message in batch.
        context: Telegram context.
    """
    # Resolve user_settings_id for multi-tenancy
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

    sent_message = await update.message.reply_text("Thinking... 🧠")

    graph = get_graph(context)
    thread_id = f"{chat_id}_{user_id}"
    config = RunnableConfig(
        {"configurable": {"thread_id": thread_id, "user_settings_id": user_settings_id}}
    )

    accumulated_response = ""
    final_response = ""
    handled_interrupt = False
    last_update_time = time.monotonic()
    update_interval = 1.0
    telegram_char_limit = 4000

    try:
        async for event in graph.astream_events(
            {"messages": [{"role": "user", "content": combined_text}]},
            config=config,
            version="v2",
        ):
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    accumulated_response += content if isinstance(content, str) else str(content)

                    current_time = time.monotonic()
                    if (
                        current_time - last_update_time > update_interval
                        and accumulated_response.strip()
                    ):
                        display_text = truncate_for_telegram(
                            accumulated_response + " ▌",
                            max_len=telegram_char_limit,
                        )
                        try:
                            await context.bot.edit_message_text(
                                chat_id=int(chat_id),
                                message_id=sent_message.message_id,
                                text=display_text,
                            )
                            last_update_time = current_time
                        except BadRequest:
                            pass
                        except TelegramError:
                            pass

        # Check for interrupt (save confirmation)
        state = await graph.aget_state(config)
        if state.next:
            # Graph is interrupted — extract the interrupt value
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
                await context.bot.edit_message_text(
                    chat_id=int(chat_id),
                    message_id=sent_message.message_id,
                    text=confirm_text,
                    reply_markup=keyboard,
                    parse_mode=ParseMode.HTML,
                )
            except TelegramError:
                await update.message.reply_text(
                    confirm_text,
                    reply_markup=keyboard,
                    parse_mode=ParseMode.HTML,
                )
            handled_interrupt = True
            return

        # No interrupt — send final response
        result = await graph.aget_state(config)
        messages = result.values.get("messages", [])
        if messages:
            final_response = messages[-1].content if hasattr(messages[-1], "content") else ""

    except Exception:
        logger.exception("Error processing message in chat %s", chat_id)
        final_response = accumulated_response or "Sorry, something went wrong. Please try again."
    finally:
        if not handled_interrupt and (final_response or accumulated_response):
            final_response = final_response or accumulated_response or "No response generated."
            safe_response = truncate_for_telegram(
                final_response,
                max_len=telegram_char_limit,
            )
            try:
                await context.bot.edit_message_text(
                    chat_id=int(chat_id),
                    message_id=sent_message.message_id,
                    text=safe_response,
                    parse_mode=ParseMode.HTML,
                )
            except TelegramError:
                await update.message.reply_text(
                    safe_response,
                    parse_mode=ParseMode.HTML,
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
    """Clean up resources after the application shuts down.

    Args:
        application: The Telegram Application instance.
    """
    message_batcher = application.bot_data["message_batcher"]
    await message_batcher.shutdown()


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

    # Initialize and store message batcher
    application.bot_data["message_batcher"] = MessageBatcher(
        timeout=1.0, process_callback=process_batch
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("timezone", timezone_command))
    application.add_handler(CommandHandler("reminders", reminders_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(ChatMemberHandler(chat_member_update, ChatMemberHandler.MY_CHAT_MEMBER))

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
