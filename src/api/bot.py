"""Telegram bot interface for ThinkBack.

This module handles the Telegram bot commands and message routing.
"""

from __future__ import annotations

import logging
import time

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.api.bot_callbacks import handle_callback
from src.api.bot_commands import (
    help_command,
    reminders_command,
    start_command,
    timezone_command,
)
from src.api.bot_graph import get_graph
from src.api.bot_helpers import truncate_for_telegram
from src.core.config import get_settings

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
    """Handle incoming messages and route to the agent graph.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    if not update.message or not update.message.from_user:
        logger.debug("handle_message invoked without message/from_user; skipping")
        return

    user_input = update.message.text
    chat = update.message.chat
    user_id = update.message.from_user.id

    sent_message = await update.message.reply_text("Thinking... 🧠")

    graph = get_graph(context)
    thread_id = f"{chat.id}_{user_id}"
    config = RunnableConfig({"configurable": {"thread_id": thread_id}})

    accumulated_response = ""
    final_response = ""
    handled_interrupt = False
    last_update_time = time.monotonic()
    update_interval = 1.0
    telegram_char_limit = 4000

    try:
        async for event in graph.astream_events(
            {"messages": [{"role": "user", "content": user_input}]},
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
                                chat_id=chat.id,
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
                        InlineKeyboardButton("✅ Save", callback_data=f"save_yes|{thread_id}"),
                        InlineKeyboardButton("❌ Cancel", callback_data=f"save_no|{thread_id}"),
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
                    chat_id=chat.id,
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
        logger.exception("Error processing message in chat %s", chat.id)
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
                    chat_id=chat.id,
                    message_id=sent_message.message_id,
                    text=safe_response,
                    parse_mode=ParseMode.HTML,
                )
            except TelegramError:
                await update.message.reply_text(
                    safe_response,
                    parse_mode=ParseMode.HTML,
                )


def create_application() -> Application:
    """Create and configure the Telegram bot application.

    Returns:
        Configured Telegram Application.
    """
    settings = get_settings()

    application = (
        Application.builder().token(settings.telegram_bot_token.get_secret_value()).build()
    )

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("timezone", timezone_command))
    application.add_handler(CommandHandler("reminders", reminders_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_callback))

    return application


def main() -> None:
    """Run the Telegram bot.

    Uses webhook mode when WEBHOOK_URL is set (production),
    otherwise falls back to polling (local development).
    """
    settings = get_settings()
    app = create_application()

    if settings.webhook_url:
        logger.info("Starting webhook mode at %s/webhook", settings.webhook_url)
        app.run_webhook(
            listen="0.0.0.0",
            port=settings.port,
            url_path="/webhook",
            webhook_url=f"{settings.webhook_url}/webhook",
            secret_token=settings.webhook_secret.get_secret_value(),
        )
    else:
        logger.info("Starting polling mode")
        app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
