"""Telegram bot interface for ThinkBack.

This module handles the Telegram bot commands and message routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.runnables import RunnableConfig

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

import logging
import time

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
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

from src.core.config import get_settings

logger = logging.getLogger(__name__)

load_dotenv()


def _get_graph(context: ContextTypes.DEFAULT_TYPE) -> CompiledStateGraph:
    """Lazily build and cache the agent graph in bot_data.

    Args:
        context: The Telegram context whose bot_data stores the graph.

    Returns:
        Compiled LangGraph instance.
    """
    if "graph" not in context.bot_data:
        from src.agent.graph import build_graph

        context.bot_data["saver"] = InMemorySaver()
        context.bot_data["graph"] = build_graph(
            checkpointer=context.bot_data["saver"],
        )
    return context.bot_data["graph"]


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle the /start command.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    await update.message.reply_text(  # type: ignore[union-attr]
        "Welcome to ThinkBack! 🧠\n\n"
        "Just type naturally:\n"
        "• Share insights and I'll offer to save them\n"
        "• Ask questions about your saved knowledge\n"
        "• Or just chat!"
    )


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
        return

    user_input = update.message.text
    chat = update.message.chat
    user_id = update.message.from_user.id

    sent_message = await update.message.reply_text("Thinking... 🧠")

    graph = _get_graph(context)
    thread_id = f"{chat.id}_{user_id}"
    config = RunnableConfig({"configurable": {"thread_id": thread_id}})

    accumulated_response = ""
    final_response = ""
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
                        display_text = accumulated_response[:telegram_char_limit] + " ▌"
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

            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton("✅ Save", callback_data=f"save_yes|{thread_id}"),
                        InlineKeyboardButton("❌ Cancel", callback_data=f"save_no|{thread_id}"),
                    ]
                ]
            )

            confirm_text = (
                f"💡 Save this insight?\n\n"
                f"<blockquote>{insight}</blockquote>\n\n"
                f'<i>Original: "{content}"</i>'
            )
            try:
                await context.bot.edit_message_text(
                    chat_id=chat.id,
                    message_id=sent_message.message_id,
                    text=confirm_text[:telegram_char_limit],
                    reply_markup=keyboard,
                    parse_mode=ParseMode.HTML,
                )
            except TelegramError:
                await update.message.reply_text(
                    confirm_text[:telegram_char_limit],
                    reply_markup=keyboard,
                    parse_mode=ParseMode.HTML,
                )
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
        if final_response or accumulated_response:
            final_response = final_response or accumulated_response or "No response generated."
            try:
                await context.bot.edit_message_text(
                    chat_id=chat.id,
                    message_id=sent_message.message_id,
                    text=final_response[:telegram_char_limit],
                    parse_mode=ParseMode.HTML,
                )
            except TelegramError:
                await update.message.reply_text(
                    final_response[:telegram_char_limit],
                    parse_mode=ParseMode.HTML,
                )


async def handle_callback(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle inline button callbacks for save confirmation.

    Args:
        update: The Telegram update containing the callback query.
        context: The Telegram context.
    """
    query = update.callback_query
    if not query or not query.data:
        return

    await query.answer()

    parts = query.data.split("|", 1)
    action = parts[0]
    thread_id = parts[1] if len(parts) > 1 else ""

    approved = action == "save_yes"

    graph = _get_graph(context)
    config = RunnableConfig({"configurable": {"thread_id": thread_id}})

    from langgraph.types import Command

    try:
        result = await graph.ainvoke(
            Command(resume={"approved": approved}),
            config=config,
        )

        messages = result.get("messages", [])
        response = messages[-1].content if messages and hasattr(messages[-1], "content") else ""
        response = response or ("Memory saved! ✅" if approved else "Save cancelled. ❌")

    except Exception:
        logger.exception("Error processing callback")
        response = "Sorry, something went wrong."

    try:
        await query.edit_message_text(text=response, parse_mode=ParseMode.HTML)
    except TelegramError:
        logger.exception("Failed to edit callback message")


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
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_callback))

    return application


def main() -> None:
    """Run the Telegram bot."""
    app = create_application()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
