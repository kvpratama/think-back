"""Telegram bot interface for ThinkBack.

This module handles the Telegram bot commands and message routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

import logging
import time

from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from telegram import Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)

from src.core.config import get_settings

logger = logging.getLogger(__name__)

# Load .env variables before langgraph or langsmith imports;
# the LangSmith SDK initializes tracing on startup and needs
# the credentials present in the process environment.
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
        "Use /save to save knowledge.\n"
        "Use /ask to query your knowledge."
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

    graph = _get_graph(context)

    # Initial thinking message
    sent_message = await update.message.reply_text("Thinking... 🧠")

    accumulated_response = ""
    final_response = ""
    last_update_time = time.monotonic()
    update_interval = 1.0  # seconds – respects Telegram's rate limits
    telegram_char_limit = 4000  # leave headroom below the 4096 hard limit

    # Use astream_events to capture streaming tokens
    try:
        async for event in graph.astream_events(
            {
                "user_input": user_input,
            },
            config={
                "configurable": {
                    "thread_id": f"{chat.id}_{user_id}",
                }
            },
            version="v2",
        ):
            # Listen for tokens from the chat model
            if event["event"] == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                if content:
                    accumulated_response += content if isinstance(content, str) else str(content)

                    # Periodic update to Telegram to avoid rate limits
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
                            pass  # "Message is not modified" or similar
                        except TelegramError:
                            pass  # transient network / rate-limit hiccup

            # Capture the final result from the graph
            elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                final_result = event["data"]["output"]
                final_response = final_result.get("response") or "No response generated."
    except Exception:
        logger.exception("Error processing message in chat %s", chat.id)
        final_response = accumulated_response or "Sorry, something went wrong. Please try again."
    finally:
        # Always replace the placeholder message, even on errors
        final_response = final_response or accumulated_response or "No response generated."
        try:
            await context.bot.edit_message_text(
                chat_id=chat.id,
                message_id=sent_message.message_id,
                text=final_response[:telegram_char_limit],
            )
        except TelegramError:
            await update.message.reply_text(
                final_response[:telegram_char_limit],
            )


def create_application() -> Application:
    """Create and configure the Telegram bot application.

    Returns:
        Configured Telegram Application.

    Example:
        >>> app = create_application()
        >>> app.run_polling()
    """
    settings = get_settings()

    # Create the application
    application = (
        Application.builder().token(settings.telegram_bot_token.get_secret_value()).build()
    )

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("save", handle_message))
    application.add_handler(CommandHandler("ask", handle_message))

    return application


def main() -> None:
    """Run the Telegram bot."""
    app = create_application()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
