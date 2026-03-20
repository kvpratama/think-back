"""Telegram bot interface for ThinkBack.

This module handles the Telegram bot commands and message routing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

import time

from langgraph.checkpoint.memory import InMemorySaver
from telegram import Update
from telegram.error import BadRequest, TelegramError
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.core.config import get_settings

_graph: CompiledStateGraph[Any, Any] | None = None
# Initialize the saver outside the getter so it survives graph recompilation
_memory_saver = InMemorySaver()


def _get_graph() -> CompiledStateGraph[Any, Any]:
    """Lazily build and cache the agent graph.

    Returns:
        Compiled LangGraph instance.
    """
    global _graph  # noqa: PLW0603
    if _graph is None:
        from src.agent.graph import build_graph

        _graph = build_graph(checkpointer=_memory_saver)
    return _graph


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
    user_input = update.message.text  # type: ignore[union-attr]
    chat = update.message.chat  # type: ignore[union-attr]

    graph = _get_graph()

    # Initial thinking message
    sent_message = await update.message.reply_text("Thinking... 🧠")  # type: ignore[union-attr]

    accumulated_response = ""
    final_response = ""
    last_update_time = time.monotonic()
    update_interval = 1.0  # seconds – respects Telegram's rate limits
    telegram_char_limit = 4000  # leave headroom below the 4096 hard limit

    # Use astream_events to capture streaming tokens
    async for event in graph.astream_events(
        {
            "user_input": user_input,
            "cleaned_input": "",
            "intent": None,
            "memories": [],
            "response": "",
            "error": None,
        },
        config={
            "configurable": {
                "thread_id": str(chat.id),
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

    # Send the final response after the stream ends (handles missing on_chain_end too)
    final_response = final_response or accumulated_response or "No response generated."
    try:
        await context.bot.edit_message_text(
            chat_id=chat.id,
            message_id=sent_message.message_id,
            text=final_response[:telegram_char_limit],
        )
    except TelegramError:
        await update.message.reply_text(  # type: ignore[union-attr]
            final_response[:telegram_char_limit],
        )


def create_application() -> Application:  # type: ignore[type-arg]
    """Create and configure the Telegram bot application.

    Returns:
        Configured Telegram Application.

    Example:
        >>> app = create_application()
        >>> app.run_polling()
    """
    settings = get_settings()

    # Create the application
    application = Application.builder().token(settings.telegram_bot_token).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("save", handle_message))
    application.add_handler(CommandHandler("ask", handle_message))
    application.add_handler(CommandHandler("query", handle_message))

    # Handle all text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    return application


def main() -> None:
    """Run the Telegram bot."""
    app = create_application()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
