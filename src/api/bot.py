"""Telegram bot interface for ThinkBack.

This module handles the Telegram bot commands and message routing.
"""

from __future__ import annotations

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from src.agent.graph import build_graph
from src.core.config import Settings

# Build the graph once at module load
graph = build_graph()


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

    # Invoke the agent graph
    result = await graph.ainvoke(
        {
            "user_input": user_input,
            "intent": None,
            "memories": [],
            "response": "",
            "error": None,
        }
    )

    # Send the response
    response = result.get("response", "Sorry, I encountered an error.")
    await update.message.reply_text(response)  # type: ignore[union-attr]


def create_application() -> Application:
    """Create and configure the Telegram bot application.

    Returns:
        Configured Telegram Application.

    Example:
        >>> app = create_application()
        >>> app.run_polling()
    """
    settings = Settings()

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
