"""Telegram bot command handlers.

This module defines the /start, /timezone, /reminders, and /help commands.
"""

from __future__ import annotations

from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ContextTypes

from src.api.bot_keyboards import (
    build_reminders_message,
    build_timezone_keyboard,
)
from src.db.user_settings import (
    get_reminders,
    get_user_settings_id,
    insert_default_reminders,
    upsert_user_settings,
)


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle the /start command.

    Upserts user settings and shows a timezone picker for new users.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    chat_id = str(update.message.chat.id)  # type: ignore[union-attr]
    is_new = upsert_user_settings(chat_id)

    await update.message.reply_text(  # type: ignore[union-attr]
        """🧠 <b>Welcome to ThinkBack</b>

Your space to capture ideas, revisit them, and actually remember what matters.

<b>Here's how you can use this space:</b>
• 💡 Share an insight → I'll help you save it
• 🔍 Ask questions → I'll search your saved knowledge
• 💬 Just chat → I'm here for that too

<b>⏰ Spaced Repetition</b>
Saved insights come back to you later (default is 12:00 noon) with reflections to strengthen your memory.
You can customize reminder times anytime with /reminders.

<b>Quick setup:</b>
• 🌍 Set your timezone: /timezone
• ❓ See all commands: /help

Start by sharing your first thought 👇
""",  # noqa: E501
        parse_mode=ParseMode.HTML,
    )

    if is_new:
        user_settings_id = get_user_settings_id(chat_id)
        if user_settings_id:
            insert_default_reminders(user_settings_id)

        keyboard = build_timezone_keyboard(update.message.chat.id)  # type: ignore[union-attr]
        await update.message.reply_text(  # type: ignore[union-attr]
            "🌍 What's your UTC offset?",
            reply_markup=keyboard,
        )


async def timezone_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle the /timezone command.

    Shows the UTC offset picker so the user can update their timezone.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    chat_id = update.message.chat.id  # type: ignore[union-attr]
    keyboard = build_timezone_keyboard(chat_id)
    await update.message.reply_text(  # type: ignore[union-attr]
        "🌍 Select your time zone:",
        reply_markup=keyboard,
    )


async def reminders_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle the /reminders command.

    Shows current reminder times with options to add or remove.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    chat_id = str(update.message.chat.id)  # type: ignore[union-attr]
    user_settings_id = get_user_settings_id(chat_id)

    if not user_settings_id:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Please run /start first to set up your account."
        )
        return

    reminders = get_reminders(user_settings_id)
    text, keyboard = build_reminders_message(reminders, user_settings_id)
    await update.message.reply_text(  # type: ignore[union-attr]
        text,
        reply_markup=keyboard,
    )


async def help_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle the /help command.

    Lists all available bot commands.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    await update.message.reply_text(  # type: ignore[union-attr]
        """📚 <b>Help & Commands</b>

<b>Core commands</b>
/start — Get started or reset your intro
/timezone — Set your local timezone
/reminders — Customize when insights resurface
/help — Show this guide

<b>How to use ThinkBack</b>
• 💡 Share an insight → I'll help you save it
• 🔍 Ask a question → I'll answer using your saved insights

What do you want to try first?
""",
        parse_mode=ParseMode.HTML,
    )
