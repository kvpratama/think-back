"""Telegram bot command handlers.

This module defines the /start, /timezone, /reminders, and /help commands.
"""

from __future__ import annotations

import asyncio

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

WELCOME_MESSAGE = """🧠 <b>Welcome to ThinkBack</b>

Your space to capture ideas, revisit them, and actually remember what matters.

<b>Here's how you can use this space:</b>
• 💡 Share an insight → I'll remember it for you
• 🔍 Ask questions → I'll search your saved knowledge
• 💬 Just chat → I'm here for that too

<b>⏰ Spaced Repetition</b>
From time to time, I'll bring back things you've saved—so they don't fade away.
You can adjust when that happens anytime with /reminders

<b>Quick setup:</b>
• 🌍 Set your timezone: /timezone
• ⏰ Set your reminder times: /reminders
• ❓ See all commands: /help

"""


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle the /start command.

    Sends the welcome message to all users. For new users, also prompts for
    timezone selection and initializes default reminders.

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    chat_id = str(update.message.chat.id)  # type: ignore[union-attr]
    is_new = await asyncio.to_thread(upsert_user_settings, chat_id)

    if is_new:
        await update.message.reply_text(  # type: ignore[union-attr]
            WELCOME_MESSAGE,
            parse_mode=ParseMode.HTML,
        )

        user_settings_id = await asyncio.to_thread(get_user_settings_id, chat_id)
        if user_settings_id:
            await asyncio.to_thread(insert_default_reminders, user_settings_id)

        keyboard = build_timezone_keyboard(
            update.message.chat.id,  # type: ignore[union-attr]
            onboarding=True,
        )
        await update.message.reply_text(  # type: ignore[union-attr]
            "🌍 What's your time zone?",
            reply_markup=keyboard,
        )
    else:
        await update.message.reply_text(  # type: ignore[union-attr]
            WELCOME_MESSAGE,
            parse_mode=ParseMode.HTML,
        )


async def chat_member_update(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """Handle ChatMemberUpdated events for auto-welcome.

    Sends the welcome message and timezone picker when a user first opens
    the bot (status changes to 'member').

    Args:
        update: The Telegram update.
        context: The Telegram context.
    """
    chat_member = update.my_chat_member
    if not chat_member or chat_member.new_chat_member.status != "member":
        return

    chat_id = str(chat_member.chat.id)
    is_new = await asyncio.to_thread(upsert_user_settings, chat_id)

    if not is_new:
        return

    await context.bot.send_message(
        chat_id=chat_member.chat.id,
        text=WELCOME_MESSAGE,
        parse_mode=ParseMode.HTML,
    )

    user_settings_id = await asyncio.to_thread(get_user_settings_id, chat_id)
    if user_settings_id:
        await asyncio.to_thread(insert_default_reminders, user_settings_id)

    keyboard = build_timezone_keyboard(chat_member.chat.id, onboarding=True)
    await context.bot.send_message(
        chat_id=chat_member.chat.id,
        text="🌍 What's your time zone?",
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
    user_settings_id = await asyncio.to_thread(get_user_settings_id, chat_id)

    if not user_settings_id:
        await update.message.reply_text(  # type: ignore[union-attr]
            "Please run /start first to set up your account."
        )
        return

    reminders = await asyncio.to_thread(get_reminders, user_settings_id)
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
