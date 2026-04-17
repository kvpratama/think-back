"""Telegram bot callback query handler.

This module dispatches inline-button callbacks: timezone selection,
reminder management, and save confirmation.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from src.api.bot_graph import get_graph
from src.api.bot_helpers import truncate_for_telegram
from src.api.bot_keyboards import (
    build_hour_picker_keyboard,
    build_reminders_message,
)
from src.db.user_settings import (
    add_reminder,
    get_reminders,
    remove_reminder,
    update_timezone,
)

logger = logging.getLogger(__name__)


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

    parts = query.data.split("|")
    action = parts[0]

    if action == "tz":
        offset = int(parts[1])
        chat_id = parts[2]
        if offset == 0:
            tz_str = "UTC"
        else:
            tz_str = f"Etc/GMT{-offset:+d}"
        await asyncio.to_thread(update_timezone, chat_id, tz_str)
        label = f"UTC{offset:+d}" if offset != 0 else "UTC+0"
        try:
            await query.edit_message_text(text=f"Timezone set to {label} ✅")
        except TelegramError:
            logger.exception("Failed to edit timezone message")
        return

    if action == "rm_rem":
        if not query.message:
            return
        idx = int(parts[1])
        user_settings_id = parts[2]
        reminders = await asyncio.to_thread(get_reminders, user_settings_id)
        if idx < len(reminders):
            await asyncio.to_thread(remove_reminder, reminders[idx]["id"])
        reminders = await asyncio.to_thread(get_reminders, user_settings_id)
        text, keyboard = build_reminders_message(reminders, user_settings_id)
        try:
            await query.edit_message_text(text=text, reply_markup=keyboard)
        except TelegramError:
            logger.exception("Failed to edit reminders message")
        return

    if action == "add_rem":
        if not query.message:
            return
        user_settings_id = parts[1]
        keyboard = build_hour_picker_keyboard(user_settings_id)
        try:
            await query.edit_message_text(
                text="🕐 Select a time for the new reminder:",
                reply_markup=keyboard,
            )
        except TelegramError:
            logger.exception("Failed to edit hour picker message")
        return

    if action == "add_hr":
        if not query.message:
            return
        hour = int(parts[1])
        user_settings_id = parts[2]
        time_str = f"{hour:02d}:00"
        result = await asyncio.to_thread(add_reminder, user_settings_id, time_str)

        # Handle discriminated result
        from src.db.user_settings import AddReminderResult

        if result == AddReminderResult.LIMIT_REACHED:
            try:
                await query.edit_message_text(text="⚠️ Maximum 5 reminders reached.")
            except TelegramError:
                logger.exception("Failed to edit max reminders message")
            return

        if result == AddReminderResult.DB_ERROR:
            try:
                await query.edit_message_text(text="❌ Couldn't add reminder.")
            except TelegramError:
                logger.exception("Failed to edit DB error message")
            return

        reminders = await asyncio.to_thread(get_reminders, user_settings_id)
        text, keyboard = build_reminders_message(reminders, user_settings_id)
        try:
            await query.edit_message_text(text=text, reply_markup=keyboard)
        except TelegramError:
            logger.exception("Failed to edit reminders message")
        return

    # Save confirmation: callback_data = "save_yes|<thread_id>" or "save_no|<thread_id>"
    if action not in ("save_yes", "save_no"):
        return

    thread_id = parts[1] if len(parts) > 1 else ""
    approved = action == "save_yes"

    graph = get_graph(context)
    config = RunnableConfig({"configurable": {"thread_id": thread_id}})

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
        await query.edit_message_text(
            text=truncate_for_telegram(response),
            parse_mode=ParseMode.HTML,
        )
    except TelegramError:
        logger.exception("Failed to edit callback message")
