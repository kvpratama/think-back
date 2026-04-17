"""Telegram bot callback query handler.

This module dispatches inline-button callbacks: timezone selection,
reminder management, and save confirmation.
"""

from __future__ import annotations

import logging

from langchain_core.runnables import RunnableConfig
from telegram import Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import ContextTypes

from src.api.bot_helpers import truncate_for_telegram
from src.api.bot_keyboards import (
    build_hour_picker_keyboard,
    build_reminders_message,
)
from src.db.user_settings import (
    add_reminder,
    get_reminders,
    get_user_settings_id,
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
        update_timezone(chat_id, tz_str)
        label = f"UTC{offset:+d}" if offset != 0 else "UTC+0"
        try:
            await query.edit_message_text(text=f"Timezone set to {label} ✅")
        except TelegramError:
            logger.exception("Failed to edit timezone message")
        return

    if action == "rm_rem":
        if not query.message:
            return
        chat_id = str(query.message.chat.id)
        user_settings_id = get_user_settings_id(chat_id)
        if not user_settings_id:
            return
        reminders = get_reminders(user_settings_id)
        idx = int(parts[1])
        if idx < len(reminders):
            remove_reminder(reminders[idx]["id"])
        reminders = get_reminders(user_settings_id)
        text, keyboard = build_reminders_message(reminders, user_settings_id)
        try:
            await query.edit_message_text(text=text, reply_markup=keyboard)
        except TelegramError:
            logger.exception("Failed to edit reminders message")
        return

    if action == "add_rem":
        if not query.message:
            return
        chat_id = str(query.message.chat.id)
        user_settings_id = get_user_settings_id(chat_id)
        if not user_settings_id:
            return
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
        chat_id = str(query.message.chat.id)
        user_settings_id = get_user_settings_id(chat_id)
        if not user_settings_id:
            return
        hour = int(parts[1])
        time_str = f"{hour:02d}:00"
        added = add_reminder(user_settings_id, time_str)
        if not added:
            try:
                await query.edit_message_text(text="⚠️ Maximum 5 reminders reached.")
            except TelegramError:
                logger.exception("Failed to edit max reminders message")
            return
        reminders = get_reminders(user_settings_id)
        text, keyboard = build_reminders_message(reminders, user_settings_id)
        try:
            await query.edit_message_text(text=text, reply_markup=keyboard)
        except TelegramError:
            logger.exception("Failed to edit reminders message")
        return

    # Save confirmation: callback_data = "save_yes|<thread_id>" or "save_no|<thread_id>"
    thread_id = parts[1] if len(parts) > 1 else ""
    approved = action == "save_yes"

    from src.api.bot import _get_graph

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
        await query.edit_message_text(
            text=truncate_for_telegram(response),
            parse_mode=ParseMode.HTML,
        )
    except TelegramError:
        logger.exception("Failed to edit callback message")
