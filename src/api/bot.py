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

from src.api.bot_helpers import truncate_for_telegram
from src.core.config import get_settings
from src.db.user_settings import (
    add_reminder,
    get_reminders,
    get_user_settings_id,
    insert_default_reminders,
    remove_reminder,
    update_timezone,
    upsert_user_settings,
)

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


def _build_timezone_keyboard(chat_id: int) -> InlineKeyboardMarkup:
    """Build an inline keyboard with UTC offset buttons.

    Args:
        chat_id: The Telegram chat ID to encode in callback data.

    Returns:
        InlineKeyboardMarkup with UTC offset buttons.
    """
    rows: list[list[InlineKeyboardButton]] = []
    offsets = list(range(-12, 15))  # UTC-12 through UTC+14
    row: list[InlineKeyboardButton] = []
    for offset in offsets:
        label = f"UTC{offset:+d}" if offset != 0 else "UTC+0"
        row.append(InlineKeyboardButton(label, callback_data=f"tz|{offset}|{chat_id}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


def _build_reminders_message(
    reminders: list[dict[str, str]],
    user_settings_id: str,
) -> tuple[str, InlineKeyboardMarkup]:
    """Build the reminders display text and inline keyboard.

    Args:
        reminders: List of reminder dicts with id and time.
        user_settings_id: The user_settings UUID for callback data.

    Returns:
        Tuple of (message text, inline keyboard markup).
    """
    if not reminders:
        text = "⏰ You have no reminders set."
    else:
        lines = ["⏰ Your reminders:"]
        for r in reminders:
            time_display = r["time"][:5]
            lines.append(f"  • {time_display}")
        text = "\n".join(lines)

    buttons: list[list[InlineKeyboardButton]] = []
    for idx, r in enumerate(reminders):
        time_display = r["time"][:5]
        buttons.append(
            [
                InlineKeyboardButton(
                    f"❌ Remove {time_display}",
                    callback_data=f"rm_rem|{idx}",
                )
            ]
        )
    if len(reminders) < 5:
        buttons.append(
            [
                InlineKeyboardButton(
                    "➕ Add reminder",
                    callback_data="add_rem",
                )
            ]
        )

    return text, InlineKeyboardMarkup(buttons)


def _build_hour_picker_keyboard(user_settings_id: str) -> InlineKeyboardMarkup:
    """Build an inline keyboard with hour buttons (00:00 - 23:00).

    Args:
        user_settings_id: The user_settings UUID for callback data.

    Returns:
        InlineKeyboardMarkup with hour buttons.
    """
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for hour in range(24):
        label = f"{hour:02d}:00"
        row.append(InlineKeyboardButton(label, callback_data=f"add_hr|{hour}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


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
        "Welcome to ThinkBack! 🧠\n\n"
        "Just type naturally:\n"
        "• Share insights and I'll offer to save them\n"
        "• Ask questions about your saved knowledge\n"
        "• Or just chat!"
    )

    if is_new:
        user_settings_id = get_user_settings_id(chat_id)
        if user_settings_id:
            insert_default_reminders(user_settings_id)

        keyboard = _build_timezone_keyboard(update.message.chat.id)  # type: ignore[union-attr]
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
    keyboard = _build_timezone_keyboard(chat_id)
    await update.message.reply_text(  # type: ignore[union-attr]
        "🌍 Select your UTC offset:",
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
    text, keyboard = _build_reminders_message(reminders, user_settings_id)
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
        "📚 <b>Available commands:</b>\n\n"
        "/start — Set up your account\n"
        "/timezone — Change your UTC offset\n"
        "/reminders — Manage reminder times\n"
        "/help — Show this message\n\n"
        "Or just type naturally to chat, save insights, "
        "or search your saved knowledge!",
        parse_mode=ParseMode.HTML,
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
        # Timezone selection: callback_data = "tz|<offset>|<chat_id>"
        offset = int(parts[1])
        chat_id = parts[2]
        # POSIX convention: Etc/GMT-7 means UTC+7 (inverted sign)
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
        # Remove reminder: callback_data = "rm_rem|<index>"
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
        text, keyboard = _build_reminders_message(reminders, user_settings_id)
        try:
            await query.edit_message_text(text=text, reply_markup=keyboard)
        except TelegramError:
            logger.exception("Failed to edit reminders message")
        return

    if action == "add_rem":
        # Show hour picker: callback_data = "add_rem"
        if not query.message:
            return
        chat_id = str(query.message.chat.id)
        user_settings_id = get_user_settings_id(chat_id)
        if not user_settings_id:
            return
        keyboard = _build_hour_picker_keyboard(user_settings_id)
        try:
            await query.edit_message_text(
                text="🕐 Select a time for the new reminder:",
                reply_markup=keyboard,
            )
        except TelegramError:
            logger.exception("Failed to edit hour picker message")
        return

    if action == "add_hr":
        # Add reminder at selected hour: callback_data = "add_hr|<hour>"
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
        text, keyboard = _build_reminders_message(reminders, user_settings_id)
        try:
            await query.edit_message_text(text=text, reply_markup=keyboard)
        except TelegramError:
            logger.exception("Failed to edit reminders message")
        return

    # Save confirmation: callback_data = "save_yes|<thread_id>" or "save_no|<thread_id>"
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
        await query.edit_message_text(
            text=truncate_for_telegram(response),
            parse_mode=ParseMode.HTML,
        )
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
    application.add_handler(CommandHandler("timezone", timezone_command))
    application.add_handler(CommandHandler("reminders", reminders_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(handle_callback))

    return application


def main() -> None:
    """Run the Telegram bot."""
    app = create_application()
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
