"""Inline keyboard builders for the Telegram bot."""

from __future__ import annotations

from telegram import InlineKeyboardButton, InlineKeyboardMarkup

from src.db.user_settings import ReminderRow


def build_timezone_keyboard(
    chat_id: int,
    *,
    onboarding: bool = False,
) -> InlineKeyboardMarkup:
    """Build an inline keyboard with UTC offset buttons.

    Args:
        chat_id: The Telegram chat ID to encode in callback data.
        onboarding: If True, encodes a flag so the callback handler
            follows up with the reminders picker.

    Returns:
        InlineKeyboardMarkup with UTC offset buttons.
    """
    rows: list[list[InlineKeyboardButton]] = []
    offsets = list(range(-12, 15))  # UTC-12 through UTC+14
    ob_flag = "1" if onboarding else "0"
    row: list[InlineKeyboardButton] = []
    for offset in offsets:
        label = f"UTC{offset:+d}" if offset != 0 else "UTC+0"
        row.append(InlineKeyboardButton(label, callback_data=f"tz|{offset}|{chat_id}|{ob_flag}"))
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)


def build_reminders_message(
    reminders: list[ReminderRow],
    user_settings_id: str,
    *,
    onboarding: bool = False,
) -> tuple[str, InlineKeyboardMarkup]:
    """Build the reminders display text and inline keyboard.

    Args:
        reminders: List of reminder dicts with id and time.
        user_settings_id: The user_settings UUID for callback data.
        onboarding: If True, encodes a flag so the Done callback
            sends a final onboarding message.

    Returns:
        Tuple of (message text, inline keyboard markup).
    """
    ob_flag = "1" if onboarding else "0"

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
                    callback_data=f"rm_rem|{idx}|{user_settings_id}|{ob_flag}",
                )
            ]
        )
    if len(reminders) < 5:
        buttons.append(
            [
                InlineKeyboardButton(
                    "➕ Add reminder",
                    callback_data=f"add_rem|{user_settings_id}|{ob_flag}",
                )
            ]
        )
    buttons.append(
        [
            InlineKeyboardButton(
                "✅ Done",
                callback_data=f"rem_done|{user_settings_id}|{ob_flag}",
            )
        ]
    )

    return text, InlineKeyboardMarkup(buttons)


def build_hour_picker_keyboard(
    user_settings_id: str,
    *,
    onboarding: bool = False,
) -> InlineKeyboardMarkup:
    """Build an inline keyboard with hour buttons (00:00 - 23:00).

    Args:
        user_settings_id: The user_settings UUID for callback data.
        onboarding: If True, encodes a flag to carry through the flow.

    Returns:
        InlineKeyboardMarkup with hour buttons.
    """
    ob_flag = "1" if onboarding else "0"
    rows: list[list[InlineKeyboardButton]] = []
    row: list[InlineKeyboardButton] = []
    for hour in range(24):
        label = f"{hour:02d}:00"
        row.append(
            InlineKeyboardButton(label, callback_data=f"add_hr|{hour}|{user_settings_id}|{ob_flag}")
        )
        if len(row) == 4:
            rows.append(row)
            row = []
    if row:
        rows.append(row)
    return InlineKeyboardMarkup(rows)
