"""Database operations for user settings and reminder times."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, NotRequired, TypedDict, cast

from src.db.client import get_supabase_client


class AddReminderResult(Enum):
    """Result of add_reminder operation."""

    SUCCESS = "success"
    LIMIT_REACHED = "limit_reached"
    DB_ERROR = "db_error"


class ReminderRow(TypedDict):
    """A reminder_times row returned from the database."""

    id: str
    user_settings_id: str
    time: str
    created_at: NotRequired[str]


def upsert_user_settings(telegram_chat_id: str) -> bool:
    """Upsert a user_settings row for the given Telegram chat ID.

    Args:
        telegram_chat_id: The Telegram chat ID string.

    Returns:
        True if the user is new (just inserted), False if existing.
    """
    client = get_supabase_client()
    result = (
        client.table("user_settings")
        .upsert(
            {"telegram_chat_id": telegram_chat_id},
            on_conflict="telegram_chat_id",
        )
        .execute()
    )
    row = cast(list[dict[str, Any]], result.data)[0]
    created = datetime.fromisoformat(row["created_at"])
    updated = datetime.fromisoformat(row["updated_at"])
    return created == updated


def get_user_settings_id(telegram_chat_id: str) -> str | None:
    """Get the user_settings UUID for a Telegram chat ID.

    Args:
        telegram_chat_id: The Telegram chat ID string.

    Returns:
        The user_settings UUID string, or None if not found.
    """
    client = get_supabase_client()
    result = (
        client.table("user_settings")
        .select("id")
        .eq("telegram_chat_id", telegram_chat_id)
        .execute()
    )
    if not result.data:
        return None
    return cast(list[dict[str, Any]], result.data)[0]["id"]


def insert_default_reminders(user_settings_id: str) -> None:
    """Insert default reminder times (12:00) for a new user.

    Args:
        user_settings_id: The user_settings UUID.
    """
    client = get_supabase_client()
    client.table("reminder_times").insert(
        [
            {"user_settings_id": user_settings_id, "time": "12:00"},
        ]
    ).execute()


def update_timezone(telegram_chat_id: str, timezone_str: str) -> None:
    """Update the timezone for a user.

    Args:
        telegram_chat_id: The Telegram chat ID string.
        timezone_str: The IANA timezone string (e.g., 'Etc/GMT-7').
    """
    client = get_supabase_client()
    client.table("user_settings").update({"timezone": timezone_str}).eq(
        "telegram_chat_id", telegram_chat_id
    ).execute()


MAX_REMINDERS = 5


def get_reminders(user_settings_id: str) -> list[ReminderRow]:
    """Get all reminder times for a user, ordered by time.

    Args:
        user_settings_id: The user_settings UUID.

    Returns:
        A list of reminder dicts with id, user_settings_id, and time.
    """
    client = get_supabase_client()
    result = (
        client.table("reminder_times")
        .select("id, user_settings_id, time")
        .eq("user_settings_id", user_settings_id)
        .order("time")
        .execute()
    )
    return cast(list[ReminderRow], result.data)


def add_reminder(user_settings_id: str, time_str: str) -> AddReminderResult | bool:
    """Add a reminder time for a user.

    Enforces the maximum of 5 reminders per user.

    Args:
        user_settings_id: The user_settings UUID.
        time_str: The time string in HH:MM format.

    Returns:
        AddReminderResult.SUCCESS (or True for backward compatibility),
        AddReminderResult.LIMIT_REACHED if at max capacity,
        AddReminderResult.DB_ERROR if upsert fails.
    """
    client = get_supabase_client()
    existing = (
        client.table("reminder_times")
        .select("id")
        .eq("user_settings_id", user_settings_id)
        .execute()
    )
    if len(existing.data) >= MAX_REMINDERS:
        return AddReminderResult.LIMIT_REACHED

    result = (
        client.table("reminder_times")
        .upsert(
            {"user_settings_id": user_settings_id, "time": time_str},
            on_conflict="user_settings_id,time",
        )
        .execute()
    )
    if len(result.data) > 0:
        return True
    return AddReminderResult.DB_ERROR


def remove_reminder(reminder_id: str) -> None:
    """Remove a reminder time by its ID.

    Args:
        reminder_id: The reminder_times UUID.
    """
    client = get_supabase_client()
    client.table("reminder_times").delete().eq("id", reminder_id).execute()
