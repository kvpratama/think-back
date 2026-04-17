"""Database operations for user settings and reminder times."""

from __future__ import annotations

from datetime import datetime

from src.db.client import get_supabase_client


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
    row = result.data[0]
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
    return result.data[0]["id"]


def insert_default_reminders(user_settings_id: str) -> None:
    """Insert default reminder times (08:00 and 20:00) for a new user.

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
