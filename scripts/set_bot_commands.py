"""Register slash-commands with Telegram (one-shot, run manually after deploy).

Usage:
    uv run python scripts/set_bot_commands.py
"""

from __future__ import annotations

import asyncio

from telegram import Bot, BotCommand

from src.core.config import get_settings


async def _main() -> None:
    """Push the current bot command list to Telegram."""
    settings = get_settings()
    bot = Bot(token=settings.telegram_bot_token.get_secret_value())
    async with bot:
        await bot.set_my_commands(
            [
                BotCommand("start", "Get started or reset your intro"),
                BotCommand("timezone", "Set your local timezone"),
                BotCommand("reminders", "Customize when insights resurface"),
                BotCommand("help", "Show available commands"),
            ]
        )
    print("Bot commands updated.")


if __name__ == "__main__":
    asyncio.run(_main())
