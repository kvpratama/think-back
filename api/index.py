"""Vercel serverless entrypoint for ThinkBack.

FastAPI app at the Vercel-conventional ``api/index.py`` location. Receives
Telegram webhook POSTs, verifies the shared secret, and dispatches updates
to the shared ``python-telegram-bot`` ``Application`` instance.
"""

from __future__ import annotations

from fastapi import FastAPI, Header, HTTPException, Request

from api._runtime import get_application
from src.core.config import get_settings

app = FastAPI(title="ThinkBack Telegram Webhook")


def _expected_secret() -> str:
    """Return the configured webhook secret (factored out so tests can patch it)."""
    return get_settings().webhook_secret.get_secret_value()


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Liveness probe.

    Returns:
        ``{"status": "ok"}`` when the function is reachable.
    """
    return {"status": "ok"}


@app.post("/api/webhook")
async def telegram_webhook(
    request: Request,
    x_telegram_bot_api_secret_token: str | None = Header(default=None),
) -> dict[str, str]:
    """Receive a Telegram update and dispatch it to the bot Application.

    Args:
        request: Raw FastAPI request (used to read the JSON body).
        x_telegram_bot_api_secret_token: Shared secret header that Telegram
            sends with every webhook POST when configured via ``setWebhook``.

    Raises:
        HTTPException: 401 when the secret is missing or wrong.

    Returns:
        ``{"status": "ok"}`` after the update has been processed.
    """
    from telegram import Update

    expected = _expected_secret()
    if not expected or x_telegram_bot_api_secret_token != expected:
        raise HTTPException(status_code=401, detail="invalid secret")

    body = await request.json()
    application = await get_application()
    update = Update.de_json(body, application.bot)
    if update is not None:
        await application.process_update(update)
    return {"status": "ok"}
