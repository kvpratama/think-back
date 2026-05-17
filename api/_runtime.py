"""Serverless runtime singletons for the Vercel deployment.

Provides a lazily-initialized, async-safe Telegram ``Application`` instance
that survives across warm function invocations on Vercel Fluid Compute.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from src.api.bot import create_application

if TYPE_CHECKING:
    from telegram.ext import Application

_application: Application | None = None
_application_lock: asyncio.Lock = asyncio.Lock()


async def get_application() -> Application:
    """Return a shared, initialized Telegram ``Application``.

    On first call, builds the Application via ``create_application()`` and
    calls ``await application.initialize()``. Subsequent calls return the
    cached instance. Safe against concurrent cold-start callers.

    Returns:
        The shared Telegram ``Application`` instance.
    """
    global _application
    if _application is not None:
        return _application

    async with _application_lock:
        if _application is not None:
            return _application
        app = create_application()
        await app.initialize()
        _application = app
        return _application
