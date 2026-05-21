"""Tests for the serverless runtime singletons."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Ensure each test starts with a fresh singleton."""
    import asyncio

    import api._runtime as runtime

    runtime._application = None
    runtime._application_lock = asyncio.Lock()


async def test_get_application_builds_once() -> None:
    """get_application should create the Application exactly once across concurrent calls."""
    import asyncio

    import api._runtime as runtime

    mock_app = MagicMock()
    mock_app.initialize = AsyncMock()

    with patch("api._runtime.create_application", return_value=mock_app) as mock_create:
        results = await asyncio.gather(
            runtime.get_application(),
            runtime.get_application(),
            runtime.get_application(),
        )

    assert all(r is mock_app for r in results)
    mock_create.assert_called_once()
    mock_app.initialize.assert_awaited_once()


async def test_get_application_returns_cached_on_warm_call() -> None:
    """A second call after init returns the same instance without re-initializing."""
    import api._runtime as runtime

    mock_app = MagicMock()
    mock_app.initialize = AsyncMock()

    with patch("api._runtime.create_application", return_value=mock_app):
        first = await runtime.get_application()
        second = await runtime.get_application()

    assert first is second
    mock_app.initialize.assert_awaited_once()
