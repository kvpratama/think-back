"""Tests for the FastAPI Vercel entrypoint."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def telegram_update_body() -> dict:
    """A minimal valid Telegram update payload."""
    return {
        "update_id": 1,
        "message": {
            "message_id": 1,
            "date": 0,
            "chat": {"id": 123, "type": "private"},
            "from": {"id": 123, "is_bot": False, "first_name": "Test"},
            "text": "hello",
        },
    }


def test_health_endpoint_returns_ok() -> None:
    """GET /api/health returns 200 with a status payload."""
    from api.index import app

    client = TestClient(app)
    resp = client.get("/api/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_webhook_rejects_missing_secret(telegram_update_body: dict) -> None:
    """POST /api/webhook without the secret header returns 401."""
    from api.index import app

    mock_app = MagicMock()
    mock_app.bot = MagicMock()
    mock_app.process_update = AsyncMock()

    with (
        patch("api.index.get_application", AsyncMock(return_value=mock_app)),
        patch("api.index._expected_secret", return_value="s3cret"),
    ):
        client = TestClient(app)
        resp = client.post("/api/webhook", content=json.dumps(telegram_update_body))

    assert resp.status_code == 401
    mock_app.process_update.assert_not_called()


def test_webhook_rejects_wrong_secret(telegram_update_body: dict) -> None:
    """POST /api/webhook with a wrong secret header returns 401."""
    from api.index import app

    mock_app = MagicMock()
    mock_app.bot = MagicMock()
    mock_app.process_update = AsyncMock()

    with (
        patch("api.index.get_application", AsyncMock(return_value=mock_app)),
        patch("api.index._expected_secret", return_value="s3cret"),
    ):
        client = TestClient(app)
        resp = client.post(
            "/api/webhook",
            content=json.dumps(telegram_update_body),
            headers={"X-Telegram-Bot-Api-Secret-Token": "wrong"},
        )

    assert resp.status_code == 401
    mock_app.process_update.assert_not_called()


def test_webhook_dispatches_update_when_secret_matches(telegram_update_body: dict) -> None:
    """Valid request: parse body to Update and call application.process_update."""
    from api.index import app

    mock_app = MagicMock()
    mock_app.bot = MagicMock()
    mock_app.process_update = AsyncMock()

    with (
        patch("api.index.get_application", AsyncMock(return_value=mock_app)),
        patch("api.index._expected_secret", return_value="s3cret"),
    ):
        client = TestClient(app)
        resp = client.post(
            "/api/webhook",
            content=json.dumps(telegram_update_body),
            headers={"X-Telegram-Bot-Api-Secret-Token": "s3cret"},
        )

    assert resp.status_code == 200
    mock_app.process_update.assert_awaited_once()
    # The arg passed to process_update should be a telegram.Update with the right id
    call_arg = mock_app.process_update.call_args.args[0]
    assert call_arg.update_id == 1
