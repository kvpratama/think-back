"""Tests for seed_memories script."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

from src.db.seed_memories import seed_memories


async def test_seed_memories_passes_user_settings_id() -> None:
    """Test that seed_memories passes user_settings_id to save_memory."""
    from src.db.seed_memories import seed_memories

    with (
        patch("src.db.seed_memories.save_memory", new_callable=AsyncMock) as mock_save,
        patch("src.db.seed_memories.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_save.return_value = {"id": "test-id", "content": "test"}

        result = await seed_memories(
            file_path="src/db/seed.example.json",
            user_settings_id="usr-123",
        )

        assert result["success"] > 0
        for call in mock_save.call_args_list:
            assert call.kwargs.get("user_settings_id") == "usr-123"


async def test_imports_all_entries_from_json(tmp_path: Path) -> None:
    """Test that seed_memories imports all entries from JSON file."""
    test_data = [
        {"content": "First memory", "summary": "First"},
        {"content": "Second memory", "summary": "Second"},
        {"content": "Third memory", "summary": "Third"},
    ]
    test_file = tmp_path / "test_seed.json"
    test_file.write_text(json.dumps(test_data))

    with (
        patch("src.db.seed_memories.save_memory", new_callable=AsyncMock) as mock_save,
        patch("src.db.seed_memories.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_save.return_value = {"id": "test-id", "content": "test"}

        result = await seed_memories(str(test_file), user_settings_id="usr-123")

        assert mock_save.call_count == 3
        assert result["success"] == 3
        assert result["failed"] == 0

        calls = mock_save.call_args_list
        assert calls[0][0] == ("First memory", "First")
        assert calls[1][0] == ("Second memory", "Second")
        assert calls[2][0] == ("Third memory", "Third")


async def test_adds_2_second_delay_between_calls(tmp_path: Path) -> None:
    """Test that seed_memories adds 2 second delay between save_memory calls."""
    test_data = [
        {"content": "First", "summary": "First"},
        {"content": "Second", "summary": "Second"},
        {"content": "Third", "summary": "Third"},
    ]
    test_file = tmp_path / "test_seed.json"
    test_file.write_text(json.dumps(test_data))

    with (
        patch("src.db.seed_memories.save_memory", new_callable=AsyncMock) as mock_save,
        patch("src.db.seed_memories.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_save.return_value = {"id": "test-id", "content": "test"}

        await seed_memories(str(test_file), user_settings_id="usr-123")

        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(2)


async def test_retries_on_error_with_2_minute_wait(tmp_path: Path) -> None:
    """Test that seed_memories retries once after waiting 2 minutes on error."""
    test_data = [
        {"content": "First", "summary": "First"},
        {"content": "Second", "summary": "Second"},
    ]
    test_file = tmp_path / "test_seed.json"
    test_file.write_text(json.dumps(test_data))

    with (
        patch("src.db.seed_memories.save_memory", new_callable=AsyncMock) as mock_save,
        patch("src.db.seed_memories.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_save.side_effect = [
            Exception("Rate limit error"),
            {"id": "test-id-1", "content": "First"},
            {"id": "test-id-2", "content": "Second"},
        ]

        result = await seed_memories(str(test_file), user_settings_id="usr-123")

        assert result["success"] == 2
        assert result["failed"] == 0

        assert mock_save.call_count == 3

        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert 120 in sleep_calls


async def test_tracks_failures_when_retry_also_fails(tmp_path: Path) -> None:
    """Test that seed_memories tracks failures when both attempts fail."""
    test_data = [
        {"content": "First", "summary": "First"},
        {"content": "Second", "summary": "Second"},
        {"content": "Third", "summary": "Third"},
    ]
    test_file = tmp_path / "test_seed.json"
    test_file.write_text(json.dumps(test_data))

    with (
        patch("src.db.seed_memories.save_memory", new_callable=AsyncMock) as mock_save,
        patch("src.db.seed_memories.asyncio.sleep", new_callable=AsyncMock),
    ):
        mock_save.side_effect = [
            Exception("Rate limit error"),
            Exception("Rate limit error again"),
            {"id": "test-id-2", "content": "Second"},
            {"id": "test-id-3", "content": "Third"},
        ]

        result = await seed_memories(str(test_file), user_settings_id="usr-123")

        assert result["success"] == 2
        assert result["failed"] == 1

        assert mock_save.call_count == 4
