"""Tests for seed_memories script."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.db.seed_memories import seed_memories


@pytest.mark.asyncio
async def test_imports_all_entries_from_json(tmp_path: Path) -> None:
    """Test that seed_memories imports all entries from JSON file."""
    # Create test JSON file
    test_data = [
        {"content": "First memory", "summary": "First"},
        {"content": "Second memory", "summary": "Second"},
        {"content": "Third memory", "summary": "Third"},
    ]
    test_file = tmp_path / "test_seed.json"
    test_file.write_text(json.dumps(test_data))

    # Mock save_memory to avoid hitting real database
    with patch("src.db.seed_memories.save_memory", new_callable=AsyncMock) as mock_save:
        mock_save.return_value = {"id": "test-id", "content": "test"}

        result = await seed_memories(str(test_file))

        # Verify all entries were processed
        assert mock_save.call_count == 3
        assert result["success"] == 3
        assert result["failed"] == 0

        # Verify correct arguments passed to save_memory
        calls = mock_save.call_args_list
        assert calls[0][0] == ("First memory", "First")
        assert calls[1][0] == ("Second memory", "Second")
        assert calls[2][0] == ("Third memory", "Third")


@pytest.mark.asyncio
async def test_adds_2_second_delay_between_calls(tmp_path: Path) -> None:
    """Test that seed_memories adds 2 second delay between save_memory calls."""
    # Create test JSON file with 3 entries
    test_data = [
        {"content": "First", "summary": "First"},
        {"content": "Second", "summary": "Second"},
        {"content": "Third", "summary": "Third"},
    ]
    test_file = tmp_path / "test_seed.json"
    test_file.write_text(json.dumps(test_data))

    # Mock both save_memory and asyncio.sleep
    with (
        patch("src.db.seed_memories.save_memory", new_callable=AsyncMock) as mock_save,
        patch("src.db.seed_memories.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        mock_save.return_value = {"id": "test-id", "content": "test"}

        await seed_memories(str(test_file))

        # Should sleep 2 seconds after each save except the last one
        # 3 entries = 2 sleeps (after 1st and 2nd, but not after 3rd)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(2)


@pytest.mark.asyncio
async def test_retries_on_error_with_2_minute_wait(tmp_path: Path) -> None:
    """Test that seed_memories retries once after waiting 2 minutes on error."""
    # Create test JSON file
    test_data = [
        {"content": "First", "summary": "First"},
        {"content": "Second", "summary": "Second"},
    ]
    test_file = tmp_path / "test_seed.json"
    test_file.write_text(json.dumps(test_data))

    # Mock save_memory to fail on first call, succeed on retry
    with (
        patch("src.db.seed_memories.save_memory", new_callable=AsyncMock) as mock_save,
        patch("src.db.seed_memories.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
    ):
        # First entry fails once then succeeds, second entry succeeds immediately
        mock_save.side_effect = [
            Exception("Rate limit error"),
            {"id": "test-id-1", "content": "First"},
            {"id": "test-id-2", "content": "Second"},
        ]

        result = await seed_memories(str(test_file))

        # Should have 2 successful imports
        assert result["success"] == 2
        assert result["failed"] == 0

        # Should have called save_memory 3 times (1 fail + 1 retry + 1 success)
        assert mock_save.call_count == 3

        # Should have slept for 120 seconds (2 minutes) after the error
        sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
        assert 120 in sleep_calls


@pytest.mark.asyncio
async def test_tracks_failures_when_retry_also_fails(tmp_path: Path) -> None:
    """Test that seed_memories tracks failures when both attempts fail."""
    # Create test JSON file
    test_data = [
        {"content": "First", "summary": "First"},
        {"content": "Second", "summary": "Second"},
        {"content": "Third", "summary": "Third"},
    ]
    test_file = tmp_path / "test_seed.json"
    test_file.write_text(json.dumps(test_data))

    # Mock save_memory to fail twice for first entry, succeed for others
    with (
        patch("src.db.seed_memories.save_memory", new_callable=AsyncMock) as mock_save,
        patch("src.db.seed_memories.asyncio.sleep", new_callable=AsyncMock),
    ):
        # First entry fails twice, second and third succeed
        mock_save.side_effect = [
            Exception("Rate limit error"),
            Exception("Rate limit error again"),
            {"id": "test-id-2", "content": "Second"},
            {"id": "test-id-3", "content": "Third"},
        ]

        result = await seed_memories(str(test_file))

        # Should have 2 successful imports and 1 failure
        assert result["success"] == 2
        assert result["failed"] == 1

        # Should have called save_memory 4 times (2 fails + 2 successes)
        assert mock_save.call_count == 4
