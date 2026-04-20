"""Tests for the ThinkBack agent tools."""

import uuid
from typing import Any
from unittest.mock import patch


async def test_search_memories_tool_returns_formatted_results() -> None:
    """Test that search_memories_tool returns formatted memory content."""
    from src.agent.tools import search_memories_tool

    with (
        patch("src.agent.tools.db_search_memories") as mock_search,
        patch("src.agent.tools.get_settings") as mock_settings,
    ):
        mock_settings.return_value.search_top_k = 5
        mock_search.return_value = [
            {"content": "Consistency beats intensity", "similarity": 0.9},
            {"content": "Identity drives habits", "similarity": 0.8},
        ]

        tool_input: dict[str, Any] = {"query": "habits"}
        result = await search_memories_tool.ainvoke(
            tool_input,
            config={"configurable": {"user_settings_id": "usr-123"}},
        )

        assert "Consistency beats intensity" in result
        assert "Identity drives habits" in result
        mock_search.assert_called_once_with("habits", user_settings_id="usr-123", top_k=5)


async def test_search_memories_tool_uses_configured_top_k() -> None:
    """Test that search_memories_tool reads top_k from Settings."""
    from src.agent.tools import search_memories_tool

    with (
        patch("src.agent.tools.db_search_memories") as mock_search,
        patch("src.agent.tools.get_settings") as mock_settings,
    ):
        mock_settings.return_value.search_top_k = 10
        mock_search.return_value = [
            {"content": "Test memory", "similarity": 0.9},
        ]

        tool_input: dict[str, Any] = {"query": "test"}
        await search_memories_tool.ainvoke(
            tool_input,
            config={"configurable": {"user_settings_id": "usr-123"}},
        )

        mock_search.assert_called_once_with("test", user_settings_id="usr-123", top_k=10)


async def test_search_memories_tool_handles_no_results() -> None:
    """Test that search_memories_tool handles empty results."""
    from src.agent.tools import search_memories_tool

    with (
        patch("src.agent.tools.db_search_memories") as mock_search,
        patch("src.agent.tools.get_settings") as mock_settings,
    ):
        mock_settings.return_value.search_top_k = 5
        mock_search.return_value = []

        tool_input: dict[str, Any] = {"query": "unknown topic"}
        result = await search_memories_tool.ainvoke(
            tool_input,
            config={"configurable": {"user_settings_id": "usr-123"}},
        )

        assert "no saved" in result.lower() or "not found" in result.lower()


async def test_save_memory_tool_calls_interrupt() -> None:
    """Test that save_memory_tool calls interrupt with content, insight, and duplicates."""
    from src.agent.tools import save_memory_tool

    with (
        patch("src.agent.tools.db_find_duplicates") as mock_find,
        patch("src.agent.tools.interrupt") as mock_interrupt,
        patch("src.agent.tools.db_save_memory") as mock_save,
    ):
        mock_find.return_value = []
        mock_interrupt.return_value = {"approved": True}
        mock_save.return_value = {
            "id": uuid.UUID("00000000-0000-0000-0000-000000000001"),
            "content": "I realized that motivation follows action",
        }

        tool_input: dict[str, Any] = {
            "content": "I realized that motivation follows action",
            "insight": "Motivation follows action",
        }
        result = await save_memory_tool.ainvoke(
            tool_input,
            config={"configurable": {"user_settings_id": "usr-123"}},
        )

        mock_find.assert_called_once_with(
            "I realized that motivation follows action",
            user_settings_id="usr-123",
        )
        mock_interrupt.assert_called_once_with(
            {
                "content": "I realized that motivation follows action",
                "insight": "Motivation follows action",
                "duplicates": [],
            }
        )
        mock_save.assert_called_once_with(
            "I realized that motivation follows action",
            summary="Motivation follows action",
            user_settings_id="usr-123",
        )
        assert "saved" in result.lower()


async def test_save_memory_tool_cancelled() -> None:
    """Test that save_memory_tool handles user rejection."""
    from src.agent.tools import save_memory_tool

    with (
        patch("src.agent.tools.db_find_duplicates") as mock_find,
        patch("src.agent.tools.interrupt") as mock_interrupt,
        patch("src.agent.tools.db_save_memory") as mock_save,
    ):
        mock_find.return_value = []
        mock_interrupt.return_value = {"approved": False}

        tool_input: dict[str, Any] = {
            "content": "Some thought",
            "insight": "A thought",
        }
        result = await save_memory_tool.ainvoke(
            tool_input,
            config={"configurable": {"user_settings_id": "usr-123"}},
        )

        mock_find.assert_called_once_with("Some thought", user_settings_id="usr-123")
        mock_save.assert_not_called()
        assert "cancel" in result.lower()


async def test_save_memory_tool_surfaces_duplicates_in_interrupt() -> None:
    """Test that save_memory_tool includes duplicates in the interrupt payload."""
    from src.agent.tools import save_memory_tool

    duplicates = [
        {"content": "Exercise is good", "similarity": 1.0, "match_type": "exact"},
    ]

    with (
        patch("src.agent.tools.db_find_duplicates") as mock_find,
        patch("src.agent.tools.interrupt") as mock_interrupt,
        patch("src.agent.tools.db_save_memory") as mock_save,
    ):
        mock_find.return_value = duplicates
        mock_interrupt.return_value = {"approved": True}
        mock_save.return_value = {
            "id": uuid.UUID("00000000-0000-0000-0000-000000000001"),
            "content": "Exercise is good",
        }

        tool_input: dict[str, Any] = {
            "content": "Exercise is good",
            "insight": "Exercise matters",
        }
        result = await save_memory_tool.ainvoke(
            tool_input,
            config={"configurable": {"user_settings_id": "usr-123"}},
        )

        mock_find.assert_called_once_with("Exercise is good", user_settings_id="usr-123")
        mock_interrupt.assert_called_once_with(
            {
                "content": "Exercise is good",
                "insight": "Exercise matters",
                "duplicates": duplicates,
            }
        )
        assert "saved" in result.lower()


async def test_save_memory_tool_sends_empty_duplicates_when_none_found() -> None:
    """Test that save_memory_tool sends empty duplicates list when no matches."""
    from src.agent.tools import save_memory_tool

    with (
        patch("src.agent.tools.db_find_duplicates") as mock_find,
        patch("src.agent.tools.interrupt") as mock_interrupt,
        patch("src.agent.tools.db_save_memory") as mock_save,
    ):
        mock_find.return_value = []
        mock_interrupt.return_value = {"approved": True}
        mock_save.return_value = {
            "id": uuid.UUID("00000000-0000-0000-0000-000000000002"),
            "content": "Brand new insight",
        }

        tool_input: dict[str, Any] = {
            "content": "Brand new insight",
            "insight": "Something new",
        }
        result = await save_memory_tool.ainvoke(
            tool_input,
            config={"configurable": {"user_settings_id": "usr-123"}},
        )

        mock_find.assert_called_once_with("Brand new insight", user_settings_id="usr-123")
        mock_interrupt.assert_called_once_with(
            {
                "content": "Brand new insight",
                "insight": "Something new",
                "duplicates": [],
            }
        )
        assert "saved" in result.lower()
