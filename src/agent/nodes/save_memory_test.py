"""Tests for the save_memory node."""

from unittest.mock import patch

from src.agent.state import AgentState


async def test_save_memory_node_saves_content() -> None:
    """Test that save_memory node saves the user input as a memory."""
    from src.agent.nodes.save_memory import save_memory

    state: AgentState = {
        "user_input": "/save Consistency beats intensity when building habits",
        "cleaned_input": "Consistency beats intensity when building habits",
        "intent": "save",
        "memories": [],
        "response": "",
        "error": None,
    }

    with patch("src.agent.nodes.save_memory.db_save_memory") as mock_save:
        mock_result = {"id": "test-id", "content": state["cleaned_input"]}
        mock_save.return_value = mock_result

        result = await save_memory(state)

        assert result["response"] == "Memory saved."
        assert len(result["memories"]) == 1
        assert result["memories"][0]["content"] == state["cleaned_input"]
        mock_save.assert_called_once_with(state["cleaned_input"])


async def test_save_memory_node_handles_error() -> None:
    """Test that save_memory node handles errors gracefully."""
    from src.agent.nodes.save_memory import save_memory

    state: AgentState = {
        "user_input": "/save Test memory",
        "cleaned_input": "Test memory",
        "intent": "save",
        "memories": [],
        "response": "",
        "error": None,
    }

    with patch("src.agent.nodes.save_memory.db_save_memory") as mock_save:
        mock_save.side_effect = Exception("Database error")

        result = await save_memory(state)

        assert result["error"] is not None
        assert "Failed to save memory" in result["error"]
