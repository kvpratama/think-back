"""Tests for the retrieve_memories node."""

from unittest.mock import patch

from src.agent.state import AgentState


async def test_retrieve_memories_node_searches_database() -> None:
    """Test that retrieve_memories node searches for relevant memories."""
    from src.agent.nodes.retrieve_memories import retrieve_memories

    state: AgentState = {
        "user_input": "/ask What do I know about habits?",
        "cleaned_input": "What do I know about habits?",
        "intent": "query",
        "memories": [],
        "response": "",
        "error": None,
    }

    with patch("src.agent.nodes.retrieve_memories.db_search_memories") as mock_search:
        mock_memories = [
            {"content": "Consistency beats intensity", "summary": "habits"},
            {"content": "Identity drives habits", "summary": "habits"},
        ]
        mock_search.return_value = mock_memories

        result = await retrieve_memories(state)

        assert result["memories"] == mock_memories
        mock_search.assert_called_once_with(state["cleaned_input"], top_k=3)


async def test_retrieve_memories_node_handles_no_results() -> None:
    """Test that retrieve_memories node handles empty results."""
    from src.agent.nodes.retrieve_memories import retrieve_memories

    state: AgentState = {
        "user_input": "/ask What do I know about sleep?",
        "cleaned_input": "What do I know about sleep?",
        "intent": "query",
        "memories": [],
        "response": "",
        "error": None,
    }

    with patch("src.agent.nodes.retrieve_memories.db_search_memories") as mock_search:
        mock_search.return_value = []

        result = await retrieve_memories(state)

        assert result["memories"] == []


async def test_retrieve_memories_node_handles_error() -> None:
    """Test that retrieve_memories node handles errors gracefully."""
    from src.agent.nodes.retrieve_memories import retrieve_memories

    state: AgentState = {
        "user_input": "/ask What do I know about habits?",
        "cleaned_input": "What do I know about habits?",
        "intent": "query",
        "memories": [],
        "response": "",
        "error": None,
    }

    with patch("src.agent.nodes.retrieve_memories.db_search_memories") as mock_search:
        mock_search.side_effect = Exception("Database error")

        result = await retrieve_memories(state)

        assert result["error"] is not None
        assert "Failed to retrieve memories" in result["error"]
