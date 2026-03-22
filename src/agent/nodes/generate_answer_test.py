"""Tests for the generate_answer node."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.state import AgentState


async def test_generate_answer_node_creates_response() -> None:
    """Test that generate_answer node creates a response from memories."""
    from src.agent.nodes.generate_answer import generate_answer

    state: AgentState = {
        "user_input": "/ask What do I know about habits?",
        "cleaned_input": "What do I know about habits?",
        "intent": "query",
        "memories": [
            {"content": "Consistency beats intensity", "id": "test-id-1"},
        ],
        "response": "",
        "error": None,
        "messages": [],
    }

    mock_response = MagicMock()
    mock_response.content = "From your saved memories:\n\n• Consistency beats intensity."
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("src.agent.nodes.generate_answer._get_llm", return_value=mock_llm):
        result = await generate_answer(state)

        assert "From your saved memories" in result["response"]
        mock_llm.ainvoke.assert_called_once()


async def test_generate_answer_node_handles_no_memories() -> None:
    """Test that generate_answer node handles empty memories."""
    from src.agent.nodes.generate_answer import generate_answer

    state: AgentState = {
        "user_input": "/ask What do I know about sleep?",
        "cleaned_input": "What do I know about sleep?",
        "intent": "query",
        "memories": [],
        "response": "",
        "error": None,
        "messages": [],
    }

    mock_response = MagicMock()
    mock_response.content = "You have no saved knowledge about sleep yet."
    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(return_value=mock_response)

    with patch("src.agent.nodes.generate_answer._get_llm", return_value=mock_llm):
        result = await generate_answer(state)

        assert "saved" in result["response"].lower() or "knowledge" in result["response"].lower()


async def test_generate_answer_node_handles_error() -> None:
    """Test that generate_answer node handles LLM errors gracefully."""
    from src.agent.nodes.generate_answer import generate_answer

    state: AgentState = {
        "user_input": "/ask What do I know about habits?",
        "cleaned_input": "What do I know about habits?",
        "intent": "query",
        "memories": [{"content": "test memory", "id": "test-id-1"}],
        "response": "",
        "error": None,
        "messages": [],
    }

    mock_llm = MagicMock()
    mock_llm.ainvoke = AsyncMock(side_effect=Exception("API error"))

    with patch("src.agent.nodes.generate_answer._get_llm", return_value=mock_llm):
        result = await generate_answer(state)

        assert result["error"] is not None
        assert "Failed to generate answer" in result["error"]
