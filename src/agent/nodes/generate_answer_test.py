"""Tests for the generate_answer node."""

from unittest.mock import AsyncMock, MagicMock, patch

from src.agent.state import AgentState


async def test_generate_answer_node_creates_response() -> None:
    """Test that generate_answer node creates a response from memories."""
    from src.agent.nodes.generate_answer import _get_llm, generate_answer

    state: AgentState = {
        "user_input": "/ask What do I know about habits?",
        "cleaned_input": "What do I know about habits?",
        "intent": "query",
        "memories": [
            {"content": "Consistency beats intensity", "summary": "habits", "id": "test-id-1"},
        ],
        "response": "",
        "error": None,
        "messages": [],
    }

    with patch("src.agent.nodes.generate_answer.init_chat_model") as mock_init_model:
        with patch("src.core.config.Settings") as mock_settings:
            mock_settings_instance = MagicMock()
            mock_settings_instance.llm_model = "gpt-4o-mini"
            mock_settings_instance.llm_provider = "openai"
            mock_settings_instance.openai_api_key = "test-key"
            mock_settings_instance.llm_provider_base_url = "https://api.openai.com/v1"
            mock_settings.return_value = mock_settings_instance

            mock_model = MagicMock()
            mock_model.ainvoke = AsyncMock()
            mock_model.ainvoke.return_value.content = (
                "From your saved memories:\n\n• Consistency beats intensity."
            )
            mock_init_model.return_value = mock_model

            # Clear cache to ensure fresh instances
            _get_llm.cache_clear()

            result = await generate_answer(state)

            assert "From your saved memories" in result["response"]
            mock_init_model.assert_called_once()


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

    result = await generate_answer(state)

    assert "saved memories" in result["response"].lower()


async def test_generate_answer_node_handles_error() -> None:
    """Test that generate_answer node handles LLM errors gracefully."""
    from src.agent.nodes.generate_answer import _get_llm, generate_answer

    state: AgentState = {
        "user_input": "/ask What do I know about habits?",
        "cleaned_input": "What do I know about habits?",
        "intent": "query",
        "memories": [{"content": "test memory", "id": "test-id-1"}],
        "response": "",
        "error": None,
        "messages": [],
    }

    with patch("src.agent.nodes.generate_answer.init_chat_model") as mock_init_model:
        with patch("src.core.config.Settings") as mock_settings:
            mock_settings_instance = MagicMock()
            mock_settings_instance.llm_model = "gpt-4o-mini"
            mock_settings_instance.llm_provider = "openai"
            mock_settings_instance.openai_api_key = "test-key"
            mock_settings_instance.llm_provider_base_url = "https://api.openai.com/v1"
            mock_settings.return_value = mock_settings_instance

            mock_model = MagicMock()
            mock_model.ainvoke = AsyncMock(side_effect=Exception("API error"))
            mock_init_model.return_value = mock_model

            # Clear cache to ensure fresh instances
            _get_llm.cache_clear()

            result = await generate_answer(state)

            assert result["error"] is not None
            assert "Failed to generate answer" in result["error"]
