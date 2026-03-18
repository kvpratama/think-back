"""Tests for the ThinkBack agent graph."""

import pytest

from src.agent.state import AgentState


def test_build_graph_returns_compiled_graph() -> None:
    """Test that build_graph returns a compiled LangGraph."""
    from src.agent.graph import build_graph

    graph = build_graph()

    assert graph is not None


@pytest.mark.asyncio
async def test_graph_save_flow() -> None:
    """Test the save memory flow through the graph."""
    from unittest.mock import patch

    from src.agent.graph import build_graph

    graph = build_graph()

    initial_state: AgentState = {
        "user_input": "/save Consistency beats intensity",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
    }

    with patch("src.agent.nodes.save_memory.db_save_memory") as mock_save:
        mock_save.return_value = {"id": "test-id", "content": "Consistency beats intensity"}

        result = await graph.ainvoke(initial_state)  # type: ignore[attr-defined]

        assert result["intent"] == "save"
        assert result["response"] == "Memory saved."


@pytest.mark.asyncio
async def test_graph_query_flow() -> None:
    """Test the query knowledge flow through the graph."""
    from unittest.mock import MagicMock, patch

    from src.agent.graph import build_graph

    graph = build_graph()

    initial_state: AgentState = {
        "user_input": "/ask What do I know about habits?",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
    }

    with patch("src.agent.nodes.retrieve_memories.db_search_memories") as mock_search:
        with patch("src.agent.nodes.generate_answer.init_chat_model") as mock_init_model:
            with patch("src.core.config.Settings") as mock_settings:
                mock_settings_instance = MagicMock()
                mock_settings_instance.llm_model = "gpt-4o-mini"
                mock_settings_instance.llm_provider = "openai"
                mock_settings_instance.openai_api_key = "test-key"
                mock_settings_instance.llm_provider_base_url = "https://api.openai.com/v1/models"
                mock_settings.return_value = mock_settings_instance

                mock_search.return_value = [
                    {"content": "Consistency beats intensity", "summary": "habits"}
                ]

                mock_model = MagicMock()
                mock_model.invoke.return_value.content = (
                    "From your saved memories:\n\n• Consistency beats intensity."
                )
                mock_init_model.return_value = mock_model

                result = await graph.ainvoke(initial_state)  # type: ignore[attr-defined]

                assert result["intent"] == "query"
                assert "From your saved memories" in result["response"]
