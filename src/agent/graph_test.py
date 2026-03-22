"""Tests for the ThinkBack agent graph."""

import uuid

from src.agent.state import AgentState


def test_build_graph_returns_compiled_graph() -> None:
    """Test that build_graph returns a compiled LangGraph."""
    from src.agent.graph import build_graph

    graph = build_graph()

    assert graph is not None


async def test_graph_save_flow() -> None:
    """Test the save memory flow through the graph."""
    from unittest.mock import patch

    from src.agent.graph import build_graph

    graph = build_graph()

    initial_state: AgentState = {
        "user_input": "/save Consistency beats intensity",
        "cleaned_input": "",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
        "messages": [],
    }

    with patch("src.agent.nodes.save_memory.db_save_memory") as mock_save:
        mock_save.return_value = {
            "id": uuid.UUID("00000000-0000-0000-0000-000000000001"),
            "content": "Consistency beats intensity",
        }

        result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": "test"}})

        assert result["intent"] == "save"
        assert result["response"] == "Memory saved."
        assert result["cleaned_input"] == "Consistency beats intensity"


async def test_graph_query_flow() -> None:
    """Test the query knowledge flow through the graph."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from src.agent.graph import build_graph

    graph = build_graph()

    initial_state: AgentState = {
        "user_input": "/ask What do I know about habits?",
        "cleaned_input": "",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
        "messages": [],
    }

    from src.agent.nodes.generate_answer import _get_llm
    from src.core.config import get_settings

    _get_llm.cache_clear()
    get_settings.cache_clear()

    try:
        with patch("src.agent.nodes.retrieve_memories.db_search_memories") as mock_search:
            with patch("src.agent.nodes.generate_answer.init_chat_model") as mock_init_model:
                with patch("src.core.config.Settings") as mock_settings:
                    mock_settings_instance = MagicMock()
                    mock_settings_instance.llm_model = "gpt-4o-mini"
                    mock_settings_instance.llm_provider = "openai"
                    mock_settings_instance.openai_api_key.get_secret_value.return_value = "test-key"
                    mock_settings_instance.llm_provider_base_url = "https://api.openai.com/v1"
                    mock_settings.return_value = mock_settings_instance

                    mock_search.return_value = [
                        {
                            "content": "Consistency beats intensity",
                            "summary": "habits",
                            "id": uuid.UUID("00000000-0000-0000-0000-000000000001"),
                        }
                    ]

                    mock_model = MagicMock()
                    mock_model.ainvoke = AsyncMock()
                    mock_model.ainvoke.return_value.content = (
                        "From your saved memories:\n\n• Consistency beats intensity."
                    )
                    mock_init_model.return_value = mock_model

                    result = await graph.ainvoke(
                        initial_state, config={"configurable": {"thread_id": "test"}}
                    )

                    assert result["intent"] == "query"
                    assert "From your saved memories" in result["response"]
    finally:
        _get_llm.cache_clear()
        get_settings.cache_clear()


async def test_graph_save_retry_behavior() -> None:
    """Test that the graph retries the save_memory node on failure."""
    from unittest.mock import patch

    from src.agent.graph import build_graph

    graph = build_graph()

    initial_state: AgentState = {
        "user_input": "/save Consistency beats intensity",
        "cleaned_input": "",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
        "messages": [],
    }

    with patch("src.agent.nodes.save_memory.db_save_memory") as mock_save:
        # Fail twice, then succeed
        mock_save.side_effect = [
            Exception("Transient error 1"),
            Exception("Transient error 2"),
            {
                "id": uuid.UUID("00000000-0000-0000-0000-000000000001"),
                "content": "Consistency beats intensity",
            },
        ]

        result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": "test"}})

        assert result["intent"] == "save"
        assert result["response"] == "Memory saved."
        assert mock_save.call_count == 3
