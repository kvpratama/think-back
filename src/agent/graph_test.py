"""Tests for the ThinkBack agent graph."""

from unittest.mock import AsyncMock, MagicMock, patch


def test_build_graph_returns_compiled_graph() -> None:
    """Test that build_graph returns a compiled agent."""
    from langgraph.checkpoint.memory import InMemorySaver

    from src.agent.graph import build_graph

    with patch("src.agent.graph._get_llm") as mock_get_llm:
        mock_get_llm.return_value = MagicMock()
        graph = build_graph(checkpointer=InMemorySaver())
        assert graph is not None


async def test_graph_query_flow() -> None:
    """Test the query flow through the agent."""
    from langgraph.checkpoint.memory import InMemorySaver

    from src.agent.graph import _get_llm, build_graph
    from src.core.config import get_settings

    _get_llm.cache_clear()
    get_settings.cache_clear()

    try:
        with patch("src.agent.graph.init_chat_model") as mock_init_model:
            with patch("src.core.config.Settings") as mock_settings_cls:
                mock_settings = MagicMock()
                mock_settings.llm_model = "gpt-4o-mini"
                mock_settings.llm_provider = "openai"
                mock_settings.openai_api_key.get_secret_value.return_value = "test-key"
                mock_settings.llm_provider_base_url = "https://api.openai.com/v1"
                mock_settings.max_turns = 5
                mock_settings_cls.return_value = mock_settings

                from langchain_core.messages import AIMessage

                # Create a mock LLM that returns a real AIMessage (no tool calls)
                mock_llm = MagicMock()
                mock_response = AIMessage(
                    content="Based on your memories: Consistency beats intensity.",
                    id="test-id",
                )
                mock_llm.ainvoke = AsyncMock(return_value=mock_response)
                mock_llm.bind_tools = MagicMock(return_value=mock_llm)
                mock_init_model.return_value = mock_llm

                with patch("src.agent.tools.db_search_memories") as mock_search:
                    mock_search.return_value = [
                        {"content": "Consistency beats intensity", "similarity": 0.9},
                    ]

                    graph = build_graph(checkpointer=InMemorySaver())
                    result = await graph.ainvoke(
                        {"messages": [{"role": "user", "content": "What do I know about habits?"}]},
                        config={"configurable": {"thread_id": "test-query"}},
                    )

                    last_msg = result["messages"][-1]
                    assert last_msg.content  # Agent produced a response
    finally:
        _get_llm.cache_clear()
        get_settings.cache_clear()


def test_build_graph_includes_trim_middleware() -> None:
    """build_graph includes trim_messages_by_turns in the middleware list."""
    from langgraph.checkpoint.memory import InMemorySaver

    from src.agent.graph import build_graph

    with patch("src.agent.graph._get_llm") as mock_get_llm:
        with patch("src.agent.graph.create_agent") as mock_create_agent:
            mock_get_llm.return_value = MagicMock()
            mock_create_agent.return_value = MagicMock()
            build_graph(checkpointer=InMemorySaver())

            call_kwargs = mock_create_agent.call_args.kwargs
            middleware_list = call_kwargs.get("middleware", [])
            middleware_names = [
                getattr(m, "__name__", "") or getattr(m, "name", "") or type(m).__name__
                for m in middleware_list
            ]
            assert "trim_messages_by_turns" in middleware_names
            trim_idx = middleware_names.index("trim_messages_by_turns")
            tool_limit_idx = middleware_names.index("ToolCallLimitMiddleware")
            assert trim_idx < tool_limit_idx
