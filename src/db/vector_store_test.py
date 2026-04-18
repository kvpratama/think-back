"""Tests for the vector store operations module."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create a mock settings object."""
    settings = MagicMock()
    settings.gemini_api_key = "test-gemini-key"
    settings.embedding_model = "gemini-embedding-exp-03-16"
    settings.supabase_url.get_secret_value.return_value = "https://test.supabase.co"
    settings.supabase_key = "test-key"
    return settings


async def test_save_memory_inserts_into_database(
    mock_settings: MagicMock,
) -> None:
    """Test that save_memory inserts a memory with user_settings_id."""
    from src.db.vector_store import _get_embeddings, save_memory

    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 768]

    mock_execute = MagicMock()
    mock_execute.data = [{"id": "00000000-0000-0000-0000-000000000001"}]
    mock_client.table.return_value.insert.return_value.execute.return_value = mock_execute

    with (
        patch("src.db.vector_store.get_supabase_client", return_value=mock_client),
        patch("src.db.vector_store._get_embeddings", return_value=mock_embeddings),
        patch("src.core.config.get_settings", return_value=mock_settings),
    ):
        _get_embeddings.cache_clear()

        result = await save_memory(
            "test content",
            user_settings_id="usr-123",
        )

        mock_client.table.assert_called_with("memories")
        insert_args = mock_client.table.return_value.insert.call_args[0][0]
        assert insert_args["user_settings_id"] == "usr-123"
        assert insert_args["content"] == "test content"
        assert result is not None


async def test_save_memory_uses_content_directly(
    mock_settings: MagicMock,
) -> None:
    """Test that save_memory stores exact content passed in."""
    from src.db.vector_store import _get_embeddings, save_memory

    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 768]

    mock_execute = MagicMock()
    mock_execute.data = [{"id": "00000000-0000-0000-0000-000000000001"}]
    mock_client.table.return_value.insert.return_value.execute.return_value = mock_execute

    with (
        patch("src.db.vector_store.get_supabase_client", return_value=mock_client),
        patch("src.db.vector_store._get_embeddings", return_value=mock_embeddings),
        patch("src.core.config.get_settings", return_value=mock_settings),
    ):
        _get_embeddings.cache_clear()

        await save_memory(
            "Remember this",
            user_settings_id="usr-123",
        )

        insert_args = mock_client.table.return_value.insert.call_args[0][0]
        assert insert_args["content"] == "Remember this"


async def test_search_memories_performs_vector_search(
    mock_settings: MagicMock,
) -> None:
    """Test that search_memories performs vector similarity search scoped by user."""
    from src.db.vector_store import _get_embeddings, search_memories

    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 768]

    mock_rpc_response = MagicMock()
    mock_rpc_response.data = [
        {
            "id": "id-1",
            "content": "memory 1",
            "metadata": {"summary": "summary 1"},
            "similarity": 0.85,
        },
        {
            "id": "id-2",
            "content": "memory 2",
            "metadata": {"summary": "summary 2"},
            "similarity": 0.75,
        },
    ]
    mock_client.rpc.return_value.execute.return_value = mock_rpc_response

    with (
        patch("src.db.vector_store.get_supabase_client", return_value=mock_client),
        patch("src.db.vector_store._get_embeddings", return_value=mock_embeddings),
        patch("src.core.config.get_settings", return_value=mock_settings),
    ):
        _get_embeddings.cache_clear()

        result = await search_memories("test query", user_settings_id="usr-123", top_k=3)

        assert len(result) == 2
        assert result[0]["content"] == "memory 1"
        assert result[0]["similarity"] == 0.85
        mock_client.rpc.assert_called_once()
        rpc_args = mock_client.rpc.call_args
        assert rpc_args[0][0] == "match_memories"
        assert rpc_args[1]["params"]["p_user_settings_id"] == "usr-123"


async def test_search_memories_uses_query_directly(
    mock_settings: MagicMock,
) -> None:
    """Test that search_memories uses query directly (no command stripping)."""
    from src.db.vector_store import _get_embeddings, search_memories

    mock_client = MagicMock()
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 768]

    mock_rpc_response = MagicMock()
    mock_rpc_response.data = [
        {
            "id": "id-1",
            "content": "memory 1",
            "metadata": {"summary": "summary 1"},
            "similarity": 0.85,
        },
    ]
    mock_client.rpc.return_value.execute.return_value = mock_rpc_response

    with (
        patch("src.db.vector_store.get_supabase_client", return_value=mock_client),
        patch("src.db.vector_store._get_embeddings", return_value=mock_embeddings),
        patch("src.core.config.get_settings", return_value=mock_settings),
    ):
        _get_embeddings.cache_clear()

        await search_memories("What are my habits?", user_settings_id="usr-123")

        mock_client.rpc.assert_called_once()


async def test_find_duplicates_returns_exact_match(
    mock_settings: MagicMock,
) -> None:
    """Test that find_duplicates detects an exact content match scoped by user."""
    from src.db.vector_store import _get_embeddings, find_duplicates

    mock_client = MagicMock()
    mock_execute = MagicMock()
    mock_execute.data = [
        {"id": "00000000-0000-0000-0000-000000000001", "content": "Exercise is good"}
    ]
    # Chain: .table().select().eq("content", ...).eq("user_settings_id", ...).execute()
    mock_eq_chain = MagicMock()
    mock_eq_chain.execute.return_value = mock_execute
    mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value = (
        mock_eq_chain
    )

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 768]

    mock_rpc_response = MagicMock()
    mock_rpc_response.data = []
    mock_client.rpc.return_value.execute.return_value = mock_rpc_response

    with (
        patch("src.db.vector_store.get_supabase_client", return_value=mock_client),
        patch("src.db.vector_store._get_embeddings", return_value=mock_embeddings),
        patch("src.core.config.get_settings", return_value=mock_settings),
    ):
        _get_embeddings.cache_clear()

        result = await find_duplicates("Exercise is good", user_settings_id="usr-123")

        assert len(result) == 1
        assert result[0]["content"] == "Exercise is good"
        assert result[0]["match_type"] == "exact"
        assert result[0]["similarity"] == 1.0


async def test_find_duplicates_returns_empty_when_no_matches(
    mock_settings: MagicMock,
) -> None:
    """Test that find_duplicates returns empty list when no duplicates exist."""
    from src.db.vector_store import _get_embeddings, find_duplicates

    mock_client = MagicMock()
    mock_execute = MagicMock()
    mock_execute.data = []
    mock_eq_chain = MagicMock()
    mock_eq_chain.execute.return_value = mock_execute
    mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value = (
        mock_eq_chain
    )

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 768]

    mock_rpc_response = MagicMock()
    mock_rpc_response.data = []
    mock_client.rpc.return_value.execute.return_value = mock_rpc_response

    with (
        patch("src.db.vector_store.get_supabase_client", return_value=mock_client),
        patch("src.db.vector_store._get_embeddings", return_value=mock_embeddings),
        patch("src.core.config.get_settings", return_value=mock_settings),
    ):
        _get_embeddings.cache_clear()

        result = await find_duplicates("Something totally new", user_settings_id="usr-123")

        assert result == []


async def test_find_duplicates_returns_semantic_matches(
    mock_settings: MagicMock,
) -> None:
    """Test that find_duplicates returns semantic matches above 0.85 threshold."""
    from src.db.vector_store import _get_embeddings, find_duplicates

    mock_client = MagicMock()
    mock_execute = MagicMock()
    mock_execute.data = []  # No exact match
    mock_eq_chain = MagicMock()
    mock_eq_chain.execute.return_value = mock_execute
    mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value = (
        mock_eq_chain
    )

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 768]

    mock_rpc_response = MagicMock()
    mock_rpc_response.data = [
        {
            "id": "id-1",
            "content": "Working out is key to staying healthy",
            "metadata": {},
            "similarity": 0.91,
        },
    ]
    mock_client.rpc.return_value.execute.return_value = mock_rpc_response

    with (
        patch("src.db.vector_store.get_supabase_client", return_value=mock_client),
        patch("src.db.vector_store._get_embeddings", return_value=mock_embeddings),
        patch("src.core.config.get_settings", return_value=mock_settings),
    ):
        _get_embeddings.cache_clear()

        result = await find_duplicates("Exercise is good for health", user_settings_id="usr-123")

        assert len(result) == 1
        assert result[0]["content"] == "Working out is key to staying healthy"
        assert result[0]["match_type"] == "semantic"
        assert result[0]["similarity"] == 0.91


async def test_find_duplicates_deduplicates_exact_and_semantic_overlap(
    mock_settings: MagicMock,
) -> None:
    """Test exact+semantic overlap is deduplicated, returning only once as 'exact'."""
    from src.db.vector_store import _get_embeddings, find_duplicates

    mock_client = MagicMock()
    mock_execute = MagicMock()
    mock_execute.data = [
        {"id": "00000000-0000-0000-0000-000000000001", "content": "Exercise is good"}
    ]
    mock_eq_chain = MagicMock()
    mock_eq_chain.execute.return_value = mock_execute
    mock_client.table.return_value.select.return_value.eq.return_value.eq.return_value = (
        mock_eq_chain
    )

    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.return_value = [[0.1] * 768]

    # Same content also shows up as a semantic match
    mock_rpc_response = MagicMock()
    mock_rpc_response.data = [
        {"id": "id-1", "content": "Exercise is good", "metadata": {}, "similarity": 0.99},
        {"id": "id-2", "content": "Working out is healthy", "metadata": {}, "similarity": 0.88},
    ]
    mock_client.rpc.return_value.execute.return_value = mock_rpc_response

    with (
        patch("src.db.vector_store.get_supabase_client", return_value=mock_client),
        patch("src.db.vector_store._get_embeddings", return_value=mock_embeddings),
        patch("src.core.config.get_settings", return_value=mock_settings),
    ):
        _get_embeddings.cache_clear()

        result = await find_duplicates("Exercise is good", user_settings_id="usr-123")

        assert len(result) == 2
        # First result is the exact match (not duplicated)
        exact_results = [r for r in result if r["match_type"] == "exact"]
        semantic_results = [r for r in result if r["match_type"] == "semantic"]
        assert len(exact_results) == 1
        assert len(semantic_results) == 1
        assert semantic_results[0]["content"] == "Working out is healthy"
