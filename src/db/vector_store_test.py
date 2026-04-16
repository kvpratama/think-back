"""Tests for the vector store operations module."""

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

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
    """Test that save_memory inserts a memory with embedding into the database."""
    from src.db.vector_store import _get_embeddings, _get_vector_store, save_memory

    mock_client = MagicMock()
    mock_vector_store = MagicMock()
    memory_id = str(uuid4())
    mock_vector_store.aadd_documents = AsyncMock(return_value=[memory_id])

    with patch("src.db.vector_store.SupabaseVectorStore", return_value=mock_vector_store):
        with patch("src.db.vector_store.get_supabase_client", return_value=mock_client):
            with patch("src.core.config.get_settings", return_value=mock_settings):
                # Clear cache to ensure fresh instances
                _get_embeddings.cache_clear()
                _get_vector_store.cache_clear()

                result = await save_memory("test content")

                mock_vector_store.aadd_documents.assert_called_once()
                assert result is not None


async def test_save_memory_uses_content_directly(
    mock_settings: MagicMock,
) -> None:
    """Test that save_memory uses content directly (no command stripping)."""
    from src.db.vector_store import _get_embeddings, _get_vector_store, save_memory

    mock_client = MagicMock()
    mock_vector_store = MagicMock()
    memory_id = str(uuid4())
    mock_vector_store.aadd_documents = AsyncMock(return_value=[memory_id])

    with patch("src.db.vector_store.SupabaseVectorStore", return_value=mock_vector_store):
        with patch("src.db.vector_store.get_supabase_client", return_value=mock_client):
            with patch("src.core.config.get_settings", return_value=mock_settings):
                _get_embeddings.cache_clear()
                _get_vector_store.cache_clear()

                await save_memory("Remember this")

                # Verify the document was stored with the exact content passed in
                call_args = mock_vector_store.aadd_documents.call_args
                document = call_args[0][0][0]
                assert document.page_content == "Remember this"


async def test_search_memories_performs_vector_search(
    mock_settings: MagicMock,
) -> None:
    """Test that search_memories performs vector similarity search."""
    from src.db.vector_store import _get_embeddings, _get_vector_store, search_memories

    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_relevance_scores = MagicMock(
        return_value=[
            (MagicMock(page_content="memory 1", metadata={"summary": "summary 1"}), 0.85),
            (MagicMock(page_content="memory 2", metadata={"summary": "summary 2"}), 0.75),
        ]
    )

    with patch("src.db.vector_store.SupabaseVectorStore", return_value=mock_vector_store):
        with patch("src.db.vector_store.get_supabase_client"):
            with patch("src.core.config.get_settings", return_value=mock_settings):
                _get_embeddings.cache_clear()
                _get_vector_store.cache_clear()

                result = await search_memories("test query", top_k=3)

                assert len(result) == 2
                assert result[0]["content"] == "memory 1"
                assert result[0]["similarity"] == 0.85
                mock_vector_store.similarity_search_with_relevance_scores.assert_called_once_with(
                    "test query", k=3
                )


async def test_search_memories_uses_query_directly(
    mock_settings: MagicMock,
) -> None:
    """Test that search_memories uses query directly (no command stripping)."""
    from src.db.vector_store import _get_embeddings, _get_vector_store, search_memories

    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_relevance_scores = MagicMock(
        return_value=[
            (MagicMock(page_content="memory 1", metadata={"summary": "summary 1"}), 0.85),
        ]
    )

    with patch("src.db.vector_store.SupabaseVectorStore", return_value=mock_vector_store):
        with patch("src.db.vector_store.get_supabase_client"):
            with patch("src.core.config.get_settings", return_value=mock_settings):
                _get_embeddings.cache_clear()
                _get_vector_store.cache_clear()

                await search_memories("What are my habits?")

                mock_vector_store.similarity_search_with_relevance_scores.assert_called_once_with(
                    "What are my habits?", k=3
                )


async def test_find_duplicates_returns_exact_match(
    mock_settings: MagicMock,
) -> None:
    """Test that find_duplicates detects an exact content match."""
    from src.db.vector_store import _get_embeddings, _get_vector_store, find_duplicates

    mock_client = MagicMock()
    # Simulate Supabase .table().select().eq().execute() returning one row
    mock_execute = MagicMock()
    mock_execute.data = [
        {"id": "00000000-0000-0000-0000-000000000001", "content": "Exercise is good"}
    ]
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        mock_execute
    )

    mock_vector_store = MagicMock()
    # No semantic matches (exact match only)
    mock_vector_store.similarity_search_with_relevance_scores = MagicMock(return_value=[])

    with patch("src.db.vector_store.SupabaseVectorStore", return_value=mock_vector_store):
        with patch("src.db.vector_store.get_supabase_client", return_value=mock_client):
            with patch("src.core.config.get_settings", return_value=mock_settings):
                _get_embeddings.cache_clear()
                _get_vector_store.cache_clear()

                result = await find_duplicates("Exercise is good")

                assert len(result) == 1
                assert result[0]["content"] == "Exercise is good"
                assert result[0]["match_type"] == "exact"
                assert result[0]["similarity"] == 1.0


async def test_find_duplicates_returns_empty_when_no_matches(
    mock_settings: MagicMock,
) -> None:
    """Test that find_duplicates returns empty list when no duplicates exist."""
    from src.db.vector_store import _get_embeddings, _get_vector_store, find_duplicates

    mock_client = MagicMock()
    mock_execute = MagicMock()
    mock_execute.data = []
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        mock_execute
    )

    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_relevance_scores = MagicMock(return_value=[])

    with patch("src.db.vector_store.SupabaseVectorStore", return_value=mock_vector_store):
        with patch("src.db.vector_store.get_supabase_client", return_value=mock_client):
            with patch("src.core.config.get_settings", return_value=mock_settings):
                _get_embeddings.cache_clear()
                _get_vector_store.cache_clear()

                result = await find_duplicates("Something totally new")

                assert result == []


async def test_find_duplicates_returns_semantic_matches(
    mock_settings: MagicMock,
) -> None:
    """Test that find_duplicates returns semantic matches above 0.85 threshold."""
    from src.db.vector_store import _get_embeddings, _get_vector_store, find_duplicates

    mock_client = MagicMock()
    mock_execute = MagicMock()
    mock_execute.data = []  # No exact match
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        mock_execute
    )

    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_relevance_scores = MagicMock(
        return_value=[
            (MagicMock(page_content="Working out is key to staying healthy", metadata={}), 0.91),
        ]
    )

    with patch("src.db.vector_store.SupabaseVectorStore", return_value=mock_vector_store):
        with patch("src.db.vector_store.get_supabase_client", return_value=mock_client):
            with patch("src.core.config.get_settings", return_value=mock_settings):
                _get_embeddings.cache_clear()
                _get_vector_store.cache_clear()

                result = await find_duplicates("Exercise is good for health")

                assert len(result) == 1
                assert result[0]["content"] == "Working out is key to staying healthy"
                assert result[0]["match_type"] == "semantic"
                assert result[0]["similarity"] == 0.91


async def test_find_duplicates_deduplicates_exact_and_semantic_overlap(
    mock_settings: MagicMock,
) -> None:
    """Test exact+semantic overlap is deduplicated, returning only once as 'exact'."""
    from src.db.vector_store import _get_embeddings, _get_vector_store, find_duplicates

    mock_client = MagicMock()
    mock_execute = MagicMock()
    mock_execute.data = [
        {"id": "00000000-0000-0000-0000-000000000001", "content": "Exercise is good"}
    ]
    mock_client.table.return_value.select.return_value.eq.return_value.execute.return_value = (
        mock_execute
    )

    mock_vector_store = MagicMock()
    # Same content also shows up as a semantic match
    mock_vector_store.similarity_search_with_relevance_scores = MagicMock(
        return_value=[
            (MagicMock(page_content="Exercise is good", metadata={}), 0.99),
            (MagicMock(page_content="Working out is healthy", metadata={}), 0.88),
        ]
    )

    with patch("src.db.vector_store.SupabaseVectorStore", return_value=mock_vector_store):
        with patch("src.db.vector_store.get_supabase_client", return_value=mock_client):
            with patch("src.core.config.get_settings", return_value=mock_settings):
                _get_embeddings.cache_clear()
                _get_vector_store.cache_clear()

                result = await find_duplicates("Exercise is good")

                assert len(result) == 2
                # First result is the exact match (not duplicated)
                exact_results = [r for r in result if r["match_type"] == "exact"]
                semantic_results = [r for r in result if r["match_type"] == "semantic"]
                assert len(exact_results) == 1
                assert len(semantic_results) == 1
                assert semantic_results[0]["content"] == "Working out is healthy"
