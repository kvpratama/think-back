"""Tests for the vector store operations module."""

from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest


@pytest.fixture
def mock_settings() -> MagicMock:
    """Create a mock settings object."""
    settings = MagicMock()
    settings.gemini_api_key = "test-gemini-key"
    settings.embedding_model = "gemini-embedding-exp-03-16"
    settings.vector_dimensions = 768
    settings.supabase_url = "https://test.supabase.co"
    settings.supabase_key = "test-key"
    return settings


def test_get_embedding_generates_embedding(
    mock_settings: MagicMock,
) -> None:
    """Test that get_embedding generates an embedding for text."""
    from src.db.vector_store import get_embedding

    with patch("src.db.vector_store.GoogleGenerativeAIEmbeddings") as mock_embeddings:
        with patch("src.core.config.Settings", return_value=mock_settings):
            mock_embedder = MagicMock()
            mock_embedder.embed_query.return_value = [0.1] * 768
            mock_embeddings.return_value = mock_embedder

            result = get_embedding("test text")

            assert result == [0.1] * 768
            mock_embeddings.assert_called_once_with(
                model="gemini-embedding-exp-03-16",
                google_api_key="test-gemini-key",
            )
            mock_embedder.embed_query.assert_called_once_with("test text")


@pytest.mark.asyncio
async def test_save_memory_inserts_into_database(
    mock_settings: MagicMock,
) -> None:
    """Test that save_memory inserts a memory with embedding into the database."""
    from src.db.vector_store import save_memory

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [{"id": str(uuid4())}]

    mock_execute = MagicMock(return_value=mock_response)
    mock_insert = MagicMock()
    mock_insert.execute = mock_execute
    mock_table = MagicMock()
    mock_table.insert.return_value = mock_insert
    mock_client.table.return_value = mock_table

    with patch("src.db.vector_store.get_supabase_client", return_value=mock_client):
        with patch("src.core.config.Settings", return_value=mock_settings):
            with patch("src.db.vector_store.get_embedding", return_value=[0.1] * 768):
                result = await save_memory("test content")

                mock_client.table.assert_called_once_with("memories")
                mock_client.table.return_value.insert.assert_called_once()
                assert result is not None


@pytest.mark.asyncio
async def test_search_memories_performs_vector_search(
    mock_settings: MagicMock,
) -> None:
    """Test that search_memories performs vector similarity search."""
    from src.db.vector_store import search_memories

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [
        {"content": "memory 1", "summary": "summary 1"},
        {"content": "memory 2", "summary": "summary 2"},
    ]

    mock_execute = MagicMock(return_value=mock_response)
    mock_rpc = MagicMock()
    mock_rpc.execute = mock_execute
    mock_client.rpc.return_value = mock_rpc

    with patch("src.db.vector_store.get_supabase_client", return_value=mock_client):
        with patch("src.core.config.Settings", return_value=mock_settings):
            with patch("src.db.vector_store.get_embedding", return_value=[0.1] * 768):
                result = await search_memories("test query", top_k=3)

                assert len(result) == 2
                mock_client.rpc.assert_called_once()
