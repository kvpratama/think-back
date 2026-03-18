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
    from pydantic import SecretStr

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
                api_key=SecretStr("test-gemini-key"),
                output_dimensionality=768,
            )
            mock_embedder.embed_query.assert_called_once_with("test text")


@pytest.mark.asyncio
async def test_save_memory_inserts_into_database(
    mock_settings: MagicMock,
) -> None:
    """Test that save_memory inserts a memory with embedding into the database."""
    from src.db.vector_store import save_memory

    mock_client = MagicMock()
    mock_vector_store = MagicMock()
    memory_id = str(uuid4())
    mock_vector_store.add_documents.return_value = [memory_id]

    mock_select_response = MagicMock()
    mock_select_response.data = [{"id": memory_id, "content": "test content"}]
    mock_select = MagicMock()
    mock_select.execute.return_value = mock_select_response
    mock_eq = MagicMock()
    mock_eq.execute = mock_select
    mock_select_method = MagicMock(return_value=mock_eq)
    mock_table = MagicMock()
    mock_table.select.return_value = mock_select_method
    mock_table.eq = mock_select_method
    mock_client.table.return_value = mock_table

    with patch("src.db.vector_store.SupabaseVectorStore", return_value=mock_vector_store):
        with patch("src.db.vector_store.get_supabase_client", return_value=mock_client):
            with patch("src.core.config.Settings", return_value=mock_settings):
                result = await save_memory("test content")

                mock_vector_store.add_documents.assert_called_once()
                mock_client.table.assert_called_with("memories")
                assert result is not None


@pytest.mark.asyncio
async def test_search_memories_performs_vector_search(
    mock_settings: MagicMock,
) -> None:
    """Test that search_memories performs vector similarity search."""
    from src.db.vector_store import search_memories

    mock_vector_store = MagicMock()
    mock_vector_store.similarity_search_with_relevance_scores.return_value = [
        (MagicMock(page_content="memory 1", metadata={"summary": "summary 1"}), 0.85),
        (MagicMock(page_content="memory 2", metadata={"summary": "summary 2"}), 0.75),
    ]

    with patch("src.db.vector_store.SupabaseVectorStore", return_value=mock_vector_store):
        with patch("src.core.config.Settings", return_value=mock_settings):
            result = await search_memories("test query", top_k=3)

            assert len(result) == 2
            assert result[0]["content"] == "memory 1"
            assert result[0]["similarity"] == 0.85
            mock_vector_store.similarity_search_with_relevance_scores.assert_called_once_with(
                "test query", k=3
            )
