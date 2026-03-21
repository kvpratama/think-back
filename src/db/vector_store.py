"""Vector store operations for memory storage and retrieval using LangChain Supabase integration.

This module handles all vector store operations including:
- Generating embeddings using Gemini
- Saving memories with vector embeddings
- Searching memories using vector similarity
"""

from __future__ import annotations

import logging
from functools import lru_cache

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from src.agent.state import Memory
from src.db.client import get_supabase_client

logger = logging.getLogger(__name__)


@lru_cache
def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create and return the embeddings instance (cached singleton).

    Returns:
        GoogleGenerativeAIEmbeddings configured instance.
    """
    from src.core.config import get_settings

    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=SecretStr(settings.gemini_api_key),
        output_dimensionality=settings.vector_dimensions,
    )


@lru_cache
def _get_vector_store() -> SupabaseVectorStore:
    """Create and return the Supabase vector store instance (cached singleton).

    Returns:
        SupabaseVectorStore configured instance.
    """
    client = get_supabase_client()
    embeddings = _get_embeddings()

    return SupabaseVectorStore(
        embedding=embeddings,
        client=client,
        table_name="memories",
        query_name="match_memories",
    )


async def get_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text.

    Uses Google Gemini Embeddings to generate a vector representation
    of the input text.

    Args:
        text: The text to embed.

    Returns:
        A list of floats representing the embedding vector.

    Example:
        >>> embedding = await get_embedding("Hello, world!")
        >>> len(embedding)
        768
    """
    embeddings = _get_embeddings()
    return await embeddings.aembed_query(text)


async def save_memory(content: str, summary: str | None = None) -> Memory:
    """Save a memory with its embedding to the database.

    Args:
        content: The memory content text (already cleaned of command prefixes).
        summary: Optional summary of the memory. Defaults to content if not provided.

    Returns:
        The inserted memory record including its generated ID.

    Example:
        >>> result = await save_memory("Consistency beats intensity.")
        >>> result["id"]
        UUID('...')
    """
    vector_store = _get_vector_store()

    # Create a document with metadata
    metadata = {"summary": summary or content}
    document = Document(page_content=content, metadata=metadata)

    # Use aadd_documents to insert with embeddings generated automatically
    ids = await vector_store.aadd_documents([document])

    return {"id": ids[0], "content": content}


async def search_memories(
    query: str,
    top_k: int = 3,
    threshold: float = 0.6,
) -> list[Memory]:
    """Search for memories similar to the query using vector similarity.

    Performs similarity search using the embedding of the query
    against stored memory embeddings.

    Args:
        query: The search query text (already cleaned of command prefixes).
        top_k: Number of results to return. Defaults to 3.
        threshold: Minimum similarity score to consider a match. Defaults to 0.6.

    Returns:
        A list of matching memory records with their content and metadata.

    Example:
        >>> memories = await search_memories("habits", top_k=5)
        >>> for memory in memories:
        ...     print(memory["content"])
    """
    vector_store = _get_vector_store()

    # Perform similarity search
    docs_with_scores = vector_store.similarity_search_with_relevance_scores(query, k=top_k)

    # Convert to the expected format
    results: list[Memory] = []
    for doc, score in docs_with_scores:
        if score >= threshold:
            result: Memory = {
                "content": doc.page_content,
                "id": doc.metadata.get("id"),
                "similarity": score,
            }
            results.append(result)
    logger.debug("Search results: %s", results)
    return results
