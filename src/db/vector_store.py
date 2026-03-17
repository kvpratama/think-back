"""Vector store operations for memory storage and retrieval.

This module handles all pgvector operations including:
- Generating embeddings using Gemini
- Saving memories with vector embeddings
- Searching memories using vector similarity
"""

from __future__ import annotations

from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.db.client import get_supabase_client


def get_embedding(text: str) -> list[float]:
    """Generate an embedding vector for the given text.

    Uses Google Gemini Embeddings to generate a vector representation
    of the input text.

    Args:
        text: The text to embed.

    Returns:
        A list of floats representing the embedding vector.

    Example:
        >>> embedding = get_embedding("Hello, world!")
        >>> len(embedding)
        768
    """
    from src.core.config import Settings

    settings = Settings()
    embeddings = GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
    )
    return embeddings.embed_query(text)


async def save_memory(content: str, summary: str | None = None) -> dict[str, Any]:
    """Save a memory with its embedding to the database.

    Args:
        content: The memory content text.
        summary: Optional summary of the memory. Defaults to content if not provided.

    Returns:
        The inserted memory record including its generated ID.

    Example:
        >>> result = await save_memory("Consistency beats intensity.")
        >>> result["id"]
        UUID('...')
    """
    embedding = get_embedding(content)

    client = get_supabase_client()
    response = (
        client.table("memories")
        .insert(
            {
                "content": content,
                "summary": summary or content,
                "embedding": embedding,
            }
        )
        .execute()
    )

    return response.data[0]  # type: ignore[no-any-return]


async def search_memories(
    query: str,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Search for memories similar to the query using vector similarity.

    Performs a pgvector similarity search using the embedding of the query
    against stored memory embeddings.

    Args:
        query: The search query text.
        top_k: Number of results to return. Defaults to 3.

    Returns:
        A list of matching memory records with their content and metadata.

    Example:
        >>> memories = await search_memories("habits", top_k=5)
        >>> for memory in memories:
        ...     print(memory["content"])
    """
    query_embedding = get_embedding(query)

    client = get_supabase_client()
    response = client.rpc(
        "match_memories",
        {
            "query_embedding": query_embedding,
            "match_threshold": 0.5,
            "match_count": top_k,
        },
    ).execute()

    return response.data  # type: ignore[no-any-return]
