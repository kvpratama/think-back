"""Vector store operations for memory storage and retrieval using LangChain Supabase integration.

This module handles all vector store operations including:
- Generating embeddings using Gemini
- Saving memories with vector embeddings
- Searching memories using vector similarity
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from functools import lru_cache

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.agent.state import DuplicateMatch, Memory
from src.db.client import get_supabase_client

logger = logging.getLogger(__name__)


@lru_cache
def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Create and return the embeddings instance (cached singleton).

    Returns:
        GoogleGenerativeAIEmbeddings configured instance.
    """
    from src.core.config import VECTOR_DIMENSIONS, get_settings

    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.gemini_api_key,
        output_dimensionality=VECTOR_DIMENSIONS,
    )


async def save_memory(
    content: str,
    summary: str | None = None,
    *,
    user_settings_id: str,
) -> Memory:
    """Save a memory with its embedding to the database.

    Args:
        content: The memory content text (already cleaned of command prefixes).
        summary: Optional summary of the memory. Defaults to content if not provided.
        user_settings_id: The user_settings UUID that owns this memory.

    Returns:
        The inserted memory record including its generated ID.
    """
    client = get_supabase_client()
    embeddings = _get_embeddings()

    vectors = await asyncio.to_thread(embeddings.embed_documents, [content])
    embedding = vectors[0]

    metadata = {"summary": summary or content}

    result = (
        client.table("memories")
        .insert(
            {
                "content": content,
                "metadata": metadata,
                "embedding": embedding,
                "user_settings_id": user_settings_id,
            }
        )
        .execute()
    )

    row = result.data[0]
    return {"id": uuid.UUID(row["id"]), "content": content}


async def search_memories(
    query: str,
    *,
    user_settings_id: str,
    top_k: int = 3,
    threshold: float = 0.6,
) -> list[Memory]:
    """Search for memories similar to the query using vector similarity.

    Args:
        query: The search query text.
        user_settings_id: The user_settings UUID to scope the search.
        top_k: Number of results to return. Defaults to 3.
        threshold: Minimum similarity score. Defaults to 0.6.

    Returns:
        A list of matching memory records with their content and metadata.
    """
    client = get_supabase_client()
    embeddings = _get_embeddings()

    vectors = await asyncio.to_thread(embeddings.embed_documents, [query])
    query_embedding = vectors[0]

    response = await asyncio.to_thread(
        lambda: client.rpc(
            "match_memories",
            params={
                "query_embedding": query_embedding,
                "p_user_settings_id": user_settings_id,
                "match_count": top_k,
            },
        ).execute()
    )

    results: list[Memory] = []
    for row in response.data:
        similarity = row.get("similarity", 0.0)
        if similarity >= threshold:
            results.append(
                {
                    "content": row["content"],
                    "similarity": similarity,
                }
            )

    logger.debug(
        "Search returned %d results, scores: %s",
        len(results),
        [r.get("similarity") for r in results],
    )
    return results


async def find_duplicates(
    content: str,
    *,
    user_settings_id: str,
) -> list[DuplicateMatch]:
    """Check for duplicate memories by exact text match and semantic similarity.

    Args:
        content: The memory content to check for duplicates.
        user_settings_id: The user_settings UUID to scope the search.

    Returns:
        A list of duplicate records.
    """
    client = get_supabase_client()

    exact_response = await asyncio.to_thread(
        client.table("memories")
        .select("id, content")
        .eq("content", content)
        .eq("user_settings_id", user_settings_id)
        .execute
    )
    exact_contents: set[str] = set()
    results: list[DuplicateMatch] = []
    for row in exact_response.data:
        exact_contents.add(row["content"])
        results.append(
            {
                "content": row["content"],
                "similarity": 1.0,
                "match_type": "exact",
            }
        )

    semantic_matches = await search_memories(
        content, user_settings_id=user_settings_id, top_k=3, threshold=0.85
    )
    for match in semantic_matches:
        if match["content"] not in exact_contents:
            results.append(
                {
                    "content": match["content"],
                    "similarity": match.get("similarity", 0.0),
                    "match_type": "semantic",
                }
            )

    return results
