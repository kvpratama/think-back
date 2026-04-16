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

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.documents import Document
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

    return {"id": uuid.UUID(ids[0]), "content": content}


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
    # Note: SupabaseVectorStore doesn't implement asimilarity_search_with_relevance_scores,
    # so we wrap the synchronous version in asyncio.to_thread to avoid blocking the event loop.
    docs_with_scores = await asyncio.to_thread(
        vector_store.similarity_search_with_relevance_scores,
        query,
        k=top_k,
    )

    # Convert to the expected format
    results: list[Memory] = []
    for doc, score in docs_with_scores:
        if score >= threshold:
            result: Memory = {
                "content": doc.page_content,
                "similarity": score,
            }
            results.append(result)
    logger.debug(
        "Search returned %d results, scores: %s",
        len(results),
        [r.get("similarity") for r in results],
    )
    return results


async def find_duplicates(content: str) -> list[DuplicateMatch]:
    """Check for duplicate memories by exact text match and semantic similarity.

    Performs two checks:
    1. Exact text match via Supabase query (no embedding cost).
    2. Semantic similarity search with a 0.85 threshold (top 3).

    Results are deduplicated — an exact match that also appears in the
    semantic results is only returned once (as "exact").

    Args:
        content: The memory content to check for duplicates.

    Returns:
        A list of duplicate records, each containing 'content',
        'similarity', and 'match_type' ("exact" or "semantic").
    """
    client = get_supabase_client()

    # Step 1: Exact text match (wrap sync client call to avoid blocking the event loop)
    exact_response = await asyncio.to_thread(
        client.table("memories").select("id, content").eq("content", content).execute
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

    # Step 2: Semantic similarity search
    semantic_matches = await search_memories(content, top_k=3, threshold=0.85)
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
