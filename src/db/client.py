"""Supabase database client singleton.

All Supabase client access goes through this module.
Never instantiate the Supabase client inline.
"""

from functools import lru_cache

from supabase import Client, create_client


@lru_cache
def get_supabase_client() -> Client:
    """Get or create the Supabase client instance.

    Returns a singleton client instance using LRU cache.
    The client is initialized using settings from environment variables.

    Returns:
        Client: The Supabase client instance.

    Example:
        >>> client = get_supabase_client()
        >>> client.table("memories").select("*").execute()
    """
    from src.core.config import get_settings

    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)
