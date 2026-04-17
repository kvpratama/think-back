"""Script to seed memories from JSON file into the database.

This script imports memories from a JSON file into the Supabase database
with rate limiting and retry logic to handle API limits.
"""

import asyncio
import json
import logging
import sys

from src.db.vector_store import save_memory

logger = logging.getLogger(__name__)

DEFAULT_SEED_FILE = "src/db/seed.json"


async def seed_memories(
    file_path: str = DEFAULT_SEED_FILE,
    show_progress: bool = False,
) -> dict[str, int]:
    """Import memories from JSON file into the database.

    Reads a JSON file containing memory entries and imports them into the database
    using the save_memory function. Includes rate limiting (2 second delay between
    calls) and retry logic (waits 2 minutes and retries once on error).

    Args:
        file_path: Path to the JSON file containing memories. Each entry should
            have 'content' and 'summary' fields. Defaults to DEFAULT_SEED_FILE.
        show_progress: Whether to print progress to stdout. Defaults to False.

    Returns:
        A dictionary containing:
            - success: Number of successfully imported memories
            - failed: Number of memories that failed to import after retry

    Example:
        >>> result = await seed_memories("memories.json")
        >>> print(f"Imported {result['success']} memories")
        Imported 120 memories
    """
    with open(file_path) as f:
        data = json.load(f)

    if show_progress:
        print(f"Loading memories from {file_path}...")
        print(f"Found {len(data)} entries to import\n")

    success_count = 0
    failed_count = 0

    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "content" not in entry:
            failed_count += 1
            if show_progress:
                print(f"[{i + 1}/{len(data)}] ✗ Invalid entry (missing 'content')")
            continue
        content = entry["content"]
        summary = entry.get("summary") or content

        # Try to save memory with retry logic
        succeeded = await _save_with_retry(content, summary)

        if succeeded:
            success_count += 1
            if show_progress:
                # Truncate summary for display
                display_summary = summary[:50] + "..." if len(summary) > 50 else summary
                print(f"[{i + 1}/{len(data)}] ✓ {display_summary}")
        else:
            failed_count += 1
            if show_progress:
                display_summary = summary[:50] + "..." if len(summary) > 50 else summary
                print(f"[{i + 1}/{len(data)}] ✗ Failed: {display_summary}")

        # Rate limiting: wait 2 seconds between calls (except after last entry)
        if i < len(data) - 1:
            if show_progress:
                print(f"Waiting 2 seconds before next call... ({i + 1}/{len(data)})")
            await asyncio.sleep(2)

    if show_progress:
        print("\nSummary:")
        print(f"✓ Successfully imported: {success_count}")
        print(f"✗ Failed: {failed_count}")

    return {"success": success_count, "failed": failed_count}


async def _save_with_retry(content: str, summary: str) -> bool:
    """Save a memory with retry logic on failure.

    Attempts to save a memory. If it fails, waits 2 minutes and retries once.

    Args:
        content: The memory content text.
        summary: Summary of the memory.

    Returns:
        True if save succeeded (either first attempt or retry), False if both failed.
    """
    try:
        await save_memory(content, summary)
        return True
    except Exception as e:
        logger.warning("Failed to save memory, retrying in 2 minutes: %s", e)
        await asyncio.sleep(120)
        try:
            await save_memory(content, summary)
            return True
        except Exception as e:
            logger.error("Failed to save memory after retry: %s", e)
            return False


async def main() -> None:
    """Main entry point for the CLI script."""
    logging.basicConfig(level=logging.INFO)

    file_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SEED_FILE

    result = await seed_memories(file_path, show_progress=True)

    # Exit with error code if any imports failed
    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
