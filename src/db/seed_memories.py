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
    *,
    user_settings_id: str,
) -> dict[str, int]:
    """Import memories from JSON file into the database.

    Args:
        file_path: Path to the JSON file containing memories.
        show_progress: Whether to print progress to stdout.
        user_settings_id: The user_settings UUID to assign to all seeded memories.

    Returns:
        A dictionary with success and failed counts.
    """
    try:
        with open(file_path) as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.error("Seed file not found: %s", file_path)
        return {"success": 0, "failed": 1}
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in seed file: %s", e)
        return {"success": 0, "failed": 1}

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

        succeeded = await _save_with_retry(content, summary, user_settings_id=user_settings_id)

        if succeeded:
            success_count += 1
            if show_progress:
                display_summary = summary[:50] + "..." if len(summary) > 50 else summary
                print(f"[{i + 1}/{len(data)}] ✓ {display_summary}")
        else:
            failed_count += 1
            if show_progress:
                display_summary = summary[:50] + "..." if len(summary) > 50 else summary
                print(f"[{i + 1}/{len(data)}] ✗ Failed: {display_summary}")

        if i < len(data) - 1:
            if show_progress:
                print(f"Waiting 2 seconds before next call... ({i + 1}/{len(data)})")
            await asyncio.sleep(2)

    if show_progress:
        print("\nSummary:")
        print(f"✓ Successfully imported: {success_count}")
        print(f"✗ Failed: {failed_count}")

    return {"success": success_count, "failed": failed_count}


async def _save_with_retry(
    content: str,
    summary: str,
    *,
    user_settings_id: str,
) -> bool:
    """Save a memory with retry logic on failure.

    Args:
        content: The memory content text.
        summary: Summary of the memory.
        user_settings_id: The user_settings UUID.

    Returns:
        True if save succeeded, False if both attempts failed.
    """
    try:
        await save_memory(content, summary, user_settings_id=user_settings_id)
        return True
    except Exception as e:
        logger.warning("Failed to save memory, retrying in 2 minutes: %s", e)
        await asyncio.sleep(120)
        try:
            await save_memory(content, summary, user_settings_id=user_settings_id)
            return True
        except Exception as e:
            logger.error("Failed to save memory after retry: %s", e)
            return False


async def main() -> None:
    """Main entry point for the CLI script."""
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m src.db.seed_memories <user_settings_id> [file_path]")
        sys.exit(1)

    user_settings_id = sys.argv[1]
    file_path = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_SEED_FILE

    result = await seed_memories(file_path, show_progress=True, user_settings_id=user_settings_id)

    if result["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
