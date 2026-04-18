"""Comprehensive multi-tenancy test covering all memory operations.

Tests:
1. Save memory (scoped by user)
2. Search memory (scoped by user)
3. Find duplicates (scoped by user)
4. Select memory for reminders (scoped by user)

Usage:
    uv run python scripts/test_multi_tenancy.py
"""

import asyncio
from datetime import UTC, datetime

from src.db.client import get_supabase_client
from src.db.vector_store import find_duplicates, save_memory, search_memories
from src.jobs.remind import select_memory


async def main() -> None:
    """Run comprehensive multi-tenancy tests."""
    client = get_supabase_client()
    failures = []

    print("=" * 60)
    print("MULTI-TENANCY TEST SUITE")
    print("=" * 60)

    # Setup: Create two test users
    print("\n[SETUP] Creating test users...")
    user1 = (
        client.table("user_settings")
        .upsert(
            {"telegram_chat_id": "test_user_1", "timezone": "UTC"},
            on_conflict="telegram_chat_id",
        )
        .execute()
    )
    user1_id = user1.data[0]["id"]

    user2 = (
        client.table("user_settings")
        .upsert(
            {"telegram_chat_id": "test_user_2", "timezone": "UTC"},
            on_conflict="telegram_chat_id",
        )
        .execute()
    )
    user2_id = user2.data[0]["id"]

    print(f"  User 1 ID: {user1_id}")
    print(f"  User 2 ID: {user2_id}")

    # Test 1: Save Memory (scoped by user)
    print("\n" + "=" * 60)
    print("TEST 1: Save Memory (user isolation)")
    print("=" * 60)

    print("\n[User 1] Saving: 'Consistency beats intensity'")
    await save_memory("Consistency beats intensity", user_settings_id=user1_id)

    print("[User 2] Saving: 'Python is great for data science'")
    await save_memory("Python is great for data science", user_settings_id=user2_id)

    print("✓ Memories saved for both users")

    # Test 2: Search Memory (scoped by user)
    print("\n" + "=" * 60)
    print("TEST 2: Search Memory (user isolation)")
    print("=" * 60)

    print("\n[User 1] Searching for 'consistency':")
    user1_results = await search_memories("consistency", user_settings_id=user1_id)
    print(f"  Found {len(user1_results)} result(s)")
    for r in user1_results:
        print(f"    - {r['content'][:50]}...")

    print("\n[User 1] Searching for 'Python' (should be empty):")
    user1_python = await search_memories("Python", user_settings_id=user1_id, threshold=0.85)
    if len(user1_python) == 0:
        print("  ✓ PASS: No results (correct isolation)")
    else:
        print(f"  ✗ FAIL: Found {len(user1_python)} results (isolation broken!)")
        print(f"  Results: {user1_python}")
        failures.append("User 1 search isolation (Python query)")

    print("\n[User 2] Searching for 'Python':")
    user2_results = await search_memories("Python", user_settings_id=user2_id)
    print(f"  Found {len(user2_results)} result(s)")
    for r in user2_results:
        print(f"    - {r['content'][:50]}...")

    print("\n[User 2] Searching for 'consistency' (should be empty):")
    user2_consistency = await search_memories(
        "consistency", user_settings_id=user2_id, threshold=0.85
    )
    if len(user2_consistency) == 0:
        print("  ✓ PASS: No results (correct isolation)")
    else:
        print(f"  ✗ FAIL: Found {len(user2_consistency)} results (isolation broken!)")
        print(f"  Results: {user2_consistency}")
        failures.append("User 2 search isolation (consistency query)")

    # Test 3: Find Duplicates (scoped by user)
    print("\n" + "=" * 60)
    print("TEST  3: Find Duplicates (user isolation)")
    print("=" * 60)

    print("\n[User 1] Saving duplicate: 'Consistency beats intensity'")
    user1_dupes = await find_duplicates("Consistency beats intensity", user_settings_id=user1_id)
    print(f"  Found {len(user1_dupes)} duplicate(s) for User 1")
    if len(user1_dupes) > 0:
        print("  ✓ PASS: Duplicate detected for same user")

    print("\n[User 2] Checking for 'Consistency beats intensity' (should be empty):")
    user2_dupes = await find_duplicates("Consistency beats intensity", user_settings_id=user2_id)
    if len(user2_dupes) == 0:
        print("  ✓ PASS: No duplicates (correct isolation)")
    else:
        print(f"  ✗ FAIL: Found {len(user2_dupes)} duplicates (isolation broken!)")
        failures.append("User 2 duplicate detection isolation")

    # Test 4: Select Memory for Reminders (scoped by user)
    print("\n" + "=" * 60)
    print("TEST 4: Select Memory for Reminders (user isolation)")
    print("=" * 60)

    print("\n[User 1] Selecting a memory for reminder:")
    user1_memory = select_memory(user_settings_id=user1_id, now=datetime.now(UTC))
    content1 = str(user1_memory["content"])
    print(f"  Selected: {content1[:50]}...")
    if "Consistency" in content1:
        print("  ✓ PASS: Selected User 1's memory")
    else:
        print("  ✗ FAIL: Selected wrong user's memory!")
        failures.append("User 1 memory selection")

    print("\n[User 2] Selecting a memory for reminder:")
    user2_memory = select_memory(user_settings_id=user2_id, now=datetime.now(UTC))
    content2 = str(user2_memory["content"])
    print(f"  Selected: {content2[:50]}...")
    if "Python" in content2:
        print("  ✓ PASS: Selected User 2's memory")
    else:
        print("  ✗ FAIL: Selected wrong user's memory!")
        failures.append("User 2 memory selection")

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    if failures:
        print(f"✗ {len(failures)} test(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        print("=" * 60)
        exit(1)
    else:
        print("✓ All multi-tenancy tests PASSED!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
