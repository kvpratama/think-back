"""End-to-end multi-tenancy test covering bot → agent → tools → DB.

Usage:
    uv run python scripts/test_multi_tenancy_e2e.py
"""

import asyncio
from typing import Any, cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command

from src.agent.graph import build_graph
from src.db.client import get_supabase_client


async def main() -> None:
    """Test full bot → agent → tools → DB flow for two users."""
    from langgraph.checkpoint.memory import InMemorySaver

    client = get_supabase_client()
    graph = build_graph(checkpointer=InMemorySaver())
    failures = []

    print("=" * 60)
    print("END-TO-END MULTI-TENANCY TEST")
    print("=" * 60)

    # Setup: Create two test users
    print("\n[SETUP] Creating test users...")
    user1 = (
        client.table("user_settings")
        .upsert(
            {"telegram_chat_id": "e2e_user_1", "timezone": "UTC"},
            on_conflict="telegram_chat_id",
        )
        .execute()
    )
    user1_id = cast(list[dict[str, Any]], user1.data)[0]["id"]

    user2 = (
        client.table("user_settings")
        .upsert(
            {"telegram_chat_id": "e2e_user_2", "timezone": "UTC"},
            on_conflict="telegram_chat_id",
        )
        .execute()
    )
    user2_id = cast(list[dict[str, Any]], user2.data)[0]["id"]

    print(f"  User 1 ID: {user1_id}")
    print(f"  User 2 ID: {user2_id}")

    # Test 1: User 1 saves a memory via agent
    print("\n" + "=" * 60)
    print("TEST 1: User 1 saves memory via agent")
    print("=" * 60)

    config1: RunnableConfig = {
        "configurable": {
            "thread_id": "e2e_thread_1",
            "user_settings_id": user1_id,
        }
    }

    # Stream to handle interrupt
    async for _chunk in graph.astream(
        {"messages": [HumanMessage(content="Remember: Consistency beats intensity")]},
        config=config1,
        stream_mode="updates",
    ):
        pass  # Process until interrupt

    # Resume with approval
    result1 = await graph.ainvoke(
        Command(resume={"approved": True}),
        config=config1,
    )
    print(f"  Agent response: {result1['messages'][-1].content[:100]}...")

    # Verify memory was saved to DB
    user1_memories = client.table("memories").select("*").eq("user_settings_id", user1_id).execute()
    user1_rows = cast(list[dict[str, Any]], user1_memories.data)
    if len(user1_rows) > 0 and "Consistency" in user1_rows[0]["content"]:
        print("  ✓ Memory saved to database")
    else:
        print("  ✗ FAIL: Memory not found in database!")
        failures.append("Test 1: User 1 memory save")

    # Test 2: User 2 saves a different memory
    print("\n" + "=" * 60)
    print("TEST 2: User 2 saves memory via agent")
    print("=" * 60)

    config2: RunnableConfig = {
        "configurable": {
            "thread_id": "e2e_thread_2",
            "user_settings_id": user2_id,
        }
    }

    # Stream to handle interrupt
    async for _chunk in graph.astream(
        {"messages": [HumanMessage(content="Remember: Python is great for data science")]},
        config=config2,
        stream_mode="updates",
    ):
        pass  # Process until interrupt

    # Resume with approval
    result2 = await graph.ainvoke(
        Command(resume={"approved": True}),
        config=config2,
    )
    print(f"  Agent response: {result2['messages'][-1].content[:100]}...")

    # Verify memory was saved to DB
    user2_memories = client.table("memories").select("*").eq("user_settings_id", user2_id).execute()
    user2_rows = cast(list[dict[str, Any]], user2_memories.data)
    if len(user2_rows) > 0 and "Python" in user2_rows[0]["content"]:
        print("  ✓ Memory saved to database")
    else:
        print("  ✗ FAIL: Memory not found in database!")
        failures.append("Test 2: User 2 memory save")

    # Test 3: User 1 searches (should only see their memory)
    print("\n" + "=" * 60)
    print("TEST 3: User 1 searches for 'consistency'")
    print("=" * 60)

    result3 = await graph.ainvoke(
        {"messages": [HumanMessage(content="What do I know about consistency?")]},
        config=config1,
    )
    response3 = result3["messages"][-1].content
    print(f"  Agent response: {response3[:200]}...")

    if "Consistency beats intensity" in response3 and "Python" not in response3:
        print("  ✓ PASS: User 1 sees only their memory")
    else:
        print("  ✗ FAIL: Cross-user leakage detected!")
        failures.append("Test 3: User 1 search isolation")

    # Test 4: User 2 searches (should only see their memory)
    print("\n" + "=" * 60)
    print("TEST 4: User 2 searches for 'Python'")
    print("=" * 60)

    result4 = await graph.ainvoke(
        {"messages": [HumanMessage(content="What do I know about Python?")]},
        config=config2,
    )
    response4 = result4["messages"][-1].content
    print(f"  Agent response: {response4[:200]}...")

    if "Python is great" in response4 and "Consistency" not in response4:
        print("  ✓ PASS: User 2 sees only their memory")
    else:
        print("  ✗ FAIL: Cross-user leakage detected!")
        failures.append("Test 4: User 2 search isolation")

    # Test 5: User 1 searches for User 2's content (should find nothing)
    print("\n" + "=" * 60)
    print("TEST 5: User 1 searches for 'Python' (should find nothing)")
    print("=" * 60)

    result5 = await graph.ainvoke(
        {"messages": [HumanMessage(content="What do I know about Python?")]},
        config=config1,
    )
    response5 = result5["messages"][-1].content
    print(f"  Agent response: {response5[:200]}...")

    # Normalize apostrophes (LLM may use smart quotes)
    response5_normalized = response5.lower().replace("'", "'")

    if (
        "Python" not in response5
        or "don't have" in response5_normalized
        or "no information" in response5_normalized
        or "not finding" in response5_normalized
    ):
        print("  ✓ PASS: User 1 cannot see User 2's memory")
    else:
        print("  ✗ FAIL: User 1 can see User 2's memory!")
        failures.append("Test 5: User 1 cannot access User 2's memory")

    # Test 6: User 2 searches for User 1's content (should find nothing)
    print("\n" + "=" * 60)
    print("TEST 6: User 2 searches for 'consistency' (should find nothing)")
    print("=" * 60)

    result6 = await graph.ainvoke(
        {"messages": [HumanMessage(content="What do I know about consistency?")]},
        config=config2,
    )
    response6 = result6["messages"][-1].content
    print(f"  Agent response: {response6[:200]}...")

    # Normalize apostrophes (LLM may use smart quotes)
    response6_normalized = response6.lower().replace("'", "'")

    if (
        "Consistency" not in response6
        or "don't have" in response6_normalized
        or "no information" in response6_normalized
        or "not finding" in response6_normalized
    ):
        print("  ✓ PASS: User 2 cannot see User 1's memory")
    else:
        print("  ✗ FAIL: User 2 can see User 1's memory!")
        failures.append("Test 6: User 2 cannot access User 1's memory")

    # Final contamination check
    print("\n" + "=" * 60)
    print("FINAL: Database contamination check")
    print("=" * 60)

    user1_all = (
        client.table("memories").select("content").eq("user_settings_id", user1_id).execute()
    )
    user2_all = (
        client.table("memories").select("content").eq("user_settings_id", user2_id).execute()
    )

    user1_contents = [m["content"] for m in cast(list[dict[str, Any]], user1_all.data)]
    user2_contents = [m["content"] for m in cast(list[dict[str, Any]], user2_all.data)]

    if any("Python" in c for c in user1_contents):
        print("  ✗ FAIL: User 1 has User 2's memory!")
        failures.append("Database contamination: User 1 has User 2's data")
    if any("Consistency" in c for c in user2_contents):
        print("  ✗ FAIL: User 2 has User 1's memory!")
        failures.append("Database contamination: User 2 has User 1's data")

    if not any("Python" in c for c in user1_contents) and not any(
        "Consistency" in c for c in user2_contents
    ):
        print("  ✓ PASS: No cross-user contamination detected")

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    if failures:
        print(f"✗ {len(failures)} test(s) FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        print("=" * 60)
        raise SystemExit(1)
    else:
        print("✓ All tests passed!")
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
