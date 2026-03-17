"""Tests for the intent_router node."""

import pytest

from src.agent.state import AgentState


@pytest.mark.asyncio
async def test_intent_router_detects_save_intent() -> None:
    """Test that intent_router detects save intent."""
    from src.agent.nodes.intent_router import intent_router

    state: AgentState = {
        "user_input": "/save Consistency beats intensity",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
    }

    result = await intent_router(state)

    assert result["intent"] == "save"


@pytest.mark.asyncio
async def test_intent_router_detects_query_intent() -> None:
    """Test that intent_router detects query intent."""
    from src.agent.nodes.intent_router import intent_router

    state: AgentState = {
        "user_input": "/ask What do I know about habits?",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
    }

    result = await intent_router(state)

    assert result["intent"] == "query"


@pytest.mark.asyncio
async def test_intent_router_handles_unknown_intent() -> None:
    """Test that intent_router handles unknown intents."""
    from src.agent.nodes.intent_router import intent_router

    state: AgentState = {
        "user_input": "Hello bot!",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
    }

    result = await intent_router(state)

    assert result["intent"] is None
    assert result["error"] is not None
