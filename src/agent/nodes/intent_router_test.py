"""Tests for the intent_router node."""

from src.agent.state import AgentState


async def test_intent_router_detects_save_intent() -> None:
    """Test that intent_router detects save intent and strips prefix."""
    from src.agent.nodes.intent_router import intent_router

    state: AgentState = {
        "user_input": "/save Consistency beats intensity",
        "cleaned_input": "",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
        "messages": [],
    }

    result = await intent_router(state)
    update = result.update or {}

    assert update["intent"] == "save"
    assert update["cleaned_input"] == "Consistency beats intensity"


async def test_intent_router_detects_query_intent() -> None:
    """Test that intent_router detects query intent and strips prefix."""
    from src.agent.nodes.intent_router import intent_router

    state: AgentState = {
        "user_input": "/ask What do I know about habits?",
        "cleaned_input": "",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
        "messages": [],
    }

    result = await intent_router(state)
    update = result.update or {}

    assert update["intent"] == "query"
    assert update["cleaned_input"] == "What do I know about habits?"


async def test_intent_router_detects_query_command() -> None:
    """Test that intent_router detects /query command."""
    from src.agent.nodes.intent_router import intent_router

    state: AgentState = {
        "user_input": "/query What are my habits?",
        "cleaned_input": "",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
        "messages": [],
    }

    result = await intent_router(state)
    update = result.update or {}

    assert update["intent"] == "query"
    assert update["cleaned_input"] == "What are my habits?"


async def test_intent_router_handles_unknown_intent() -> None:
    """Test that intent_router handles unknown intents."""
    from src.agent.nodes.intent_router import intent_router

    state: AgentState = {
        "user_input": "Hello bot!",
        "cleaned_input": "",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
        "messages": [],
    }

    result = await intent_router(state)
    update = result.update or {}

    assert update["intent"] is None
    assert update["error"] is not None


async def test_intent_router_strips_whitespace() -> None:
    """Test that intent_router strips extra whitespace from cleaned_input."""
    from src.agent.nodes.intent_router import intent_router

    state: AgentState = {
        "user_input": "  /save   Remember this  ",
        "cleaned_input": "",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
        "messages": [],
    }

    result = await intent_router(state)
    update = result.update or {}

    assert update["intent"] == "save"
    assert update["cleaned_input"] == "Remember this"
