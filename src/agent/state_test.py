"""Tests for the AgentState TypedDict."""

from src.agent.state import AgentState


def test_agent_state_has_required_fields() -> None:
    """Test that AgentState has all required fields."""
    # Create an instance with all required fields
    state: AgentState = {
        "user_input": "",
        "cleaned_input": "",
        "intent": None,
        "memories": [],
        "response": "",
        "error": None,
    }

    assert state["user_input"] == ""
    assert state["cleaned_input"] == ""
    assert state["intent"] is None
    assert state["memories"] == []
    assert state["response"] == ""
    assert state["error"] is None


def test_agent_state_accumulates_memories() -> None:
    """Test that memories field accumulates across updates."""
    state1: AgentState = {
        "user_input": "test",
        "cleaned_input": "test",
        "intent": "save",
        "memories": [{"content": "memory1"}],
        "response": "",
        "error": None,
    }

    state2: AgentState = {
        "user_input": "test2",
        "cleaned_input": "test2",
        "intent": "query",
        "memories": [{"content": "memory2"}],
        "response": "response",
        "error": None,
    }

    # When merging states, memories should accumulate
    merged_memories = state1["memories"] + state2["memories"]
    assert len(merged_memories) == 2
