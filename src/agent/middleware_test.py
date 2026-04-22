"""Tests for turn-based message trim middleware."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from src.agent.middleware import _trim_messages_by_turns_impl


def _make_turn(user_content: str, ai_content: str, turn_id: str = "") -> list:
    """Helper to create a simple user-assistant turn."""
    if turn_id:
        return [
            HumanMessage(content=user_content, id=f"u_{turn_id}"),
            AIMessage(content=ai_content, id=f"a_{turn_id}"),
        ]
    return [HumanMessage(content=user_content), AIMessage(content=ai_content)]


def test_trim_no_op_when_under_limit() -> None:
    """Returns None when turns <= max_turns (no modification needed)."""
    msgs = [
        SystemMessage(content="system prompt"),
        *_make_turn("q1", "a1"),
        *_make_turn("q2", "a2"),
    ]
    state = {"messages": msgs}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_turns=5)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    assert result is None


def test_trim_keeps_last_n_turns() -> None:
    """Trims to last max_turns turns, returning RemoveMessage objects for dropped turns."""
    msgs = [
        SystemMessage(content="system prompt", id="sys_1"),
        *_make_turn("q1", "a1", "1"),
        *_make_turn("q2", "a2", "2"),
        *_make_turn("q3", "a3", "3"),
        *_make_turn("q4", "a4", "4"),
    ]
    state = {"messages": msgs}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_turns=2)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    assert result is not None
    trimmed = result["messages"]
    # Should drop turns 1 and 2 (4 messages total)
    assert len(trimmed) == 4
    assert all(isinstance(m, RemoveMessage) for m in trimmed)
    assert trimmed[0].id == "u_1"
    assert trimmed[1].id == "a_1"
    assert trimmed[2].id == "u_2"
    assert trimmed[3].id == "a_2"


def test_trim_preserves_tool_chains() -> None:
    """Tool call chains within a turn are kept intact."""
    msgs = [
        SystemMessage(content="system prompt", id="sys_1"),
        *_make_turn("old", "old-answer", "1"),
        HumanMessage(content="search something", id="u_2"),
        AIMessage(
            content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "1"}], id="a_2"
        ),
        ToolMessage(content="result", tool_call_id="1", id="t_1"),
        AIMessage(content="here's what I found", id="a_3"),
    ]
    state = {"messages": msgs}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_turns=1)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    assert result is not None
    trimmed = result["messages"]
    # Should drop the first turn (2 messages)
    assert len(trimmed) == 2
    assert all(isinstance(m, RemoveMessage) for m in trimmed)
    assert trimmed[0].id == "u_1"
    assert trimmed[1].id == "a_1"


def test_trim_no_system_message() -> None:
    """Works correctly when there is no system message."""
    msgs = [
        *_make_turn("q1", "a1", "1"),
        *_make_turn("q2", "a2", "2"),
        *_make_turn("q3", "a3", "3"),
    ]
    state = {"messages": msgs}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_turns=1)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    assert result is not None
    trimmed = result["messages"]
    # Should drop turns 1 and 2 (4 messages)
    assert len(trimmed) == 4
    assert all(isinstance(m, RemoveMessage) for m in trimmed)
    assert trimmed[0].id == "u_1"
    assert trimmed[1].id == "a_1"
    assert trimmed[2].id == "u_2"
    assert trimmed[3].id == "a_2"


def test_trim_empty_messages() -> None:
    """Returns None for empty message list."""
    state = {"messages": []}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_turns=5)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    assert result is None


def test_turn_counted_by_human_messages() -> None:
    """A turn is counted by HumanMessage — AI/Tool messages don't increment the count."""
    msgs = [
        SystemMessage(content="system prompt", id="sys_1"),
        HumanMessage(content="q1", id="u_1"),
        AIMessage(content="", tool_calls=[{"name": "t", "args": {}, "id": "c1"}], id="a_1"),
        ToolMessage(content="result", tool_call_id="c1", id="t_1"),
        AIMessage(content="final", id="a_2"),
        HumanMessage(content="q2", id="u_2"),
        AIMessage(content="reply", id="a_3"),
    ]
    state = {"messages": msgs}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        # max_turns=2 means keep 2 HumanMessages worth of turns
        mock_settings.return_value = MagicMock(max_turns=2)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    # 2 HumanMessages == 2 turns, so nothing should be trimmed
    assert result is None


def test_trim_no_op_at_exact_limit() -> None:
    """Returns None when turn count exactly equals max_turns."""
    msgs = [
        SystemMessage(content="system prompt", id="sys_1"),
        *_make_turn("q1", "a1", "1"),
        *_make_turn("q2", "a2", "2"),
        *_make_turn("q3", "a3", "3"),
    ]
    state = {"messages": msgs}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_turns=3)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    assert result is None
