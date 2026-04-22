"""Tests for turn-based message trim middleware."""

from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.agent.middleware import _trim_messages_by_turns_impl, split_into_turns


def test_split_single_turn() -> None:
    """A single user-assistant exchange is one turn."""
    msgs: list[BaseMessage] = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
    ]
    turns = split_into_turns(msgs)
    assert len(turns) == 1
    assert len(turns[0]) == 2


def test_split_multiple_turns() -> None:
    """Each HumanMessage starts a new turn."""
    msgs: list[BaseMessage] = [
        HumanMessage(content="q1"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
        AIMessage(content="a2"),
        HumanMessage(content="q3"),
        AIMessage(content="a3"),
    ]
    turns = split_into_turns(msgs)
    assert len(turns) == 3


def test_split_turn_with_tool_calls() -> None:
    """Tool call chains stay within the same turn."""
    msgs: list[BaseMessage] = [
        HumanMessage(content="search something"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "1"}]),
        ToolMessage(content="result", tool_call_id="1"),
        AIMessage(content="here's what I found"),
    ]
    turns = split_into_turns(msgs)
    assert len(turns) == 1
    assert len(turns[0]) == 4


def test_split_orphan_ai_messages() -> None:
    """Standalone AIMessages before the first HumanMessage are grouped with the first turn."""
    msgs: list[BaseMessage] = [
        AIMessage(content="reminder"),
        HumanMessage(content="thanks"),
        AIMessage(content="you're welcome"),
    ]
    turns = split_into_turns(msgs)
    assert len(turns) == 1
    assert len(turns[0]) == 3


def test_split_empty_list() -> None:
    """Empty message list returns empty turns."""
    turns = split_into_turns([])
    assert turns == []


def test_split_excludes_system_messages() -> None:
    """SystemMessages should not be passed to split_into_turns.

    This test documents that if they are, they don't start turns
    (they'd be grouped with the next turn).
    """
    msgs: list[BaseMessage] = [
        SystemMessage(content="system"),
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
    ]
    turns = split_into_turns(msgs)
    # SystemMessage groups with the first turn
    assert len(turns) == 1
    assert len(turns[0]) == 3


def _make_turn(user_content: str, ai_content: str) -> list:
    """Helper to create a simple user-assistant turn."""
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
    """Trims to last max_turns turns, preserving system message."""
    msgs = [
        SystemMessage(content="system prompt"),
        *_make_turn("q1", "a1"),
        *_make_turn("q2", "a2"),
        *_make_turn("q3", "a3"),
        *_make_turn("q4", "a4"),
    ]
    state = {"messages": msgs}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_turns=2)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    assert result is not None
    trimmed = result["messages"]
    # System message + 2 turns (4 messages) = 5
    assert len(trimmed) == 5
    assert isinstance(trimmed[0], SystemMessage)
    assert trimmed[1].content == "q3"
    assert trimmed[3].content == "q4"


def test_trim_preserves_tool_chains() -> None:
    """Tool call chains within a turn are kept intact."""
    msgs = [
        SystemMessage(content="system prompt"),
        *_make_turn("old", "old-answer"),
        HumanMessage(content="search something"),
        AIMessage(content="", tool_calls=[{"name": "search", "args": {"q": "x"}, "id": "1"}]),
        ToolMessage(content="result", tool_call_id="1"),
        AIMessage(content="here's what I found"),
    ]
    state = {"messages": msgs}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_turns=1)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    assert result is not None
    trimmed = result["messages"]
    # System + 1 turn with tool chain (4 msgs) = 5
    assert len(trimmed) == 5
    assert isinstance(trimmed[0], SystemMessage)
    assert trimmed[1].content == "search something"


def test_trim_no_system_message() -> None:
    """Works correctly when there is no system message."""
    msgs = [
        *_make_turn("q1", "a1"),
        *_make_turn("q2", "a2"),
        *_make_turn("q3", "a3"),
    ]
    state = {"messages": msgs}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_turns=1)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    assert result is not None
    trimmed = result["messages"]
    assert len(trimmed) == 2
    assert trimmed[0].content == "q3"


def test_trim_empty_messages() -> None:
    """Returns None for empty message list."""
    state = {"messages": []}
    with patch("src.agent.middleware.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock(max_turns=5)
        result = _trim_messages_by_turns_impl(state, MagicMock())
    assert result is None
