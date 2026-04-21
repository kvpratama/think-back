"""Tests for bot_graph module."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.mark.asyncio
async def test_aget_graph_builds_and_caches() -> None:
    """Test that aget_graph builds graph on first call and caches it."""
    from src.api.bot_graph import aget_graph

    mock_context = MagicMock()
    mock_context.bot_data = {}

    mock_checkpointer = MagicMock()
    mock_graph = MagicMock()

    with (
        patch("src.db.checkpointer.aget_checkpointer", return_value=mock_checkpointer) as mock_get,
        patch("src.agent.graph.build_graph", return_value=mock_graph) as mock_build,
    ):
        result = await aget_graph(mock_context)

    mock_get.assert_awaited_once()
    mock_build.assert_called_once_with(checkpointer=mock_checkpointer)
    assert result is mock_graph
    assert mock_context.bot_data["graph"] is mock_graph


@pytest.mark.asyncio
async def test_aget_graph_returns_cached() -> None:
    """Test that aget_graph returns cached instance without calling aget_checkpointer."""
    from src.api.bot_graph import aget_graph

    mock_graph = MagicMock()
    mock_context = MagicMock()
    mock_context.bot_data = {"graph": mock_graph}

    with (
        patch("src.db.checkpointer.aget_checkpointer") as mock_get,
        patch("src.agent.graph.build_graph") as mock_build,
    ):
        result = await aget_graph(mock_context)

    mock_get.assert_not_awaited()
    mock_build.assert_not_called()
    assert result is mock_graph
