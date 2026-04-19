"""Tests for MessageBatcher."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.bot_batcher import MessageBatcher


@pytest.fixture
def mock_update():
    """Create a mock Telegram Update."""
    update = MagicMock()
    update.message.text = "test message"
    update.message.chat.id = 123
    return update


@pytest.fixture
def mock_context():
    """Create a mock Telegram context."""
    return MagicMock()


@pytest.mark.asyncio
async def test_add_message_creates_buffer(mock_update, mock_context):
    """Test that adding a message creates a buffer for the chat."""
    batcher = MessageBatcher(timeout=1.0)

    await batcher.add_message(
        chat_id="123",
        text="Hello",
        update=mock_update,
        context=mock_context,
    )

    assert "123" in batcher.buffers
    assert len(batcher.buffers["123"]) == 1
    assert batcher.buffers["123"][0].text == "Hello"


@pytest.mark.asyncio
async def test_add_message_starts_timer(mock_update, mock_context):
    """Test that adding a message starts a timer."""
    batcher = MessageBatcher(timeout=1.0)

    await batcher.add_message(
        chat_id="123",
        text="Hello",
        update=mock_update,
        context=mock_context,
    )

    assert "123" in batcher.timers
    assert not batcher.timers["123"].done()


@pytest.mark.asyncio
async def test_process_batch_combines_messages(mock_update, mock_context):
    """Test that batch processing combines multiple messages."""
    import asyncio

    process_callback = AsyncMock()
    batcher = MessageBatcher(timeout=0.1, process_callback=process_callback)

    # Add multiple messages
    await batcher.add_message("123", "First", mock_update, mock_context)
    await batcher.add_message("123", "Second", mock_update, mock_context)
    await batcher.add_message("123", "Third", mock_update, mock_context)

    # Wait for timer to expire
    await asyncio.sleep(0.15)

    # Verify callback was called with combined text
    process_callback.assert_called_once()
    call_args = process_callback.call_args[1]
    assert call_args["chat_id"] == "123"
    assert call_args["combined_text"] == "First\n\nSecond\n\nThird"
    assert call_args["update"] == mock_update
    assert call_args["context"] == mock_context


@pytest.mark.asyncio
async def test_shutdown_cancels_timers(mock_update, mock_context):
    """Test that shutdown cancels all active timers."""

    process_callback = AsyncMock()
    batcher = MessageBatcher(timeout=10.0, process_callback=process_callback)

    # Add messages to multiple chats
    await batcher.add_message("123", "Message 1", mock_update, mock_context)
    await batcher.add_message("456", "Message 2", mock_update, mock_context)

    assert len(batcher.timers) == 2

    # Shutdown
    await batcher.shutdown()

    # Verify timers cancelled and buffers cleared
    assert len(batcher.timers) == 0
    assert len(batcher.buffers) == 0

    # Verify callback was not called (messages discarded)
    process_callback.assert_not_called()


@pytest.mark.asyncio
async def test_timer_reset_on_new_message(mock_update, mock_context):
    """Test that new messages reset the timer."""
    import asyncio

    process_callback = AsyncMock()
    batcher = MessageBatcher(timeout=0.1, process_callback=process_callback)

    # Add first message
    await batcher.add_message("123", "First", mock_update, mock_context)
    first_timer = batcher.timers["123"]

    # Wait 0.05s (half the timeout)
    await asyncio.sleep(0.05)

    # Add second message (should reset timer)
    await batcher.add_message("123", "Second", mock_update, mock_context)
    second_timer = batcher.timers["123"]

    # Verify timer was replaced
    assert first_timer != second_timer

    # Wait another 0.08s (total 0.13s from first, but only 0.08s from second)
    await asyncio.sleep(0.08)

    # Callback should not have been called yet
    process_callback.assert_not_called()

    # Wait for second timer to complete
    await asyncio.sleep(0.05)

    # Now callback should be called with both messages
    process_callback.assert_called_once()
    call_args = process_callback.call_args[1]
    assert call_args["combined_text"] == "First\n\nSecond"


@pytest.mark.asyncio
async def test_concurrent_chats_independent(mock_context):
    """Test that different chats have independent buffers and timers."""
    import asyncio

    process_callback = AsyncMock()
    batcher = MessageBatcher(timeout=0.1, process_callback=process_callback)

    # Create separate updates for different chats
    update_123 = MagicMock()
    update_123.message.text = "Chat 123 message"
    update_123.message.chat.id = 123
    update_123.message.from_user.id = 1

    update_456 = MagicMock()
    update_456.message.text = "Chat 456 message"
    update_456.message.chat.id = 456
    update_456.message.from_user.id = 2

    # Add messages to different chats
    await batcher.add_message("123", "Message A", update_123, mock_context)
    await batcher.add_message("456", "Message B", update_456, mock_context)

    # Verify independent buffers
    assert len(batcher.buffers) == 2
    assert len(batcher.buffers["123"]) == 1
    assert len(batcher.buffers["456"]) == 1

    # Verify independent timers
    assert len(batcher.timers) == 2
    assert "123" in batcher.timers
    assert "456" in batcher.timers

    # Wait for both timers to complete
    await asyncio.sleep(0.15)

    # Verify both batches processed independently
    assert process_callback.call_count == 2

    # Verify correct chat_ids
    call_chat_ids = {call[1]["chat_id"] for call in process_callback.call_args_list}
    assert call_chat_ids == {"123", "456"}


@pytest.mark.asyncio
async def test_empty_messages_filtered(mock_update, mock_context):
    """Test that empty messages are not added to buffer."""
    import asyncio

    process_callback = AsyncMock()
    batcher = MessageBatcher(timeout=0.1, process_callback=process_callback)

    # Try to add empty messages
    await batcher.add_message("123", "", mock_update, mock_context)
    await batcher.add_message("123", "   ", mock_update, mock_context)

    # Verify no buffer created
    assert "123" not in batcher.buffers
    assert "123" not in batcher.timers

    # Wait to ensure no processing happens
    await asyncio.sleep(0.15)
    process_callback.assert_not_called()
