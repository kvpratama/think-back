"""Message batching for Telegram bot.

This module provides MessageBatcher to combine rapid successive messages
before processing, preventing context loss from Telegram's auto-split behavior.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from telegram import Update
    from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


@dataclass
class PendingMessage:
    """A message waiting to be batched."""

    text: str
    update: Update
    context: ContextTypes.DEFAULT_TYPE


class MessageBatcher:
    """Batches rapid successive messages before processing.

    Messages arriving within the timeout window are combined and processed
    together. Each chat has an independent buffer and timer.
    """

    def __init__(
        self,
        timeout: float = 1.0,
        process_callback: Callable[..., Awaitable[None]] | None = None,
    ):
        """Initialize the message batcher.

        Args:
            timeout: Seconds to wait for additional messages before processing.
            process_callback: Async function to call when batch is ready.
                Receives: chat_id, user_id, combined_text, update, context.
        """
        self.timeout = timeout
        self.process_callback = process_callback
        self.buffers: dict[tuple[str, int], list[PendingMessage]] = {}
        self.timers: dict[tuple[str, int], asyncio.Task[None]] = {}
        self.locks: dict[tuple[str, int], asyncio.Lock] = {}

    async def add_message(
        self,
        chat_id: str,
        user_id: int,
        text: str,
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Add a message to the batch buffer and start/reset timer.

        Args:
            chat_id: Unique identifier for the chat.
            text: Message text content.
            update: Telegram Update object.
            context: Telegram context object.
        """
        # Skip empty messages
        if not text or not text.strip():
            logger.debug("Skipping empty message for chat %s (user %d)", chat_id, user_id)
            return

        key = (chat_id, user_id)

        # Create lock for this key if it doesn't exist
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()

        async with self.locks[key]:
            # Initialize buffer if needed
            if key not in self.buffers:
                self.buffers[key] = []

            # Add message to buffer
            self.buffers[key].append(PendingMessage(text=text, update=update, context=context))

            # Cancel existing timer if any
            if key in self.timers:
                self.timers[key].cancel()

            # Start new timer
            self.timers[key] = asyncio.create_task(self._timer_callback(key))

    async def _timer_callback(self, key: tuple[str, int]) -> None:
        """Wait for timeout then process batch.

        Args:
            key: Composite key (chat_id, user_id) for the batch to process.
        """
        await asyncio.sleep(self.timeout)
        await self._process_batch(key)

    async def _process_batch(self, key: tuple[str, int]) -> None:
        """Process all buffered messages for a specific user in a chat.

        Args:
            key: Composite key (chat_id, user_id) for the batch to process.
        """
        if key not in self.locks:
            logger.warning("No lock found for key %s during batch processing", key)
            return

        chat_id, user_id = key

        async with self.locks[key]:
            # Get buffered messages
            messages = self.buffers.get(key, [])
            if not messages:
                logger.debug("No messages to process for chat %s (user %d)", chat_id, user_id)
                return

            # Clear buffer and timer
            self.buffers.pop(key, None)
            self.timers.pop(key, None)

            # Combine message texts
            combined_text = "\n\n".join(msg.text for msg in messages)

            # Use first message's update and context
            first_message = messages[0]
            update = first_message.update
            context = first_message.context

        logger.info(
            "Processing batch for chat %s (user %d): %d messages, %d chars",
            chat_id,
            user_id,
            len(messages),
            len(combined_text),
        )

        # Call the processing callback outside the lock
        if self.process_callback:
            await self.process_callback(
                chat_id=chat_id,
                user_id=user_id,
                combined_text=combined_text,
                update=update,
                context=context,
            )

    async def flush(self, chat_id: str, user_id: int) -> None:
        """Process buffered messages immediately and cancel timer."""
        key = (chat_id, user_id)
        timer = self.timers.pop(key, None)
        if timer is not None:
            timer.cancel()
        await self._process_batch(key)

    async def shutdown(self) -> None:
        """Cancel all timers and clear buffers on bot shutdown."""
        logger.info("Shutting down MessageBatcher, cancelling %d timers", len(self.timers))

        # Cancel all timers and await them to avoid leaving tasks un-awaited.
        timers = list(self.timers.values())
        for timer in timers:
            timer.cancel()
        await asyncio.gather(*timers, return_exceptions=True)

        # Clear all state
        self.timers.clear()
        self.buffers.clear()
        self.locks.clear()
