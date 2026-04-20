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
        self.buffers: dict[str, list[PendingMessage]] = {}
        self.timers: dict[str, asyncio.Task[None]] = {}
        self.locks: dict[str, asyncio.Lock] = {}

    async def add_message(
        self,
        chat_id: str,
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
            logger.debug("Skipping empty message for chat %s", chat_id)
            return

        # Create lock for this chat if it doesn't exist
        if chat_id not in self.locks:
            self.locks[chat_id] = asyncio.Lock()

        async with self.locks[chat_id]:
            # Initialize buffer if needed
            if chat_id not in self.buffers:
                self.buffers[chat_id] = []

            # Add message to buffer
            self.buffers[chat_id].append(PendingMessage(text=text, update=update, context=context))

            # Cancel existing timer if any
            if chat_id in self.timers:
                self.timers[chat_id].cancel()

            # Start new timer
            self.timers[chat_id] = asyncio.create_task(self._timer_callback(chat_id))

    async def _timer_callback(self, chat_id: str) -> None:
        """Wait for timeout then process batch.

        Args:
            chat_id: Chat identifier for the batch to process.
        """
        await asyncio.sleep(self.timeout)
        await self._process_batch(chat_id)

    async def _process_batch(self, chat_id: str) -> None:
        """Process all buffered messages for a chat.

        Args:
            chat_id: Chat identifier for the batch to process.
        """
        if chat_id not in self.locks:
            logger.warning("No lock found for chat %s during batch processing", chat_id)
            return

        async with self.locks[chat_id]:
            # Get buffered messages
            messages = self.buffers.get(chat_id, [])
            if not messages:
                logger.debug("No messages to process for chat %s", chat_id)
                return

            # Clear buffer and timer
            self.buffers.pop(chat_id, None)
            self.timers.pop(chat_id, None)

            # Combine message texts
            combined_text = "\n\n".join(msg.text for msg in messages)

            # Use first message's update and context
            first_message = messages[0]
            message = first_message.update.message
            if message is None or message.from_user is None:
                logger.warning("Cannot determine user_id for chat %s, skipping batch", chat_id)
                return
            user_id = message.from_user.id

            logger.info(
                "Processing batch for chat %s: %d messages, %d chars",
                chat_id,
                len(messages),
                len(combined_text),
            )

            # Call the processing callback if provided
            if self.process_callback:
                await self.process_callback(
                    chat_id=chat_id,
                    user_id=user_id,
                    combined_text=combined_text,
                    update=first_message.update,
                    context=first_message.context,
                )

    async def flush(self, chat_id: str) -> None:
        """Process buffered messages immediately and cancel timer."""
        timer = self.timers.pop(chat_id, None)
        if timer is not None:
            timer.cancel()
        await self._process_batch(chat_id)

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
