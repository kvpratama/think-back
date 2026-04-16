"""Spaced repetition reminder job.

Surfaces saved memories via Telegram with AI-generated insights
and reflective questions. Designed to run as a cron job.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel
from telegram import Bot
from telegram.constants import ParseMode

from src.db.client import get_supabase_client

logger = logging.getLogger(__name__)

INSIGHT_SYSTEM_PROMPT = """You are ThinkBack, a reflective companion. \
Given a saved memory (a quote or highlight), do two things:

1. Insight: Rephrase the core lesson in your own words (1-2 sentences).
2. Question: Ask one reflective question — a call to action, \
self-reflection, or application prompt (1 sentence).

Keep it warm, concise, and grounded in the original quote."""


def get_due_users(now: datetime | None = None) -> list[str]:
    """Find users whose reminder time matches the current hour.

    Queries user_settings for users with reminders_enabled=True whose
    reminder_time_1 or reminder_time_2 falls within the current hour,
    accounting for each user's timezone.

    Args:
        now: Override for the current time (for testing). Defaults to UTC now.

    Returns:
        A list of telegram_chat_id strings for users due for a reminder.
    """
    if now is None:
        now = datetime.now(UTC)

    client = get_supabase_client()
    response = (
        client.table("user_settings")
        .select("telegram_chat_id, reminder_time_1, reminder_time_2, timezone, reminders_enabled")
        .eq("reminders_enabled", True)
        .execute()
    )

    due_users: list[str] = []
    for row in response.data:
        user_tz = ZoneInfo(row["timezone"])
        user_now = now.astimezone(user_tz)
        current_hour = user_now.hour

        for time_field in ("reminder_time_1", "reminder_time_2"):
            raw = row[time_field]
            # Supabase returns time as "HH:MM:SS" string
            reminder_hour = int(raw.split(":")[0])
            if current_hour == reminder_hour:
                due_users.append(row["telegram_chat_id"])
                break  # Don't add same user twice if both times match

    return due_users


def select_memory(now: datetime | None = None) -> dict[str, str | int | None]:
    """Select a memory for review using weighted random selection.

    Queries all memories and applies a weighting formula:
    - Novel (never reviewed): weight = days since created
    - Reviewed: weight = days since last review / review count

    This creates a natural spaced repetition curve — neglected and novel
    memories surface more often; frequently-reviewed ones fade.

    Args:
        now: Override for the current time (for testing). Defaults to UTC now.

    Returns:
        A single memory dict with id, content, source, created_at,
        last_reviewed_at, and review_count.

    Raises:
        RuntimeError: If no memories exist.
    """
    if now is None:
        now = datetime.now(UTC)

    client = get_supabase_client()
    response = (
        client.table("memories")
        .select("id, content, source, created_at, last_reviewed_at, review_count")
        .execute()
    )

    candidates = response.data
    if not candidates:
        raise RuntimeError("No memories available for review.")

    weights: list[float] = []
    for mem in candidates:
        created_at = datetime.fromisoformat(mem["created_at"])
        days_since_created = (now - created_at).total_seconds() / 86400

        if mem["last_reviewed_at"] is None:
            # Novel memory — weight by age since creation
            weight = max(days_since_created, 0.01)
        else:
            last_reviewed = datetime.fromisoformat(mem["last_reviewed_at"])
            days_since_reviewed = (now - last_reviewed).total_seconds() / 86400
            review_count = max(mem["review_count"], 1)
            weight = max(days_since_reviewed / review_count, 0.01)

        weights.append(weight)

    return random.choices(candidates, weights=weights, k=1)[0]


class InsightResponse(BaseModel):
    """Structured response from the LLM for a memory insight.

    Attributes:
        insight: A 1-2 sentence rephrasing of the core lesson.
        question: A reflective question — call to action, self-reflection,
            or application prompt.
    """

    insight: str
    question: str


@lru_cache
def _get_remind_llm() -> BaseChatModel:
    """Create and return the LLM instance for reminder generation.

    Uses the same model configuration as the main agent but with
    temperature=0.7 for variety in generated insights.

    Returns:
        The configured LLM instance.
    """
    from src.core.config import get_settings

    settings = get_settings()
    return init_chat_model(
        model=settings.llm_model,
        model_provider=settings.llm_provider,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.llm_provider_base_url,
        temperature=0.7,
    )


async def generate_insight(content: str, source: str | None = None) -> InsightResponse:
    """Generate an insight and reflective question from a memory.

    Calls the LLM with a focused prompt to rephrase the memory into
    an insight and generate a reflective question. Uses temperature=0.7
    for variety on repeated calls.

    Args:
        content: The memory content (quote or highlight).
        source: Optional attribution (book title, author, newsletter).

    Returns:
        InsightResponse with insight and question fields.
    """
    llm = _get_remind_llm()
    structured_llm = llm.with_structured_output(InsightResponse)

    user_content = f'Memory: "{content}"'
    if source:
        user_content += f"\nSource: {source}"

    messages = [
        SystemMessage(content=INSIGHT_SYSTEM_PROMPT),
        HumanMessage(content=user_content),
    ]

    result = await structured_llm.ainvoke(messages)
    assert isinstance(result, InsightResponse)
    return result


async def send_reminder(
    chat_id: str,
    content: str,
    source: str | None,
    insight: str,
    question: str,
) -> None:
    """Send a formatted reminder message via Telegram.

    Formats the memory, insight, and question as an HTML message
    and sends it to the specified Telegram chat.

    Args:
        chat_id: The Telegram chat ID to send to.
        content: The original memory content (quote/highlight).
        source: Optional attribution (author, book, newsletter).
        insight: The AI-generated insight.
        question: The AI-generated reflective question.
    """
    from src.core.config import get_settings

    settings = get_settings()
    bot = Bot(token=settings.telegram_bot_token.get_secret_value())

    parts = [f"📖 <blockquote>{content}</blockquote>"]
    if source:
        parts.append(f"— {source}")
    parts.append(f"\n💡 <b>Insight:</b> {insight}")
    parts.append(f"\n❓ <b>Reflect:</b> {question}")

    text = "\n".join(parts)

    await bot.send_message(
        chat_id=chat_id,
        text=text,
        parse_mode=ParseMode.HTML,
    )


def update_memory(memory_id: str, review_count: int) -> None:
    """Update a memory's review metadata after surfacing.

    Sets last_reviewed_at to now and increments review_count by 1.

    Args:
        memory_id: The UUID of the memory to update.
        review_count: The current review count (will be incremented).
    """
    client = get_supabase_client()
    client.table("memories").update(
        {
            "last_reviewed_at": datetime.now(UTC).isoformat(),
            "review_count": review_count + 1,
        }
    ).eq("id", memory_id).execute()


async def main() -> None:
    """Run the spaced repetition reminder job.

    Orchestrates the full flow: find due users, select a memory,
    generate an insight, send via Telegram, and update the memory.
    """
    due_users = get_due_users()
    if not due_users:
        logger.info("No users due for a reminder at this hour.")
        return

    memory = select_memory()

    content = memory["content"]
    source = memory.get("source")
    memory_id = memory["id"]
    review_count = memory["review_count"]

    assert isinstance(content, str)
    assert source is None or isinstance(source, str)
    assert isinstance(memory_id, str)
    assert isinstance(review_count, int)

    insight_resp = await generate_insight(content=content, source=source)

    for chat_id in due_users:
        await send_reminder(
            chat_id=chat_id,
            content=content,
            source=source,
            insight=insight_resp.insight,
            question=insight_resp.question,
        )

    update_memory(memory_id=memory_id, review_count=review_count)
    logger.info("Reminder sent for memory %s to %d user(s).", memory_id, len(due_users))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
