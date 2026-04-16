"""Tests for the spaced repetition reminder job."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def mock_supabase() -> MagicMock:
    """Create a mock Supabase client."""
    client = MagicMock()
    return client


class TestGetDueUsers:
    """Tests for get_due_users()."""

    def test_returns_user_whose_reminder_time_matches_current_hour(
        self, mock_supabase: MagicMock
    ) -> None:
        mock_data = [
            {
                "telegram_chat_id": "123456",
                "reminder_time_1": "08:00:00",
                "reminder_time_2": "20:00:00",
                "timezone": "UTC",
                "reminders_enabled": True,
            },
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = mock_data  # noqa: E501

        with patch("src.jobs.remind.get_supabase_client", return_value=mock_supabase):
            from src.jobs.remind import get_due_users

            # Simulate running at 08:30 UTC
            fake_now = datetime(2026, 4, 16, 8, 30, tzinfo=UTC)
            result = get_due_users(now=fake_now)

        assert result == ["123456"]

    def test_skips_user_whose_reminder_time_does_not_match(self, mock_supabase: MagicMock) -> None:
        mock_data = [
            {
                "telegram_chat_id": "123456",
                "reminder_time_1": "08:00:00",
                "reminder_time_2": "20:00:00",
                "timezone": "UTC",
                "reminders_enabled": True,
            },
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = mock_data  # noqa: E501

        with patch("src.jobs.remind.get_supabase_client", return_value=mock_supabase):
            from src.jobs.remind import get_due_users

            # Simulate running at 15:00 UTC — neither 08 nor 20
            fake_now = datetime(2026, 4, 16, 15, 0, tzinfo=UTC)
            result = get_due_users(now=fake_now)

        assert result == []

    def test_respects_user_timezone(self, mock_supabase: MagicMock) -> None:
        mock_data = [
            {
                "telegram_chat_id": "999",
                "reminder_time_1": "08:00:00",
                "reminder_time_2": "20:00:00",
                "timezone": "Asia/Jakarta",  # UTC+7
                "reminders_enabled": True,
            },
        ]
        mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = mock_data  # noqa: E501

        with patch("src.jobs.remind.get_supabase_client", return_value=mock_supabase):
            from src.jobs.remind import get_due_users

            # 01:00 UTC = 08:00 Jakarta → should match reminder_time_1
            fake_now = datetime(2026, 4, 16, 1, 0, tzinfo=UTC)
            result = get_due_users(now=fake_now)

        assert result == ["999"]

    def test_skips_user_with_reminders_disabled(self, mock_supabase: MagicMock) -> None:
        # The Supabase query filters by reminders_enabled=True,
        # so disabled users won't appear in the response.
        mock_result = mock_supabase.table.return_value.select.return_value.eq.return_value.execute.return_value  # noqa: E501
        mock_result.data = []

        with patch("src.jobs.remind.get_supabase_client", return_value=mock_supabase):
            from src.jobs.remind import get_due_users

            fake_now = datetime(2026, 4, 16, 8, 0, tzinfo=UTC)
            result = get_due_users(now=fake_now)

        assert result == []


class TestSelectMemory:
    """Tests for select_memory()."""

    def test_weights_novel_memories_by_age(self, mock_supabase: MagicMock) -> None:
        now = datetime(2026, 4, 16, 8, 0, tzinfo=UTC)

        mock_supabase.table.return_value.select.return_value.execute.return_value.data = [
            {
                "id": "aaa",
                "content": "Old wisdom",
                "source": "Book A",
                "created_at": (now - timedelta(days=10)).isoformat(),
                "last_reviewed_at": None,
                "review_count": 0,
            },
            {
                "id": "bbb",
                "content": "Fresh wisdom",
                "source": "Book B",
                "created_at": (now - timedelta(hours=2)).isoformat(),
                "last_reviewed_at": None,
                "review_count": 0,
            },
        ]

        with (
            patch("src.jobs.remind.get_supabase_client", return_value=mock_supabase),
            patch("random.choices") as mock_choices,
        ):
            mock_choices.return_value = [
                mock_supabase.table.return_value.select.return_value.execute.return_value.data[0]
            ]

            from src.jobs.remind import select_memory

            select_memory(now=now)

            # Verify weights: old memory should have much higher weight
            call_args = mock_choices.call_args
            weights = call_args.kwargs["weights"]
            assert weights[0] > weights[1]  # 10 days > 0.08 days

    def test_weights_reviewed_memories_by_staleness(self, mock_supabase: MagicMock) -> None:
        now = datetime(2026, 4, 16, 8, 0, tzinfo=UTC)

        mock_supabase.table.return_value.select.return_value.execute.return_value.data = [
            {
                "id": "aaa",
                "content": "Stale memory",
                "source": None,
                "created_at": (now - timedelta(days=30)).isoformat(),
                "last_reviewed_at": (now - timedelta(days=14)).isoformat(),
                "review_count": 2,
            },
            {
                "id": "bbb",
                "content": "Fresh memory",
                "source": None,
                "created_at": (now - timedelta(days=30)).isoformat(),
                "last_reviewed_at": (now - timedelta(hours=3)).isoformat(),
                "review_count": 5,
            },
        ]

        with (
            patch("src.jobs.remind.get_supabase_client", return_value=mock_supabase),
            patch("random.choices") as mock_choices,
        ):
            mock_choices.return_value = [
                mock_supabase.table.return_value.select.return_value.execute.return_value.data[0]
            ]

            from src.jobs.remind import select_memory

            select_memory(now=now)

            call_args = mock_choices.call_args
            weights = call_args.kwargs["weights"]
            # Stale: 14 / 2 = 7.0, Fresh: 0.125 / 5 = 0.025
            assert weights[0] > weights[1]

    def test_raises_when_no_memories_exist(self, mock_supabase: MagicMock) -> None:
        mock_supabase.table.return_value.select.return_value.execute.return_value.data = []

        from src.jobs.remind import select_memory

        with (
            patch("src.jobs.remind.get_supabase_client", return_value=mock_supabase),
            pytest.raises(RuntimeError, match="No memories available"),
        ):
            select_memory()


class TestGenerateInsight:
    """Tests for generate_insight()."""

    async def test_returns_structured_insight_and_question(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()

        from src.jobs.remind import InsightResponse

        mock_structured.ainvoke = AsyncMock(
            return_value=InsightResponse(
                insight="Systems beat goals.",
                question="Where are you relying on motivation instead of a system?",
            )
        )
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("src.jobs.remind._get_remind_llm", return_value=mock_llm):
            from src.jobs.remind import generate_insight

            result = await generate_insight(
                content="We don't rise to the level of our goals.",
                source="James Clear",
            )

        assert result.insight == "Systems beat goals."
        assert result.question == "Where are you relying on motivation instead of a system?"
        # Verify the prompt includes the content and source
        call_args = mock_structured.ainvoke.call_args[0][0]
        prompt_text = call_args[-1].content  # last message is the user message
        assert "We don't rise to the level of our goals." in prompt_text
        assert "James Clear" in prompt_text

    async def test_omits_source_when_none(self) -> None:
        mock_llm = MagicMock()
        mock_structured = MagicMock()

        from src.jobs.remind import InsightResponse

        mock_structured.ainvoke = AsyncMock(
            return_value=InsightResponse(
                insight="An insight.",
                question="A question?",
            )
        )
        mock_llm.with_structured_output.return_value = mock_structured

        with patch("src.jobs.remind._get_remind_llm", return_value=mock_llm):
            from src.jobs.remind import generate_insight

            await generate_insight(content="Some quote", source=None)

        call_args = mock_structured.ainvoke.call_args[0][0]
        prompt_text = call_args[-1].content
        assert "Source:" not in prompt_text


class TestSendReminder:
    """Tests for send_reminder()."""

    async def test_sends_formatted_message_with_source(self) -> None:
        mock_bot_class = MagicMock()
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        mock_bot_class.return_value = mock_bot

        with patch("src.jobs.remind.Bot", mock_bot_class):
            from src.jobs.remind import send_reminder

            await send_reminder(
                chat_id="123456",
                content="We don't rise to the level of our goals.",
                source="James Clear",
                insight="Systems beat goals.",
                question="Where are you relying on motivation?",
            )

        mock_bot.send_message.assert_called_once()
        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert call_kwargs["chat_id"] == "123456"
        text = call_kwargs["text"]
        assert "We don't rise to the level of our goals." in text
        assert "James Clear" in text
        assert "Systems beat goals." in text
        assert "Where are you relying on motivation?" in text

    async def test_omits_source_line_when_none(self) -> None:
        mock_bot_class = MagicMock()
        mock_bot = MagicMock()
        mock_bot.send_message = AsyncMock()
        mock_bot_class.return_value = mock_bot

        with patch("src.jobs.remind.Bot", mock_bot_class):
            from src.jobs.remind import send_reminder

            await send_reminder(
                chat_id="123456",
                content="Some quote",
                source=None,
                insight="An insight.",
                question="A question?",
            )

        text = mock_bot.send_message.call_args.kwargs["text"]
        assert "—" not in text


class TestUpdateMemory:
    """Tests for update_memory()."""

    def test_increments_review_count_and_sets_last_reviewed_at(
        self, mock_supabase: MagicMock
    ) -> None:
        # mock_result = mock_supabase.table.return_value.update.return_value.eq.return_value.execute.return_value  # noqa: E501
        # mock_result = MagicMock()

        with patch("src.jobs.remind.get_supabase_client", return_value=mock_supabase):
            from src.jobs.remind import update_memory

            update_memory(memory_id="abc-123", review_count=3)

        mock_supabase.table.assert_called_with("memories")
        update_call = mock_supabase.table.return_value.update.call_args[0][0]
        assert update_call["review_count"] == 4
        assert "last_reviewed_at" in update_call


class TestMain:
    """Tests for main() orchestration."""

    async def test_full_flow_sends_reminder_and_updates_memory(self) -> None:
        from src.jobs.remind import InsightResponse

        fake_memory = {
            "id": "mem-1",
            "content": "A great quote",
            "source": "Author",
            "created_at": "2026-04-01T00:00:00+00:00",
            "last_reviewed_at": None,
            "review_count": 0,
        }

        with (
            patch("src.jobs.remind.get_due_users", return_value=["chat-123"]) as mock_get_users,
            patch("src.jobs.remind.select_memory", return_value=fake_memory) as mock_select,
            patch(
                "src.jobs.remind.generate_insight",
                new_callable=AsyncMock,
                return_value=InsightResponse(insight="Insight.", question="Question?"),
            ) as mock_generate,
            patch("src.jobs.remind.send_reminder", new_callable=AsyncMock) as mock_send,
            patch("src.jobs.remind.update_memory") as mock_update,
        ):
            from src.jobs.remind import main

            await main()

        mock_get_users.assert_called_once()
        mock_select.assert_called_once()
        mock_generate.assert_called_once_with(content="A great quote", source="Author")
        mock_send.assert_called_once_with(
            chat_id="chat-123",
            content="A great quote",
            source="Author",
            insight="Insight.",
            question="Question?",
        )
        mock_update.assert_called_once_with(memory_id="mem-1", review_count=0)

    async def test_exits_early_when_no_due_users(self) -> None:
        with (
            patch("src.jobs.remind.get_due_users", return_value=[]),
            patch("src.jobs.remind.select_memory") as mock_select,
        ):
            from src.jobs.remind import main

            await main()

        mock_select.assert_not_called()
