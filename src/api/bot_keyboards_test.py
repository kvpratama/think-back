"""Tests for src/api/bot_keyboards.py."""

from __future__ import annotations

from src.api.bot_keyboards import (
    build_hour_picker_keyboard,
    build_reminders_message,
    build_timezone_keyboard,
)


class TestBuildTimezoneKeyboard:
    """Tests for build_timezone_keyboard."""

    def test_returns_inline_keyboard(self) -> None:
        """Should return an InlineKeyboardMarkup with UTC offset buttons."""
        keyboard = build_timezone_keyboard(67890)
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        labels = [btn.text for btn in buttons]
        assert "UTC+0" in labels
        assert "UTC+7" in labels
        assert "UTC-5" in labels

    def test_callback_data_includes_chat_id(self) -> None:
        """Each button callback_data should encode the chat_id."""
        keyboard = build_timezone_keyboard(12345)
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        assert all("12345" in str(btn.callback_data) for btn in buttons)


class TestBuildRemindersMessage:
    """Tests for build_reminders_message."""

    def test_no_reminders(self) -> None:
        """Should show 'no reminders' text when list is empty."""
        text, keyboard = build_reminders_message([], "aaa")
        assert "no reminders" in text.lower()
        # Should still have the "Add reminder" button
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        assert any("Add" in btn.text for btn in buttons)

    def test_with_reminders(self) -> None:
        """Should list reminder times and include remove buttons."""
        reminders = [
            {"id": "r1", "time": "08:00:00"},
            {"id": "r2", "time": "20:00:00"},
        ]
        text, keyboard = build_reminders_message(reminders, "aaa")
        assert "08:00" in text
        assert "20:00" in text
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        remove_btns = [btn for btn in buttons if "Remove" in btn.text]
        assert len(remove_btns) == 2

    def test_max_reminders_hides_add(self) -> None:
        """Should not show 'Add reminder' when at 5 reminders."""
        reminders = [{"id": f"r{i}", "time": f"{i:02d}:00:00"} for i in range(5)]
        _, keyboard = build_reminders_message(reminders, "aaa")
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        assert not any("Add" in btn.text for btn in buttons)


class TestBuildHourPickerKeyboard:
    """Tests for build_hour_picker_keyboard."""

    def test_24_hour_buttons(self) -> None:
        """Should have 24 hour buttons."""
        keyboard = build_hour_picker_keyboard("aaa")
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        assert len(buttons) == 24
        assert buttons[0].text == "00:00"
        assert buttons[23].text == "23:00"
