"""Tests for src/api/bot_keyboards.py."""

from __future__ import annotations

from typing import cast

from src.api.bot_keyboards import (
    build_hour_picker_keyboard,
    build_reminders_message,
    build_timezone_keyboard,
)
from src.db.user_settings import ReminderRow


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
        reminders = cast(
            list[ReminderRow],
            [
                {"id": "r1", "time": "08:00:00"},
                {"id": "r2", "time": "20:00:00"},
            ],
        )
        text, keyboard = build_reminders_message(reminders, "aaa")
        assert "08:00" in text
        assert "20:00" in text
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        remove_btns = [btn for btn in buttons if "Remove" in btn.text]
        assert len(remove_btns) == 2

    def test_max_reminders_hides_add(self) -> None:
        """Should not show 'Add reminder' when at 5 reminders."""
        reminders = cast(
            list[ReminderRow],
            [{"id": f"r{i}", "time": f"{i:02d}:00:00"} for i in range(5)],
        )
        _, keyboard = build_reminders_message(reminders, "aaa")
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        assert not any("Add" in btn.text for btn in buttons)

    def test_callback_data_includes_user_settings_id(self) -> None:
        """Callback data should include user_settings_id for parsing."""
        reminders = cast(list[ReminderRow], [{"id": "r1", "time": "08:00:00"}])
        _, keyboard = build_reminders_message(reminders, "test-uuid-123")
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        # Remove button should have format: rm_rem|{idx}|{user_settings_id}|{ob}
        assert buttons[0].callback_data == "rm_rem|0|test-uuid-123|0"
        # Add button should have format: add_rem|{user_settings_id}|{ob}
        assert buttons[1].callback_data == "add_rem|test-uuid-123|0"


class TestBuildHourPickerKeyboard:
    """Tests for build_hour_picker_keyboard."""

    def test_24_hour_buttons(self) -> None:
        """Should have 24 hour buttons."""
        keyboard = build_hour_picker_keyboard("aaa")
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        assert len(buttons) == 24
        assert buttons[0].text == "00:00"
        assert buttons[23].text == "23:00"

    def test_callback_data_includes_user_settings_id(self) -> None:
        """Callback data should include user_settings_id for parsing."""
        keyboard = build_hour_picker_keyboard("test-uuid-456")
        buttons = [btn for row in keyboard.inline_keyboard for btn in row]
        # Each button should have format: add_hr|{hour}|{user_settings_id}|{ob}
        assert buttons[0].callback_data == "add_hr|0|test-uuid-456|0"
        assert buttons[12].callback_data == "add_hr|12|test-uuid-456|0"
        assert buttons[23].callback_data == "add_hr|23|test-uuid-456|0"
