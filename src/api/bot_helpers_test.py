"""Tests for src/api/bot_helpers.py."""

from __future__ import annotations

from src.api.bot_helpers import sanitize_for_telegram_html, truncate_for_telegram


class TestSanitizeForTelegramHtml:
    """Tests for the sanitize_for_telegram_html function."""

    def test_plain_text_unchanged(self) -> None:
        """Plain text without HTML should pass through unchanged."""
        assert sanitize_for_telegram_html("Hello world") == "Hello world"

    def test_supported_tags_preserved(self) -> None:
        """Supported tags like <b>, <i>, <code> should be kept."""
        assert (
            sanitize_for_telegram_html("<b>bold</b> <i>italic</i>") == "<b>bold</b> <i>italic</i>"
        )

    def test_strong_em_preserved(self) -> None:
        """<strong> and <em> are supported aliases."""
        assert (
            sanitize_for_telegram_html("<strong>bold</strong> <em>italic</em>")
            == "<strong>bold</strong> <em>italic</em>"
        )

    def test_unordered_list_to_bullets(self) -> None:
        """<ul><li>...</li></ul> should become bullet points."""
        html = "<ul><li>Item A</li><li>Item B</li><li>Item C</li></ul>"
        result = sanitize_for_telegram_html(html)
        assert "• Item A" in result
        assert "• Item B" in result
        assert "• Item C" in result
        assert "<ul>" not in result
        assert "<li>" not in result

    def test_ordered_list_to_numbered(self) -> None:
        """<ol><li>...</li></ol> should become numbered list."""
        html = "<ol><li>First</li><li>Second</li><li>Third</li></ol>"
        result = sanitize_for_telegram_html(html)
        assert "1. First" in result
        assert "2. Second" in result
        assert "3. Third" in result
        assert "<ol>" not in result

    def test_heading_to_bold(self) -> None:
        """<h1>-<h6> should become <b>...</b>."""
        assert sanitize_for_telegram_html("<h1>Title</h1>") == "<b>Title</b>"
        assert sanitize_for_telegram_html("<h3>Sub</h3>") == "<b>Sub</b>"

    def test_br_to_newline(self) -> None:
        """<br> and <br/> should become newlines."""
        assert sanitize_for_telegram_html("Line1<br>Line2") == "Line1\nLine2"
        assert sanitize_for_telegram_html("Line1<br/>Line2") == "Line1\nLine2"

    def test_unsupported_tag_stripped_content_kept(self) -> None:
        """Unsupported tags like <div> should be stripped but content kept."""
        assert sanitize_for_telegram_html("<div>content</div>") == "content"

    def test_bare_angle_brackets_escaped(self) -> None:
        """Standalone < and > should be HTML-escaped."""
        assert sanitize_for_telegram_html("5 < 10") == "5 &lt; 10"
        assert sanitize_for_telegram_html("a > b") == "a &gt; b"

    def test_ampersand_escaped(self) -> None:
        """Standalone & should be escaped."""
        assert sanitize_for_telegram_html("A & B") == "A &amp; B"

    def test_nested_list_handling(self) -> None:
        """Nested lists inside <ul> should not break."""
        html = "<ul><li>Outer<li><ul><li>Inner</li></ul></li></ul>"
        result = sanitize_for_telegram_html(html)
        # Should not raise; inner list gets processed first
        assert "<ul>" not in result
        assert "<li>" not in result

    def test_orphaned_li_becomes_bullet(self) -> None:
        """Standalone <li> without <ul> should become a bullet."""
        html = "<li>Orphan item</li>"
        result = sanitize_for_telegram_html(html)
        assert "• Orphan item" in result
        assert "<li>" not in result

    def test_code_block_preserved(self) -> None:
        """<code> and <pre> tags should be preserved."""
        html = "Use <code>print()</code> and <pre>block</pre>"
        assert sanitize_for_telegram_html(html) == html

    def test_link_preserved(self) -> None:
        """<a href> links should be preserved."""
        html = '<a href="https://example.com">link</a>'
        assert sanitize_for_telegram_html(html) == html

    def test_mixed_supported_and_unsupported(self) -> None:
        """Mix of supported and unsupported tags handled correctly."""
        html = "<b>Bold</b> <unknown>text</unknown> <i>italic</i>"
        result = sanitize_for_telegram_html(html)
        assert "<b>Bold</b>" in result
        assert "<i>italic</i>" in result
        assert "text" in result
        assert "<unknown>" not in result

    def test_empty_string(self) -> None:
        """Empty string should return empty string."""
        assert sanitize_for_telegram_html("") == ""

    def test_complex_llm_output(self) -> None:
        """Realistic LLM output with mixed formatting."""
        html = (
            "Here are the steps:\n"
            "<ol><li>Install deps</li><li>Run tests</li><li>Deploy</li></ol>\n"
            "<b>Note:</b> check <code>config.yaml</code>"
        )
        result = sanitize_for_telegram_html(html)
        assert "1. Install deps" in result
        assert "2. Run tests" in result
        assert "3. Deploy" in result
        assert "<b>Note:</b>" in result
        assert "<code>config.yaml</code>" in result
        assert "<ol>" not in result

    def test_idempotent(self) -> None:
        """Running sanitize twice should produce the same result."""
        html = "<ul><li>A</li><li>B</li></ul>"
        once = sanitize_for_telegram_html(html)
        twice = sanitize_for_telegram_html(once)
        assert once == twice

    def test_blockquote_preserved(self) -> None:
        """<blockquote> is supported and should be preserved."""
        html = "<blockquote>quoted text</blockquote>"
        assert sanitize_for_telegram_html(html) == html

    def test_ul_with_attributes(self) -> None:
        """<ul class='...'> should still be handled."""
        html = '<ul class="list"><li>Item</li></ul>'
        result = sanitize_for_telegram_html(html)
        assert "• Item" in result
        assert "<ul" not in result


class TestTruncateForTelegram:
    """Tests for the truncate_for_telegram function."""

    def test_short_text_unchanged(self) -> None:
        """Text under the limit should pass through unchanged."""
        assert truncate_for_telegram("Hello") == "Hello"

    def test_long_text_truncated(self) -> None:
        """Text over the limit should be truncated with ellipsis."""
        text = "A" * 5000
        result = truncate_for_telegram(text, max_len=100)
        assert len(result) <= 100
        assert result.endswith("…")

    def test_closes_unclosed_bold(self) -> None:
        """Unclosed <b> should be closed at truncation point."""
        text = "<b>Bold text that is very long " + "x" * 200
        result = truncate_for_telegram(text, max_len=50)
        assert result.count("<b>") == 1
        assert result.count("</b>") == 1
        assert result.endswith("</b>…")

    def test_closes_nested_tags(self) -> None:
        """Nested unclosed tags should all be closed."""
        text = "<b><i>Text " + "x" * 200
        result = truncate_for_telegram(text, max_len=30)
        # Should close </i> then </b>
        assert result.endswith("</i></b>…")

    def test_no_extra_close_for_matched_tags(self) -> None:
        """Already-closed tags should not get duplicate closing tags."""
        text = "<b>Done</b> " + "extra " * 50
        result = truncate_for_telegram(text, max_len=30)
        # Should not add extra </b> since it's already closed
        assert result.count("</b>") == 1
        assert result.endswith("…")

    def test_sanitization_applied_first(self) -> None:
        """Truncation should sanitize first (e.g. convert <ul> to bullets)."""
        text = "<ul><li>" + "item " * 100 + "</li></ul>"
        result = truncate_for_telegram(text, max_len=50)
        assert "<ul>" not in result
        assert "• item" in result

    def test_exact_limit(self) -> None:
        """Text exactly at the limit should not be truncated."""
        text = "A" * 100
        result = truncate_for_telegram(text, max_len=100)
        assert result == text
        assert "…" not in result

    def test_one_over_limit(self) -> None:
        """Text one char over the limit should be truncated."""
        text = "A" * 101
        result = truncate_for_telegram(text, max_len=100)
        assert len(result) <= 100
        assert result.endswith("…")
