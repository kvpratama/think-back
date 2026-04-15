"""Helper utilities for Telegram bot messaging.

This module provides functions to sanitize LLM output so it's compatible
with Telegram's limited HTML parse mode.
"""

from __future__ import annotations

import re

# Telegram HTML supports only: b, strong, i, em, u, ins, s, strike, del,
# a, code, pre, tg-spoiler, blockquote, expandable_blockquote
SUPPORTED_TAGS = {
    "b",
    "strong",
    "i",
    "em",
    "u",
    "ins",
    "s",
    "strike",
    "del",
    "a",
    "code",
    "pre",
    "tg-spoiler",
    "blockquote",
    "expandable_blockquote",
}


def sanitize_for_telegram_html(text: str) -> str:
    """Sanitize a string so it's safe for Telegram's HTML parse mode.

    Transforms unsupported HTML tags into Telegram-compatible equivalents:
    - ``<ul>`` / ``<ol>`` list items become bullet/numbered lines
    - ``<h1>``-``<h6>`` become ``<b>...</b>``
    - ``<br>`` / ``<br/>`` become newlines
    - All other unsupported tags are stripped (content preserved)
    - Bare ``<``, ``>``, ``&`` are escaped

    Args:
        text: The raw string (potentially containing HTML from an LLM).

    Returns:
        A string safe to pass to Telegram with ``parse_mode=ParseMode.HTML``.
    """
    # --- Transform known unsupported tags ---

    # <br> / <br/> → newline
    text = re.sub(r"<\s*/?\s*br\s*/?\s*>", "\n", text, flags=re.IGNORECASE)

    # <h1>-<h6> → <b>...</b>
    text = re.sub(
        r"<\s*h[1-6][^>]*>(.*?)<\s*/\s*h[1-6]\s*>",
        r"<b>\1</b>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # <ul><li>...</li></ul> → bullet points
    def _replace_ul(match: re.Match[str]) -> str:
        inner = match.group(1)
        items = re.split(r"<\s*/?\s*li\s*/?\s*>", inner, flags=re.IGNORECASE)
        bullets = [item.strip() for item in items if item.strip()]
        return "\n".join(f"• {item}" for item in bullets)

    text = re.sub(
        r"<\s*ul[^>]*>(.*?)<\s*/\s*ul\s*>",
        _replace_ul,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # <ol><li>...</li></ol> → numbered list
    def _replace_ol(match: re.Match[str]) -> str:
        inner = match.group(1)
        items = re.split(r"<\s*/?\s*li\s*/?\s*>", inner, flags=re.IGNORECASE)
        items = [item.strip() for item in items if item.strip()]
        lines = [f"{i + 1}. {item}" for i, item in enumerate(items)]
        return "\n".join(lines)

    text = re.sub(
        r"<\s*ol[^>]*>(.*?)<\s*/\s*ol\s*>",
        _replace_ol,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # Orphaned <li> (not inside <ul>/<ol>) → bullet
    text = re.sub(
        r"<\s*li[^>]*>(.*?)<\s*/\s*li\s*>",
        r"• \1",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    # --- Strip remaining unsupported tags (preserve content) ---
    def _strip_or_keep(match: re.Match[str]) -> str:
        tag_match = re.match(r"<\s*/?\s*(\w+)", match.group(0))
        if tag_match and tag_match.group(1).lower() in SUPPORTED_TAGS:
            return match.group(0)  # keep supported tags as-is
        # For unsupported tags, return empty string (strip the tag)
        return ""

    text = re.sub(r"<[^>]+>", _strip_or_keep, text)

    # --- Escape bare < > & that aren't part of valid tags ---
    # Protect existing valid tags temporarily
    placeholder = "\x00TAG_PLACEHOLDER\x00"
    tag_spans: list[str] = []

    def _capture_tag(match: re.Match[str]) -> str:
        tag_spans.append(match.group(0))
        return placeholder

    text = re.sub(r"<[^>]+>", _capture_tag, text)

    # Escape remaining special characters
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    # Restore captured tags
    idx = 0

    def _restore_tag(match: re.Match[str]) -> str:
        nonlocal idx
        result = tag_spans[idx]
        idx += 1
        return result

    text = re.sub(re.escape(placeholder), _restore_tag, text)

    return text
