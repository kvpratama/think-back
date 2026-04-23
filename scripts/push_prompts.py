"""One-time script to push prompts to LangSmith Hub.

Run once: uv run python scripts/push_prompts.py

Pushes all prompts from _DEFAULTS to LangSmith Hub with :prod tag.
"""

from langsmith import Client

from src.core.prompts import _DEFAULTS


def main() -> None:
    """Push all prompts to LangSmith Hub."""
    client = Client()

    for name, prompt in _DEFAULTS.items():
        url = client.push_prompt(name, object=prompt, commit_tags=["prod"])
        print(f"✓ Pushed {name}: {url}")


if __name__ == "__main__":
    main()
