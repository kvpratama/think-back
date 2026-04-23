"""One-time script to push prompts to LangSmith Hub.

Run once: uv run python scripts/push_prompts.py

Pushes all prompts from _DEFAULTS to LangSmith Hub with :prod tag.
"""

import os
import sys

from dotenv import load_dotenv
from langsmith import Client

from src.core.prompt_defaults import DEFAULTS as _DEFAULTS

load_dotenv()


def main() -> None:
    """Push all prompts to LangSmith Hub."""
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("Error: LANGSMITH_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = Client()
    successes: list[tuple[str, str]] = []
    failures: list[tuple[str, str]] = []

    for name, prompt in _DEFAULTS.items():
        try:
            url = client.push_prompt(name, object=prompt, commit_tags=["prod"])
            successes.append((name, url))
            print(f"✓ Pushed {name}: {url}")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"✗ Failed to push {name}: {exc}", file=sys.stderr)

    print("\n--- Summary ---")
    print(f"Pushed: {len(successes)}/{len(_DEFAULTS)}")
    if failures:
        print(f"Failed: {len(failures)}")
        for name, error in failures:
            print(f"  - {name}: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
