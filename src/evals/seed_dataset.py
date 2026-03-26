"""Seed the LangSmith 'thinkback-eval' dataset and tag the snapshot.

Versioning strategy — uses LangSmith's native snapshot-based versioning:
  - A single dataset 'thinkback-eval' is maintained (no more thinkback-eval-v1, v2 etc.)
  - Every add/update/delete automatically creates a new timestamped snapshot
  - DATASET_TAG pins a human-readable label to the snapshot after seeding
  - Evaluations run against a tag (e.g. as_of="v1") to get a stable, reproducible corpus
  - Bumping DATASET_TAG to "v2" and re-running adds new examples without touching v1

To evaluate against a specific version:
    client.list_examples(dataset_name="thinkback-eval", as_of="v1")

To cut a new version:
    1. Add/edit examples in dataset_examples.py
    2. Bump DATASET_TAG to "v2"
    3. Re-run this script — existing examples are preserved, new ones added, snapshot tagged

Usage:
    python -m src.evals.seed_dataset

Requires:
    LANGCHAIN_API_KEY set in environment (or .env file)
"""

from datetime import UTC, datetime

from dotenv import load_dotenv
from langsmith import Client

from src.evals.dataset_examples import (
    DATASET_DESCRIPTION,
    DATASET_NAME,
    DATASET_TAG,
    EXAMPLES,
)


def main() -> None:
    """Seed the LangSmith dataset with hand-crafted eval examples and tag the snapshot."""
    load_dotenv()

    client = Client()

    # Get or create the single long-lived dataset
    existing = next((d for d in client.list_datasets() if d.name == DATASET_NAME), None)

    if existing:
        dataset = existing
        print(f"Dataset '{DATASET_NAME}' already exists ({dataset.id}) — adding examples...")
    else:
        print(f"Creating dataset: '{DATASET_NAME}'...")
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESCRIPTION,
        )
        print(f"Dataset created: {dataset.id}")

    print(f"Seeding {len(EXAMPLES)} examples...")
    client.create_examples(
        dataset_id=dataset.id,
        examples=EXAMPLES,
    )

    # Verify upload count matches what we sent
    remote_count = client.read_dataset(dataset_name=DATASET_NAME).example_count
    assert remote_count >= len(EXAMPLES), (
        f"Upload mismatch: expected at least {len(EXAMPLES)} examples, got {remote_count}"
    )

    # Tag this snapshot with the version label
    # This pins the current state — evaluate with as_of=DATASET_TAG for reproducibility
    tagged_at = datetime.now(UTC)
    client.update_dataset_tag(
        dataset_name=DATASET_NAME,
        as_of=tagged_at,
        tag=DATASET_TAG,
    )
    print(f"Snapshot tagged as '{DATASET_TAG}' at {tagged_at.isoformat()}")

    happy = sum(1 for e in EXAMPLES if e["metadata"]["case_type"] == "happy_path")
    edge = sum(1 for e in EXAMPLES if e["metadata"]["case_type"] == "edge_case")
    print(f"Done. {happy} happy path + {edge} edge case examples seeded.")
    print(
        f"Evaluate with: client.list_examples(dataset_name='{DATASET_NAME}', as_of='{DATASET_TAG}')"
    )
    print(f"View at: https://smith.langchain.com/datasets/{dataset.id}")


if __name__ == "__main__":
    main()
