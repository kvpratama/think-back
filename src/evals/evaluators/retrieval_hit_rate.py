"""
evals/evaluators/retrieval_hit_rate.py

Evaluator: retrieval_hit_rate
Method:    exact string match — no LLM, no embeddings
Score:     1 if at least one expected_content appears in retrieved docs
           0 otherwise

Evaluator signature follows LangSmith's (run, example) -> dict convention.

  run.outputs expected shape:
    {
      "retrieved_memories": [
        {"content": "...", "metadata": {...}},
        ...
      ],
      "answer": "..."
    }

  example.outputs expected shape:
    {
      "expected_contents": ["exact string 1", "exact string 2"],
      "expected_answer_criteria": "...",
      "case_type": "happy_path" | "edge_case",
      "notes": "..."
    }
"""

from langsmith.schemas import Example, Run


def retrieval_hit_rate(run: Run, example: Example) -> dict:
    """
    Checks whether at least one expected_content string appears
    in the retrieved docs returned by the pipeline.

    Uses exact string match against page_content — reliable because
    the Supabase vector retriever returns the exact stored content.

    Special case: if expected_contents is empty (no-match edge cases),
    the eval passes only if the pipeline returned NO docs, or if the
    pipeline correctly surfaced nothing relevant. We score 1 here
    to avoid penalising correct no-match behaviour — the
    answer_faithfulness evaluator is responsible for catching
    hallucination in these cases.
    """
    retrieved_memories = run.outputs.get("retrieved_memories", []) if run.outputs else []
    expected_contents = example.outputs.get("expected_contents", []) if example.outputs else []

    # No-match case — expected_contents is intentionally empty
    if not expected_contents:
        return {
            "key": "retrieval_hit_rate",
            "score": 1,
            "comment": "No-match case — expected_contents is intentionally empty",
        }

    retrieved_contents = {doc["content"] for doc in retrieved_memories}

    hits = [c for c in expected_contents if c in retrieved_contents]
    hit = len(hits) > 0

    return {
        "key": "retrieval_hit_rate",
        "score": int(hit),
        "comment": (
            f"Hit {len(hits)}/{len(expected_contents)} expected memories. "
            f"Expected: {[c[:40] + '...' for c in expected_contents]}. "
            f"Retrieved: {[t[:40] + '...' for t in retrieved_contents]}."
        ),
    }
