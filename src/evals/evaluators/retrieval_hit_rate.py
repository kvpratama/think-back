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
    Compute retrieval hit rate via exact string matching.

    Checks whether at least one string in `expected_contents` appears
    in the retrieved documents (`retrieved_memories[*]["content"]`).

    Special case:
        If `expected_contents` is empty (intended no-match case),
        returns score = 1. Hallucinations should be handled by
        a separate evaluator (e.g., answer_faithfulness).

    Args:
        run (Run): Contains pipeline outputs:
            {
                "retrieved_memories": [{"content": "...", ...}, ...],
                "answer": "..."
            }
        example (Example): Contains expected outputs:
            {
                "expected_contents": [str, ...],
                "case_type": str,
                ...
            }

    Returns:
        dict:
            {
                "key": "retrieval_hit_rate",
                "score": 0 | 1,
                "comment": str
            }

    Raises:
        KeyError: If run.outputs or example.outputs is missing expected keys
            (e.g., "retrieved_memories" or "expected_contents").
        TypeError: If retrieved_memories is not a list of dicts with "content"
            keys, or if expected_contents is not a list of strings.
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

    retrieved_contents = {doc["content"] for doc in retrieved_memories if "content" in doc}

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
