"""Tests for the retrieval_hit_rate evaluator."""

import uuid
from datetime import datetime

from langsmith.schemas import Example, Run

from src.evals.evaluators.retrieval_hit_rate import retrieval_hit_rate


def _make_run(retrieved_memories: list[dict] | None = None) -> Run:
    """Build a minimal Run with the given retrieved_memories in outputs."""
    outputs = {"retrieved_memories": retrieved_memories} if retrieved_memories is not None else None
    return Run(
        id=uuid.uuid4(),
        name="test",
        run_type="chain",
        inputs={},
        start_time=datetime.now(),
        outputs=outputs,
    )


def _make_example(
    expected_contents: list[str] | None = None,
    metadata: dict | None = None,
) -> Example:
    """Build a minimal Example with the given expected_contents in outputs."""
    outputs = {"expected_contents": expected_contents} if expected_contents is not None else None
    return Example(
        id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        inputs={"user_input": "test question"},
        outputs=outputs,
        metadata=metadata,
    )


# ── No-match cases (empty expected_contents) ────────────────────────────────


class TestNoMatchCase:
    """When expected_contents is empty, score should be 1 (correct no-match)."""

    def test_empty_expected_contents_scores_1(self) -> None:
        run = _make_run(retrieved_memories=[{"content": "anything"}])
        example = _make_example(expected_contents=[])

        result = retrieval_hit_rate(run, example)

        assert result["key"] == "retrieval_hit_rate"
        assert result["score"] == 1

    def test_empty_expected_contents_comment_explains_no_match(self) -> None:
        run = _make_run(retrieved_memories=[])
        example = _make_example(expected_contents=[])

        result = retrieval_hit_rate(run, example)

        assert "no-match" in result["comment"].lower()


# ── Hit cases ────────────────────────────────────────────────────────────────


class TestHitCases:
    """When at least one expected content is found in retrieved docs, score 1."""

    def test_single_expected_single_retrieved_exact_match(self) -> None:
        memory = "Consistency beats intensity"
        run = _make_run(retrieved_memories=[{"content": memory}])
        example = _make_example(expected_contents=[memory])

        result = retrieval_hit_rate(run, example)

        assert result["score"] == 1

    def test_one_of_many_expected_found(self) -> None:
        found = "Mood Follows Action"
        missing = "Goals are for people who care about winning once."
        run = _make_run(retrieved_memories=[{"content": found}])
        example = _make_example(expected_contents=[found, missing])

        result = retrieval_hit_rate(run, example)

        assert result["score"] == 1

    def test_all_expected_found(self) -> None:
        mem_a = "Consistency beats intensity"
        mem_b = "Mood Follows Action"
        run = _make_run(
            retrieved_memories=[{"content": mem_a}, {"content": mem_b}, {"content": "extra"}]
        )
        example = _make_example(expected_contents=[mem_a, mem_b])

        result = retrieval_hit_rate(run, example)

        assert result["score"] == 1

    def test_comment_includes_hit_count(self) -> None:
        mem = "Consistency beats intensity"
        run = _make_run(retrieved_memories=[{"content": mem}])
        example = _make_example(expected_contents=[mem])

        result = retrieval_hit_rate(run, example)

        assert "1/1" in result["comment"]


# ── Miss cases ───────────────────────────────────────────────────────────────


class TestMissCases:
    """When no expected content is found in retrieved docs, score 0."""

    def test_no_overlap_scores_0(self) -> None:
        run = _make_run(retrieved_memories=[{"content": "unrelated memory"}])
        example = _make_example(expected_contents=["Consistency beats intensity"])

        result = retrieval_hit_rate(run, example)

        assert result["score"] == 0

    def test_empty_retrieval_scores_0(self) -> None:
        run = _make_run(retrieved_memories=[])
        example = _make_example(expected_contents=["Consistency beats intensity"])

        result = retrieval_hit_rate(run, example)

        assert result["score"] == 0

    def test_miss_comment_shows_0_hits(self) -> None:
        run = _make_run(retrieved_memories=[{"content": "wrong"}])
        example = _make_example(expected_contents=["expected"])

        result = retrieval_hit_rate(run, example)

        assert "0/1" in result["comment"]


# ── Edge cases (missing/None outputs) ────────────────────────────────────────


class TestEdgeCases:
    """Graceful handling when run or example outputs are None."""

    def test_run_outputs_none_with_expected_contents_scores_0(self) -> None:
        run = _make_run()  # outputs=None
        example = _make_example(expected_contents=["something"])

        result = retrieval_hit_rate(run, example)

        assert result["score"] == 0

    def test_example_outputs_none_treated_as_no_match(self) -> None:
        run = _make_run(retrieved_memories=[{"content": "anything"}])
        example = _make_example()  # outputs=None

        result = retrieval_hit_rate(run, example)

        assert result["score"] == 1
