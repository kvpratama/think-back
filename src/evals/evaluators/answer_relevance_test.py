"""Tests for the answer_relevance evaluator."""

import uuid
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.prompts import PromptTemplate
from langsmith.schemas import Example, Run

from src.evals.evaluators.answer_relevance import (
    AnswerRelevanceModel,
    _get_llm_judge,
    answer_relevance,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_run(
    response: str = "test answer",
    *,
    with_outputs: bool = True,
) -> Run:
    """Build a minimal Run with the given outputs."""
    outputs: dict[str, Any] | None = None
    if with_outputs:
        outputs = {"retrieved_memories": [], "response": response}
    return Run(
        id=uuid.uuid4(),
        name="test",
        run_type="chain",
        inputs={},
        start_time=datetime.now(),
        outputs=outputs,
    )


def _make_example(
    user_input: str = "What is Python?",
    case_type: str = "happy_path",
    *,
    with_inputs: bool = True,
    with_metadata: bool = True,
) -> Example:
    """Build a minimal Example with inputs and metadata."""
    return Example(
        id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        inputs={"user_input": user_input} if with_inputs else {},
        outputs={"expected_contents": [], "expected_answer_criteria": ""},
        metadata={"case_type": case_type, "notes": ""} if with_metadata else None,
    )


@pytest.fixture(autouse=True)
def _clear_judge_cache() -> None:
    """Clear the lru_cache on _get_llm_judge before each test."""
    _get_llm_judge.cache_clear()


# ---------------------------------------------------------------------------
# answer_relevance — early-exit paths
# ---------------------------------------------------------------------------


class TestAnswerRelevanceEarlyExit:
    """Tests for guard-clause / early-exit branches."""

    async def test_no_run_outputs_returns_score_0(self) -> None:
        run = _make_run(with_outputs=False)
        example = _make_example()

        result = await answer_relevance(run, example)

        assert result.score == 0
        assert "Missing run outputs" in (result.comment or "")

    async def test_no_example_returns_score_0(self) -> None:
        run = _make_run()

        result = await answer_relevance(run, None)

        assert result.score == 0

    async def test_no_example_inputs_returns_score_0(self) -> None:
        run = _make_run()
        example = _make_example(with_inputs=False)

        result = await answer_relevance(run, example)

        assert result.score == 0

    async def test_no_example_metadata_returns_score_0(self) -> None:
        run = _make_run()
        example = _make_example(with_metadata=False)

        result = await answer_relevance(run, example)

        assert result.score == 0

    async def test_empty_question_returns_score_0(self) -> None:
        run = _make_run()
        example = _make_example(user_input="")

        result = await answer_relevance(run, example)

        assert result.score == 0
        assert "Missing question or answer" in (result.comment or "")

    async def test_empty_response_returns_score_0(self) -> None:
        run = _make_run(response="")
        example = _make_example()

        result = await answer_relevance(run, example)

        assert result.score == 0
        assert "Missing question or answer" in (result.comment or "")


# ---------------------------------------------------------------------------
# answer_relevance — LLM judge paths
# ---------------------------------------------------------------------------


class TestAnswerRelevanceLLMJudge:
    """Tests for the LLM-as-judge scoring logic."""

    @patch("src.evals.evaluators.answer_relevance._get_llm_judge")
    @patch("src.evals.evaluators.answer_relevance.get_prompt")
    async def test_relevant_answer_scores_1(
        self, mock_prompt: MagicMock, mock_get_judge: MagicMock
    ) -> None:
        mock_prompt.return_value = PromptTemplate.from_template("{question}\n{answer}")
        mock_judge = AsyncMock()
        mock_judge.ainvoke.return_value = AnswerRelevanceModel(
            reason="Directly addresses the question", score=1
        )
        mock_get_judge.return_value = mock_judge

        run = _make_run(response="Python is a programming language.")
        example = _make_example(user_input="What is Python?")

        result = await answer_relevance(run, example)

        assert result.score == 1
        assert "Directly addresses" in (result.comment or "")

    @patch("src.evals.evaluators.answer_relevance._get_llm_judge")
    @patch("src.evals.evaluators.answer_relevance.get_prompt")
    async def test_irrelevant_answer_scores_0(
        self, mock_prompt: MagicMock, mock_get_judge: MagicMock
    ) -> None:
        mock_prompt.return_value = PromptTemplate.from_template("{question}\n{answer}")
        mock_judge = AsyncMock()
        mock_judge.ainvoke.return_value = AnswerRelevanceModel(reason="Off-topic", score=0)
        mock_get_judge.return_value = mock_judge

        run = _make_run(response="The weather is nice today.")
        example = _make_example(user_input="What is Python?")

        result = await answer_relevance(run, example)

        assert result.score == 0
        assert "Off-topic" in (result.comment or "")

    @patch("src.evals.evaluators.answer_relevance._get_llm_judge")
    @patch("src.evals.evaluators.answer_relevance.get_prompt")
    async def test_unexpected_response_type_scores_0(
        self, mock_prompt: MagicMock, mock_get_judge: MagicMock
    ) -> None:
        mock_prompt.return_value = PromptTemplate.from_template("{question}\n{answer}")
        mock_judge = AsyncMock()
        mock_judge.ainvoke.return_value = "unexpected string"
        mock_get_judge.return_value = mock_judge

        run = _make_run()
        example = _make_example()

        result = await answer_relevance(run, example)

        assert result.score == 0
        assert "No response from LLM" in (result.comment or "")

    @patch("src.evals.evaluators.answer_relevance._get_llm_judge")
    @patch("src.evals.evaluators.answer_relevance.get_prompt")
    async def test_comment_includes_case_type(
        self, mock_prompt: MagicMock, mock_get_judge: MagicMock
    ) -> None:
        mock_prompt.return_value = PromptTemplate.from_template("{question}\n{answer}")
        mock_judge = AsyncMock()
        mock_judge.ainvoke.return_value = AnswerRelevanceModel(reason="Good", score=1)
        mock_get_judge.return_value = mock_judge

        run = _make_run()
        example = _make_example(case_type="edge_case")

        result = await answer_relevance(run, example)

        assert "[edge_case]" in (result.comment or "")
