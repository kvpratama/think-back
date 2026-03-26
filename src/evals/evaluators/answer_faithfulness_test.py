"""Tests for the answer_faithfulness evaluator."""

import json
import uuid
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langsmith.schemas import Example, Run
from pydantic import SecretStr

from src.evals.evaluators.answer_faithfulness import (
    AnswerFaithfulnessModel,
    _build_jury,
    _format_memories,
    _invoke_judge,
    answer_faithfulness,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jury_json(judges: list[dict[str, str]]) -> str:
    """Serialize a list of judge config dicts to the JSON string format."""
    return json.dumps(judges)


def _two_judges(**overrides: str) -> str:
    """Return a valid 2-judge JSON config, optionally overriding fields."""
    base = [
        {"model": "gpt-4o", "provider": "openai", "api_key_field": "openai_api_key"},
        {"model": "gemini", "provider": "gemini", "api_key_field": "gemini_api_key"},
    ]
    for judge in base:
        judge.update(overrides)
    return _jury_json(base)


def _fake_settings(
    jury_json: str = "",
    extra_attrs: dict[str, Any] | None = None,
) -> MagicMock:
    """Build a mock Settings object whose model_fields mirrors the real ones."""
    settings = MagicMock()
    settings.eval_jury_judges = jury_json

    # Provide SecretStr values for known secret fields
    settings.openai_api_key = SecretStr("sk-openai-fake")
    settings.gemini_api_key = SecretStr("sk-gemini-fake")

    # model_fields that mirrors real Settings SecretStr annotations
    settings.model_fields = {
        "openai_api_key": MagicMock(annotation=SecretStr),
        "gemini_api_key": MagicMock(annotation=SecretStr),
        "supabase_url": MagicMock(annotation=SecretStr),
        "supabase_key": MagicMock(annotation=SecretStr),
        "telegram_bot_token": MagicMock(annotation=SecretStr),
        "eval_llm_api_key": MagicMock(annotation=SecretStr),
        "llm_model": MagicMock(annotation=str),
        "llm_provider": MagicMock(annotation=str),
    }

    if extra_attrs:
        for k, v in extra_attrs.items():
            setattr(settings, k, v)

    return settings


def _make_run(
    retrieved_memories: list[dict] | None = None,
    response: str = "test answer",
) -> Run:
    """Build a minimal Run with the given outputs."""
    outputs: dict[str, Any] | None = None
    if retrieved_memories is not None:
        outputs = {"retrieved_memories": retrieved_memories, "response": response}
    return Run(
        id=uuid.uuid4(),
        name="test",
        run_type="chain",
        inputs={},
        start_time=datetime.now(),
        outputs=outputs,
    )


def _make_example(
    expected_answer_criteria: str = "Answer must be grounded.",
    expected_contents: list[str] | None = None,
    case_type: str = "happy_path",
) -> Example:
    """Build a minimal Example with outputs and metadata."""
    return Example(
        id=uuid.uuid4(),
        dataset_id=uuid.uuid4(),
        inputs={"user_input": "test question"},
        outputs={
            "expected_contents": expected_contents or [],
            "expected_answer_criteria": expected_answer_criteria,
        },
        metadata={"case_type": case_type, "notes": ""},
    )


@pytest.fixture(autouse=True)
def _clear_jury_cache() -> None:
    """Clear the lru_cache on _build_jury before each test."""
    _build_jury.cache_clear()


# ---------------------------------------------------------------------------
# _format_memories
# ---------------------------------------------------------------------------


class TestFormatMemories:
    """Tests for the _format_memories helper."""

    def test_empty_list_returns_placeholder(self) -> None:
        assert _format_memories([]) == "(no memories were retrieved)"

    def test_single_memory(self) -> None:
        result = _format_memories([{"content": "Memory one"}])
        assert "[Memory 1]: Memory one" in result

    def test_multiple_memories_numbered(self) -> None:
        result = _format_memories([{"content": "A"}, {"content": "B"}])
        assert "[Memory 1]: A" in result
        assert "[Memory 2]: B" in result


# ---------------------------------------------------------------------------
# _build_jury — validation
# ---------------------------------------------------------------------------


class TestBuildJuryValidation:
    """Tests for _build_jury input validation (no LLM calls)."""

    @patch("src.evals.evaluators.answer_faithfulness.get_settings")
    def test_empty_jury_config_raises(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _fake_settings(jury_json="")

        with pytest.raises(ValueError, match="not set"):
            _build_jury()

    @patch("src.evals.evaluators.answer_faithfulness.get_settings")
    def test_invalid_json_raises(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _fake_settings(jury_json="not json")

        with pytest.raises(ValueError, match="not valid JSON"):
            _build_jury()

    @patch("src.evals.evaluators.answer_faithfulness.get_settings")
    def test_fewer_than_two_judges_raises(self, mock_get: MagicMock) -> None:
        one_judge = _jury_json(
            [{"model": "gpt-4o", "provider": "openai", "api_key_field": "openai_api_key"}]
        )
        mock_get.return_value = _fake_settings(jury_json=one_judge)

        with pytest.raises(ValueError, match="at least 2 judges"):
            _build_jury()

    @patch("src.evals.evaluators.answer_faithfulness.get_settings")
    def test_raw_secret_as_field_name_raises(self, mock_get: MagicMock) -> None:
        bad_judges = _jury_json(
            [
                {"model": "gpt-4o", "provider": "openai", "api_key_field": "sk-AAAA1234567890"},
                {"model": "gemini", "provider": "gemini", "api_key_field": "gemini_api_key"},
            ]
        )
        mock_get.return_value = _fake_settings(jury_json=bad_judges)

        with pytest.raises(ValueError, match="raw secret key"):
            _build_jury()

    @patch("src.evals.evaluators.answer_faithfulness.get_settings")
    def test_nonexistent_field_raises(self, mock_get: MagicMock) -> None:
        bad_judges = _jury_json(
            [
                {"model": "gpt-4o", "provider": "openai", "api_key_field": "no_such_field"},
                {"model": "gemini", "provider": "gemini", "api_key_field": "gemini_api_key"},
            ]
        )
        settings = _fake_settings(jury_json=bad_judges)
        # getattr will return MagicMock for any attr by default; set this one to None
        settings.no_such_field = None
        # Also ensure it falls through the "not in valid_fields" check
        mock_get.return_value = settings

        with pytest.raises(ValueError, match="not a SecretStr field"):
            _build_jury()

    @patch("src.evals.evaluators.answer_faithfulness.get_settings")
    def test_non_secret_field_raises(self, mock_get: MagicMock) -> None:
        """api_key_field pointing to a non-SecretStr field must be rejected."""
        bad_judges = _jury_json(
            [
                {"model": "gpt-4o", "provider": "openai", "api_key_field": "llm_model"},
                {"model": "gemini", "provider": "gemini", "api_key_field": "gemini_api_key"},
            ]
        )
        settings = _fake_settings(jury_json=bad_judges)
        # llm_model exists but is str, not SecretStr
        settings.llm_model = "gpt-4o-mini"
        mock_get.return_value = settings

        with pytest.raises(ValueError, match="not a SecretStr field"):
            _build_jury()

    @patch("src.evals.evaluators.answer_faithfulness.init_chat_model")
    @patch("src.evals.evaluators.answer_faithfulness.get_settings")
    def test_valid_config_builds_jury(self, mock_get: MagicMock, mock_init: MagicMock) -> None:
        mock_get.return_value = _fake_settings(jury_json=_two_judges())
        mock_runnable = MagicMock()
        mock_init.return_value.with_structured_output.return_value = mock_runnable

        jury = _build_jury()

        assert len(jury) == 2
        assert jury[0][0] == "gpt-4o"
        assert jury[1][0] == "gemini"
        assert mock_init.call_count == 2


# ---------------------------------------------------------------------------
# _invoke_judge
# ---------------------------------------------------------------------------


class TestInvokeJudge:
    """Tests for the _invoke_judge helper."""

    async def test_successful_invocation(self) -> None:
        mock_judge = AsyncMock()
        mock_judge.ainvoke.return_value = AnswerFaithfulnessModel(reason="Grounded", score=1)

        label, score, reason = await _invoke_judge("gpt-4o", mock_judge, "prompt")

        assert label == "gpt-4o"
        assert score == 1
        assert reason == "Grounded"

    async def test_unexpected_response_type_scores_0(self) -> None:
        mock_judge = AsyncMock()
        mock_judge.ainvoke.return_value = "unexpected string"

        label, score, reason = await _invoke_judge("gpt-4o", mock_judge, "prompt")

        assert score == 0
        assert "unexpected response type" in reason

    async def test_exception_scores_0(self) -> None:
        mock_judge = AsyncMock()
        mock_judge.ainvoke.side_effect = RuntimeError("API timeout")

        label, score, reason = await _invoke_judge("gpt-4o", mock_judge, "prompt")

        assert score == 0
        assert "invocation failed" in reason


# ---------------------------------------------------------------------------
# answer_faithfulness (end-to-end with mocked jury)
# ---------------------------------------------------------------------------


class TestAnswerFaithfulness:
    """Tests for the answer_faithfulness evaluator function."""

    async def test_no_run_outputs_returns_score_0(self) -> None:
        run = _make_run()  # outputs=None
        example = _make_example()

        result = await answer_faithfulness(run, example)

        assert result.score == 0
        assert "No outputs found" in (result.comment or "")

    async def test_no_example_returns_score_0(self) -> None:
        run = _make_run(retrieved_memories=[{"content": "mem"}])

        result = await answer_faithfulness(run, None)

        assert result.score == 0

    @patch("src.evals.evaluators.answer_faithfulness._build_jury")
    async def test_all_judges_pass_scores_1(self, mock_jury: MagicMock) -> None:
        judge_a = AsyncMock()
        judge_a.ainvoke.return_value = AnswerFaithfulnessModel(reason="OK", score=1)
        judge_b = AsyncMock()
        judge_b.ainvoke.return_value = AnswerFaithfulnessModel(reason="OK", score=1)
        mock_jury.return_value = [("judge_a", judge_a), ("judge_b", judge_b)]

        run = _make_run(retrieved_memories=[{"content": "fact"}])
        example = _make_example()

        result = await answer_faithfulness(run, example)

        assert result.score == 1

    @patch("src.evals.evaluators.answer_faithfulness._build_jury")
    async def test_one_veto_scores_0(self, mock_jury: MagicMock) -> None:
        judge_pass = AsyncMock()
        judge_pass.ainvoke.return_value = AnswerFaithfulnessModel(reason="OK", score=1)
        judge_veto = AsyncMock()
        judge_veto.ainvoke.return_value = AnswerFaithfulnessModel(reason="Hallucinated", score=0)
        mock_jury.return_value = [("pass", judge_pass), ("veto", judge_veto)]

        run = _make_run(retrieved_memories=[{"content": "fact"}])
        example = _make_example()

        result = await answer_faithfulness(run, example)

        assert result.score == 0
        assert "VETO" in (result.comment or "")
