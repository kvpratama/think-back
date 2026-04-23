"""
evals/evaluators/answer_faithfulness.py

Evaluator: answer_faithfulness
Method:    LLM-as-judge with config-driven jury + negative veto
Score:     1 if ALL judges score the answer as fully grounded in the retrieved memories
           0 if ANY judge scores the answer as unfaithful (negative veto)

Jury is configured via Settings.eval_jury_judges (EVAL_JURY_JUDGES in .env).
Each entry is a JSON object. The api_key_field field is the NAME of an existing
SecretStr field on Settings — no new secrets need to be added to config.py.

  EVAL_JURY_JUDGES='[
    {"model": "gpt-4o", "provider": "openai", "api_key_field": "openai_api_key", "base_url": ""},
    {"model": "gemini", "provider": "gemini", "api_key_field": "gemini_api_key", "base_url": ""}
  ]'

  api_key_field must match a SecretStr attribute name on the Settings class
  (e.g. "openai_api_key", "gemini_api_key"). Adding a new provider requires
  adding its SecretStr to Settings, then referencing it here.

  To add a third judge: append a JSON object to EVAL_JURY_JUDGES — no code changes.

  run.outputs expected shape:
    {
      "retrieved_memories": [{"content": "...", "metadata": {...}}, ...],
      "response": "..."
    }

  example.outputs expected shape:
    {
      "expected_contents": [...],
      "expected_answer_criteria": "plain-English rubric string"
    }
  example.metadata expected shape:
    {
      "case_type": "happy_path" | "edge_case",
      "notes": "..."
    }
"""

import asyncio
import json
import re
from functools import lru_cache
from typing import Any, Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage
from langchain_core.runnables import Runnable
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field

from src.core.config import get_settings

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AnswerFaithfulnessModel(BaseModel):
    """Structured output of the evaluation."""

    reason: str = Field(description="Rationale that explains the score.")
    score: Literal[0, 1] = Field(
        description="1 if the answer is fully grounded in the retrieved memories, 0 otherwise."
    )


class JudgeConfig(BaseModel):
    """Shape of each entry in EVAL_JURY_JUDGES."""

    model: str
    provider: str
    api_key_field: str  # must match a SecretStr attribute name on Settings
    base_url: str = ""


# ---------------------------------------------------------------------------
# Jury construction — built from Settings, cached after first call
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _build_jury() -> list[tuple[str, Runnable]]:
    """
    Parse Settings.eval_jury_judges and return cached (label, runnable) pairs.

    api_key_field is resolved against the Settings instance so all secret
    handling stays inside Pydantic — no raw os.environ calls needed.
    """
    settings = get_settings()

    raw = settings.eval_jury_judges
    if not raw:
        raise ValueError(
            "Settings.eval_jury_judges (EVAL_JURY_JUDGES in .env) is not set. "
            "Add a JSON array of judge config objects."
        )

    # Collapse newlines/extra whitespace — multiline .env values break json.loads.
    # JSON is whitespace-insensitive so this is always safe.
    raw = " ".join(raw.split())

    try:
        parsed: list[dict[str, Any]] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"EVAL_JURY_JUDGES is not valid JSON: {exc}\nReceived (normalised): {raw!r}"
        ) from exc

    if len(parsed) < 2:
        raise ValueError("EVAL_JURY_JUDGES must contain at least 2 judges for jury voting.")

    jury: list[tuple[str, Runnable]] = []
    for entry in parsed:
        cfg = JudgeConfig(**entry)

        # Resolve api_key_field against the Settings instance.
        # This keeps secret access consistent with the rest of the codebase.

        # Detect the common mistake of pasting the raw secret instead of the field name.
        # A valid field name is short, lowercase, and contains no special characters.
        if len(cfg.api_key_field) > 64 or not re.match(r"^[a-z][a-z0-9_]*$", cfg.api_key_field):
            raise ValueError(
                f"api_key_field for model '{cfg.model}' looks like a raw secret key, not a "
                f"field name. Set it to the Settings attribute name that holds the key "
                f"(e.g. 'openai_api_key'), not the key value itself."
            )

        valid_fields = [
            k for k, v in settings.model_fields.items() if "SecretStr" in str(v.annotation)
        ]

        secret = getattr(settings, cfg.api_key_field, None)
        if secret is None or cfg.api_key_field not in valid_fields:
            raise ValueError(
                f"api_key_field '{cfg.api_key_field}' is not a SecretStr field on Settings. "
                f"Available SecretStr fields: {valid_fields}"
            )

        llm = init_chat_model(
            model=cfg.model,
            model_provider=cfg.provider,
            api_key=secret.get_secret_value(),
            base_url=cfg.base_url or None,
            temperature=0,
        ).with_structured_output(AnswerFaithfulnessModel)

        jury.append((cfg.model, llm))

    return jury


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_memories(retrieved_docs: list[dict]) -> str:
    """Format retrieved memories into a readable string for the judge prompt.

    Args:
        retrieved_docs: List of retrieved memory documents, each containing a 'content' key.

    Returns:
        A formatted string with numbered memory entries, or a placeholder message
        if no memories were retrieved.
    """
    if not retrieved_docs:
        return "(no memories were retrieved)"
    return "\n\n".join(
        f"[Memory {i + 1}]: {doc['content']}" for i, doc in enumerate(retrieved_docs)
    )


async def _invoke_judge(
    label: str, judge: Runnable, messages: list[BaseMessage]
) -> tuple[str, int, str]:
    """Invoke one judge. Returns (label, score, reason). Defaults to 0 on failure."""
    try:
        response = await judge.ainvoke(messages)
        if isinstance(response, AnswerFaithfulnessModel):
            return label, response.score, response.reason
        return label, 0, "unexpected response type"
    except Exception as exc:  # noqa: BLE001
        return label, 0, f"invocation failed — {exc}"


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------


async def answer_faithfulness(run: Run, example: Example | None) -> EvaluationResult:
    """
    Config-driven jury with negative veto.

    All judges run in parallel. If ANY judge returns 0 (unfaithful), the
    final score is 0. This asymmetric rule prioritises catching hallucinations
    (high TNR) over never false-flagging good answers.

    To add or remove a judge, update EVAL_JURY_JUDGES in .env — no code changes needed.
    """
    from src.core.prompts import get_prompt

    if not run.outputs or not example or not example.outputs or not example.metadata:
        return EvaluationResult(
            key="answer_faithfulness",
            score=0,
            comment="No outputs found",
        )

    retrieved_docs = run.outputs.get("retrieved_memories", [])
    answer = run.outputs.get("response", "")
    criteria = example.outputs.get("expected_answer_criteria", "")
    case_type = example.metadata.get("case_type", "")

    prompt_template = get_prompt("thinkback-judge-faithfulness")
    prompt_value = prompt_template.invoke(
        {
            "retrieved_memories": _format_memories(retrieved_docs),
            "answer": answer,
            "criteria": criteria,
        }
    )
    messages = prompt_value.to_messages()

    results: list[tuple[str, int, str]] = await asyncio.gather(
        *[_invoke_judge(label, judge, messages) for label, judge in _build_jury()]
    )

    # Negative veto — any 0 overrides all 1s
    vetoes = [(label, reason) for label, score, reason in results if score == 0]
    final_score = 0 if vetoes else 1

    verdicts = "\n".join(
        f"{label}={'PASS' if score == 1 else 'VETO'}: {reason}" for label, score, reason in results
    )
    disagreement = final_score == 0 and any(score == 1 for _, score, _ in results)
    veto_summary = f"\nVETOED by: {', '.join(label for label, _ in vetoes)}" if disagreement else ""

    return EvaluationResult(
        key="answer_faithfulness",
        score=final_score,
        comment=f"[{case_type}]\n{verdicts}{veto_summary}",
    )
