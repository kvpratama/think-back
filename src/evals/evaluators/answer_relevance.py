"""
evals/evaluators/answer_relevance.py

Evaluator: answer_relevance
Method:    LLM-as-judge
Score:     1 if the answer directly addresses the user's question
           0 if the answer is off-topic, evasive, or answers a different question

Evaluator signature follows LangSmith's (run, example) -> dict convention.

  run.outputs expected shape:
    {
      "retrieved_memories": [...],
      "response": "..."
    }

  example.inputs expected shape:
    {
      "user_input": "..."
    }

  example.outputs expected shape:
    {
      "case_type": "happy_path" | "edge_case",
      "notes": "..."
    }

NOTE: Unlike answer_faithfulness, this evaluator reads from example.inputs
(the question) rather than example.outputs — relevance is question-driven,
not rubric-driven.
"""

from functools import lru_cache
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.runnables import Runnable
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.prompts import get_prompt


class AnswerRelevanceModel(BaseModel):
    """Structured output of the evaluation."""

    reason: str = Field(description="Rationale that explains the score.")
    score: Literal[0, 1] = Field(
        description="1 if the answer directly addresses the question, 0 otherwise."
    )


@lru_cache(maxsize=1)
def _get_llm_judge() -> Runnable:
    """Lazily build and cache the LLM judge so env vars are not read at import time."""
    settings = get_settings()
    return init_chat_model(
        model=settings.eval_llm_model,
        model_provider=settings.eval_llm_provider,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.eval_llm_provider_base_url,
        temperature=0,
    ).with_structured_output(AnswerRelevanceModel)


async def answer_relevance(run: Run, example: Example | None) -> EvaluationResult:
    """
    LLM-as-judge evaluator. Sends the user's question and the generated
    answer to an LLM and asks it to score 0 or 1 based on whether the
    answer is relevant to what was asked.

    Deliberately does NOT use expected_answer_criteria — relevance is
    judged against the question alone, not a rubric. This makes it
    complementary to answer_faithfulness rather than redundant.
    """

    if not run.outputs or not example or not example.inputs or not example.metadata:
        return EvaluationResult(
            key="answer_relevance",
            score=0,
            comment="Missing run outputs or example inputs",
        )

    question = example.inputs.get("user_input", "")
    answer = run.outputs.get("response", "")
    case_type = example.metadata.get("case_type", "")

    if not question or not answer:
        return EvaluationResult(
            key="answer_relevance",
            score=0,
            comment="Missing question or answer",
        )

    prompt_template = get_prompt("thinkback-judge-relevance")
    prompt_value = prompt_template.invoke({"question": question, "answer": answer})

    response = await _get_llm_judge().ainvoke(prompt_value.to_messages())

    if isinstance(response, AnswerRelevanceModel):
        score = response.score
        reason = response.reason
    else:
        score = 0
        reason = "No response from LLM"

    return EvaluationResult(
        key="answer_relevance",
        score=score,
        comment=f"[{case_type}] {reason}",
    )
