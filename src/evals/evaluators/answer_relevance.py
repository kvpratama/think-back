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

from typing import Literal

from langchain.chat_models import init_chat_model
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field

from src.core.config import get_settings

JUDGE_PROMPT = """\
You are evaluating the answer quality of a RAG system called ThinkBack. \
ThinkBack answers questions using the user's saved personal memories.

Your job: decide whether the answer actually addresses the question asked.
Focus ONLY on relevance — not on whether the answer is grounded or detailed.

Score 1 if:
- The answer directly responds to what the user asked
- The answer attempts to address the question even if no relevant memories exist
  (e.g. "I don't have any memories about X" is still relevant if X was asked)

Score 0 if:
- The answer responds to a different question than the one asked
- The answer is evasive and avoids the topic entirely
- The answer provides generic information with no connection to the question

---

User question:
{question}

---

Generated answer:
{answer}

"""


class AnswerRelevanceModel(BaseModel):
    """Structured output of the evaluation."""

    reason: str = Field(description="Rationale that explains the score.")
    score: Literal[0, 1] = Field(
        description="1 if the answer directly addresses the question, 0 otherwise."
    )


async def answer_relevance(run: Run, example: Example | None) -> EvaluationResult:
    """
    LLM-as-judge evaluator. Sends the user's question and the generated
    answer to Claude and asks it to score 0 or 1 based on whether the
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

    settings = get_settings()
    llm_judge = init_chat_model(
        model=settings.eval_llm_model,
        model_provider=settings.eval_llm_provider,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.eval_llm_provider_base_url,
        temperature=0,
    ).with_structured_output(AnswerRelevanceModel)

    # Pull from inputs (the question) — not outputs
    run_outputs = run.outputs if hasattr(run, "outputs") else run.get("outputs", {}) or {}
    example_inputs = (
        example.inputs if hasattr(example, "inputs") else example.get("inputs", {}) or {}
    )
    example_metadata = (
        example.metadata if hasattr(example, "metadata") else example.get("metadata", {}) or {}
    )

    question = example_inputs.get("user_input", "")
    answer = run_outputs.get("response", "")
    case_type = example_metadata.get("case_type", "")

    if not question or not answer:
        return EvaluationResult(
            key="answer_relevance",
            score=0,
            comment="Missing question or answer",
        )

    prompt = JUDGE_PROMPT.format(question=question, answer=answer)

    response = await llm_judge.ainvoke(
        [{"role": "user", "content": prompt}],
    )

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
