"""
evals/evaluators/answer_faithfulness.py

Evaluator: answer_faithfulness
Method:    LLM-as-judge (claude-sonnet-4 via Anthropic API)
Score:     1 if the answer is fully grounded in the retrieved memories
           0 if the answer adds information not present in the memories

Evaluator signature follows LangSmith's (run, example) -> dict convention.

  run.outputs expected shape:
    {
      "retrieved_memories": [
        {"page_content": "...", "metadata": {...}},
        ...
      ],
      "response": "..."
    }

  example.outputs expected shape:
    {
      "expected_contents": [...],
      "expected_answer_criteria": "plain-English rubric string",
      "case_type": "happy_path" | "edge_case",
      "notes": "..."
    }
"""

from typing import Literal

from langchain.chat_models import init_chat_model
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field

from src.core.config import get_settings

JUDGE_PROMPT = """\
You are evaluating the answer quality of a RAG (retrieval-augmented generation) system \
called ThinkBack. ThinkBack answers questions using ONLY the user's saved personal memories. \
It must never add information from outside those memories.

You will be given:
1. The retrieved memories that were passed to the system
2. The answer the system generated
3. A quality rubric describing what a good answer looks like

Your job: score the answer 1 or 0.

Score 1 if:
- The answer is fully grounded in the retrieved memories
- The answer satisfies the quality rubric
- The answer does not add facts, frameworks, quotes, or advice not present in the memories

Score 0 if:
- The answer includes information not present in the retrieved memories
- The answer hallucinates quotes, biographical facts, or external knowledge
- The answer fails to meet the quality rubric

---

Retrieved memories:
{retrieved_memories}

---

Generated answer:
{answer}

---

Quality rubric:
{criteria}

---

Respond with valid JSON only. No preamble, no explanation outside the JSON.
{{
  "score": 0 or 1,
  "reason": "one sentence explaining your score"
}}
"""


class AnswerFaithfulnessModel(BaseModel):
    """Structured output of the evaluation."""

    reason: str = Field(description="Rationale that explains the score.")
    score: Literal[0, 1] = Field(
        description="1 if the answer is fully grounded in the retrieved memories, 0 otherwise."
    )


def answer_faithfulness(run: Run, example: Example) -> dict:
    """
    LLM-as-judge evaluator. Sends retrieved memories, the generated
    answer, and the expected_answer_criteria rubric to Claude and
    asks it to score 0 or 1.
    """

    if not run.outputs or not example.outputs:
        return {
            "key": "answer_faithfulness",
            "score": 0,
            "comment": "No outputs found",
        }

    settings = get_settings()
    llm_judge = init_chat_model(
        model=settings.eval_llm_model,
        model_provider=settings.eval_llm_provider,
        api_key=settings.openai_api_key.get_secret_value(),
        base_url=settings.eval_llm_provider_base_url,
        temperature=0,
    ).with_structured_output(AnswerFaithfulnessModel)

    retrieved_docs = run.outputs.get("retrieved_memories", [])
    answer = run.outputs.get("response", "")
    criteria = example.outputs.get("expected_answer_criteria", "")
    case_type = example.outputs.get("metadata", {}).get("case_type", "")

    # Format retrieved memories for the judge prompt
    if retrieved_docs:
        retrieved_memories = "\n\n".join(
            f"[Memory {i + 1}]: {doc['content']}" for i, doc in enumerate(retrieved_docs)
        )
    else:
        retrieved_memories = "(no memories were retrieved)"

    prompt = JUDGE_PROMPT.format(
        retrieved_memories=retrieved_memories,
        answer=answer,
        criteria=criteria,
    )

    response = llm_judge.invoke(
        [{"role": "user", "content": prompt}],
    )

    if isinstance(response, AnswerFaithfulnessModel):
        score = response.score
        reason = response.reason
    else:
        score = 0
        reason = "No response from LLM"

    return {
        "key": "answer_faithfulness",
        "score": score,
        "comment": f"[{case_type}] {reason}",
    }
