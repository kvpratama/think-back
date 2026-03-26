"""
evals/run_eval.py

Runs the ThinkBack eval against the 'thinkback-eval' LangSmith dataset.

Wires together:
  - real LangGraph pipeline
  - retrieval_hit_rate  — exact string match, deterministic
  - answer_faithfulness — LLM-as-judge

Usage:
    uv run run-evals

"""

import uuid

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langsmith.evaluation import aevaluate

from src.evals.evaluators.answer_faithfulness import answer_faithfulness
from src.evals.evaluators.answer_relevance import answer_relevance
from src.evals.evaluators.retrieval_hit_rate import retrieval_hit_rate

load_dotenv()


async def run_pipeline(inputs: dict) -> dict:
    """
    Adapter that calls your real LangGraph pipeline and normalises
    the output into the shape the evaluators expect.
    """

    from src.agent.graph import build_graph

    config: RunnableConfig = {"configurable": {"thread_id": f"eval-{uuid.uuid4()}"}}
    graph = build_graph()
    response = await graph.ainvoke(inputs, config=config)
    return {"retrieved_memories": response["memories"], "response": response["response"]}


# ---------------------------------------------------------------------------
# Run the evaluation
# ---------------------------------------------------------------------------
async def main() -> None:
    """
    Run the ThinkBack evaluation pipeline against the 'thinkback-eval' LangSmith dataset.

    Executes the LangGraph pipeline with three evaluators:
      - retrieval_hit_rate: exact string match, deterministic
      - answer_faithfulness: LLM-as-judge for faithfulness
      - answer_relevance: LLM-as-judge for relevance

    Prints a quick summary of scores to stdout and logs full results to LangSmith.

    Returns:
        None
    """
    results = await aevaluate(
        run_pipeline,
        data="thinkback-eval",
        evaluators=[
            retrieval_hit_rate,
            answer_faithfulness,
            answer_relevance,
        ],
        experiment_prefix="thinkback",
        metadata={
            "description": "Retrieval hit rate + answer faithfulness + answer relevance.",
        },
        max_concurrency=1,
    )

    # Print a quick summary to stdout
    print("\n── Eval Results ──────────────────────────────")
    async for result in results:
        ex = result.get("example", {})
        fb = result.get("evaluation_results", {}).get("results", [])
        if ex.inputs:
            query = ex.inputs.get("user_input", "?")
            scores = {r.key: r.score for r in fb}
            print(f"  {query[:50]:<50}  {scores}")

    print("\nView full results in LangSmith.")


def run() -> None:
    """
    Entry point for running evaluations via `uv run run-evals`.

    Synchronously executes the async main() function using asyncio.run().

    Returns:
        None
    """
    import asyncio

    asyncio.run(main())
