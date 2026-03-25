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

# from src.evals.evaluators.answer_faithfulness import answer_faithfulness
from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from langsmith.evaluation import aevaluate

from src.evals.evaluators.retrieval_hit_rate import retrieval_hit_rate

load_dotenv()


async def run_pipeline(inputs: dict) -> dict:
    """
    Adapter that calls your real LangGraph pipeline and normalises
    the output into the shape the evaluators expect.
    """

    from src.agent.graph import build_graph

    config: RunnableConfig = {"configurable": {"thread_id": "evals_thread_id"}}
    graph = build_graph()
    response = await graph.ainvoke(inputs, config=config)
    return {"retrieved_memories": response["memories"], "answer": response["response"]}


# ---------------------------------------------------------------------------
# Run the evaluation
# ---------------------------------------------------------------------------
async def main():
    results = await aevaluate(
        run_pipeline,
        data="thinkback-eval",
        evaluators=[
            retrieval_hit_rate,
            # answer_faithfulness,
        ],
        experiment_prefix="thinkback",
        metadata={
            "description": "Retrieval hit rate + answer faithfulness.",
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


def run():
    import asyncio

    asyncio.run(main())
