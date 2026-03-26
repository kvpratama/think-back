"""Eval dataset definitions for the ThinkBack RAG pipeline.

Contains the shared types, memory strings, and hand-crafted examples used by
both the seed script (seed_dataset.py) and the evaluation runner.

Content-based design — no UUIDs:
  - expected_contents uses exact page_content strings from the retriever
  - Survives database reseeds without any changes to this file
  - expected_answer_criteria is a plain-English rubric for the LLM judge

Schema:
  inputs:
    user_input             — the user's /ask message

  outputs:
    expected_contents      — list of exact page_content strings that should
                             appear in retrieved_docs. evaluator checks
                             set membership (exact string match).
    expected_answer_criteria — plain-English rubric. LLM judge reads this
                               alongside the generated answer and scores 0/1.

  metadata:
    case_type              — "happy_path" | "edge_case"
    match_mode             — "exact" (default) | "any" (skip retrieval check,
                             any retrieved memory is acceptable)
    notes                  — why this example exists
    version                — dataset version this example was introduced in
"""

# ruff: noqa: E501

from typing import TypedDict


class ExampleInput(TypedDict):
    """Input fields for an eval example."""

    user_input: str


class ExampleOutput(TypedDict):
    """Expected output fields for an eval example."""

    expected_contents: list[str]
    expected_answer_criteria: str


class ExampleMetadata(TypedDict):
    """Metadata fields for an eval example."""

    case_type: str
    match_mode: str
    notes: str
    version: str


class EvalExample(TypedDict):
    """A single evaluation example for the ThinkBack dataset."""

    inputs: ExampleInput
    outputs: ExampleOutput
    metadata: ExampleMetadata


DATASET_TAG: str = "v1"  # bump to "v2", "v3" etc. when cutting a new version
DATASET_NAME: str = "thinkback-eval"
DATASET_DESCRIPTION: str = (
    "Hand-crafted eval examples for ThinkBack RAG pipeline. "
    "Content-based — no UUIDs. Survives database reseeds. "
    "Covers retrieval quality and answer faithfulness across happy path and edge cases. "
    "Versioned via LangSmith native tags — evaluate with as_of='v1' etc."
)

# ---------------------------------------------------------------------------
# Exact page_content strings as returned by the retriever.
# These are the stable identifiers — not UUIDs.
#
# Rule: if you edit memory content in Supabase, update the matching
# string here and re-run the seed script to recreate the dataset.
# ---------------------------------------------------------------------------
MEMORIES: dict[str, str] = {
    "fitzgerald": "Writer F. Scott Fitzgerald on what makes a brilliant mind: The test of a first-rate intellect is the ability to hold two opposing ideas in your head at the same time, and still retain the ability to function.",
    "youthful": "When you are young, do what older people wish they could do.",
    "mood": "Mood Follows Action",
    "not_too_late": "It is not too late to do what you want to do—if you stop waiting for the time to be right.",
    "mackey": 'John Mackey: "Life is too short to not follow your heart"',
    "helpfulness": "Many of the good things in life are downstream from being helpful and useful to others. What can you do today to be useful to someone else?",
    "cs50": "CS50x Final Class: What ultimately matters in this course is not so much where you end up relative to your classmates but where you end up relative to yourself when you began",
    "goals": "Goals are for people who care about winning once. Systems are for people who care about winning repeatedly.",
    "pain": "Pain is temporary. It may last a minute, or an hour, or a day, or a year, but eventually it will subside and something else will take its place. If I quit, however, it lasts forever. Lance Armstrong",
    "priorities": "Think about what's important to you. Spend some time—real, uninterrupted time—thinking about what's important to you, what your priorities are. Then, work toward that and forsake all the others. It's not enough to wish and hope. One must act—and act right",
    "coelho": "Paulo Coelho:“One day you will wake up and there won’t be any more time to do the things you’ve always wanted. Do it now.”",
    "consistency": "Consistency beats intensity",
    "standards": "Life is harder when you expect a lot of the world and little of yourself. Life is easier when you expect a lot of yourself and little of the world. High standards, low expectations.",
    "garcia": "“The way I feel about it is that if you read a paper or listen to the news, what do you get? It’s negative. All you get is death, war, riots, unpleasantness. I think that if you tune yourself away from that, it’s possible for you to rediscover what it is that’s positive about being alive. And I think it’s that consciousness that’s essential for survival. It’s like, the more that we believe the negative reality, the more real it is. For those people that don’t give it any energy, that don’t play the game, the game is non-existent. It’s only as strong as the people who are willing to let themselves be slaves to it.” Jerry Garcia 1971 |",
}

EXAMPLES: list[EvalExample] = [
    # ── HAPPY PATH ──────────────────────────────────────────────────────────
    {
        "inputs": {
            "user_input": "what do I know about habits?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["consistency"], MEMORIES["goals"]],
            "expected_answer_criteria": "Answer references consistency or systems thinking. Does not invent habit advice not present in the memories.",
        },
        "metadata": {
            "case_type": "happy_path",
            "match_mode": "exact",
            "notes": "Core retrieval case. Two strongly relevant memories should both surface.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what have I saved about taking action?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["mood"], MEMORIES["not_too_late"], MEMORIES["coelho"]],
            "expected_answer_criteria": "Answer emphasises acting now rather than waiting. Cites at least two of the three relevant memories.",
        },
        "metadata": {
            "case_type": "happy_path",
            "match_mode": "exact",
            "notes": "Multiple highly relevant memories. Tests whether retrieval surfaces more than one.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what do I know about resilience and pain?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["pain"]],
            "expected_answer_criteria": "Answer paraphrases the Lance Armstrong quote about pain being temporary. Does not hallucinate other resilience frameworks.",
        },
        "metadata": {
            "case_type": "happy_path",
            "match_mode": "exact",
            "notes": "Single strongly relevant memory. Key faithfulness check.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what do I know about intelligence and thinking?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["fitzgerald"]],
            "expected_answer_criteria": "Answer references holding opposing ideas in mind simultaneously. Cites Fitzgerald as the source.",
        },
        "metadata": {
            "case_type": "happy_path",
            "match_mode": "exact",
            "notes": "Tests source attribution — the Fitzgerald quote has a named author.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what have I saved about following your dreams?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["mackey"], MEMORIES["coelho"], MEMORIES["not_too_late"]],
            "expected_answer_criteria": "Answer captures urgency around pursuing what matters. References at least two of the three memories.",
        },
        "metadata": {
            "case_type": "happy_path",
            "match_mode": "exact",
            "notes": "Broad motivational query with several relevant memories.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what do I know about systems vs goals?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["goals"]],
            "expected_answer_criteria": "Answer directly contrasts goals with systems using the saved memory. Does not add external frameworks like OKRs or James Clear unless present in a memory.",
        },
        "metadata": {
            "case_type": "happy_path",
            "match_mode": "exact",
            "notes": "Exact-match query. Tests faithfulness against popular hallucinated frameworks.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "tell me about the news and media",
        },
        "outputs": {
            "expected_contents": [MEMORIES["garcia"]],
            "expected_answer_criteria": "Answer reflects Jerry Garcia's view on the negativity of news. Does not add modern media criticism not in the memory.",
        },
        "metadata": {
            "case_type": "happy_path",
            "match_mode": "exact",
            "notes": "Niche topic with exactly one relevant memory. Tests precision.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what do I know about standards and expectations?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["standards"]],
            "expected_answer_criteria": "Answer captures the 'high standards, low expectations' framing. Does not conflate with the CS50 memory about relative progress.",
        },
        "metadata": {
            "case_type": "happy_path",
            "match_mode": "exact",
            "notes": "Tests that retrieval doesn't confuse two adjacent self-improvement memories.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what do I know about being helpful to others?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["helpfulness"]],
            "expected_answer_criteria": "Answer references the idea that good things come from being useful to others. Preserves the 'what can you do today' framing.",
        },
        "metadata": {
            "case_type": "happy_path",
            "match_mode": "exact",
            "notes": "Single relevant memory. Tests faithfulness to a reflective/question-style memory.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what have I saved about learning and personal progress?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["cs50"], MEMORIES["consistency"]],
            "expected_answer_criteria": "Answer talks about progress relative to one's past self and/or the value of consistency. Does not invent learning science concepts.",
        },
        "metadata": {
            "case_type": "happy_path",
            "match_mode": "exact",
            "notes": "Two relevant memories with different angles on learning.",
            "version": DATASET_TAG,
        },
    },
    # ── EDGE CASES ───────────────────────────────────────────────────────────
    {
        "inputs": {
            "user_input": "what do I know about sleep?",
        },
        "outputs": {
            "expected_contents": [],
            "expected_answer_criteria": "Answer honestly says there are no saved memories about sleep. Does not hallucinate sleep advice.",
        },
        "metadata": {
            "case_type": "edge_case",
            "match_mode": "exact",
            "notes": "No-match case. Retriever will return low-similarity results — faithfulness evaluator must catch hallucination.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "motivation",
        },
        "outputs": {
            "expected_contents": [MEMORIES["mood"], MEMORIES["pain"]],
            "expected_answer_criteria": "Answer makes sense as a response to a single-word query. References at least one relevant memory. Does not lecture the user.",
        },
        "metadata": {
            "case_type": "edge_case",
            "match_mode": "exact",
            "notes": "Minimal query — tests whether retrieval still works with no surrounding context.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what do I know about the meaning of life and happiness?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["helpfulness"], MEMORIES["mackey"]],
            "expected_answer_criteria": "Answer stays grounded in the saved memories. Does not import philosophical frameworks like Aristotle or Maslow not present in the memories.",
        },
        "metadata": {
            "case_type": "edge_case",
            "match_mode": "exact",
            "notes": "Broad philosophical query — high hallucination risk.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what do I know about time management and productivity?",
        },
        "outputs": {
            "expected_contents": [
                MEMORIES["priorities"],
                MEMORIES["goals"],
                MEMORIES["consistency"],
            ],
            "expected_answer_criteria": "Answer references priorities, systems, and/or consistency. Does not add productivity frameworks like GTD, Pomodoro, or time-blocking unless present in a memory.",
        },
        "metadata": {
            "case_type": "edge_case",
            "match_mode": "exact",
            "notes": "LLMs reliably hallucinate productivity frameworks here. Key faithfulness test.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what has Lance Armstrong said?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["pain"]],
            "expected_answer_criteria": "Answer attributes the pain quote correctly to Lance Armstrong. Does not fabricate other Armstrong quotes or cycling references.",
        },
        "metadata": {
            "case_type": "edge_case",
            "match_mode": "exact",
            "notes": "Named entity query. Tests source attribution.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "tell me something I haven't thought about in a while",
        },
        "outputs": {
            "expected_contents": [],  # any memory is a valid hit — skip exact retrieval check
            "expected_answer_criteria": "Answer returns at least one memory from the saved set. Does not refuse or say it can't help.",
        },
        "metadata": {
            "case_type": "edge_case",
            "match_mode": "any",  # evaluator should skip exact retrieval check for this example
            "notes": "Vague surfacing-style query. Any memory is a valid hit.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "I want to start a business, what do I know?",
        },
        "outputs": {
            "expected_contents": [
                MEMORIES["helpfulness"],
                MEMORIES["goals"],
                MEMORIES["priorities"],
            ],
            "expected_answer_criteria": "Answer applies saved knowledge to the business framing without hallucinating startup advice not present in the memories.",
        },
        "metadata": {
            "case_type": "edge_case",
            "match_mode": "exact",
            "notes": "Contextualised query. Tests applying memories to a new frame vs. hallucinating domain knowledge.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what did Paulo Coelho say about time?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["coelho"]],
            "expected_answer_criteria": "Answer accurately paraphrases the Paulo Coelho memory. Does not fabricate other Coelho quotes.",
        },
        "metadata": {
            "case_type": "edge_case",
            "match_mode": "exact",
            "notes": "Specific author query. Tests faithfulness prevents fabricated quotes.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "what do I know about quitting?",
        },
        "outputs": {
            "expected_contents": [MEMORIES["pain"]],
            "expected_answer_criteria": "Answer references the 'if I quit it lasts forever' framing from the Lance Armstrong memory. Does not add generic quitting advice beyond what's in the memory.",
        },
        "metadata": {
            "case_type": "edge_case",
            "match_mode": "exact",
            "notes": "Semantic inference — the word 'quitting' doesn't appear verbatim in the best matching memory.",
            "version": DATASET_TAG,
        },
    },
    {
        "inputs": {
            "user_input": "tell me about Jerry Garcia",
        },
        "outputs": {
            "expected_contents": [MEMORIES["garcia"]],
            "expected_answer_criteria": "Answer only reflects what is saved about Garcia (his view on news/negativity). Does not add biographical facts about the Grateful Dead or his personal life.",
        },
        "metadata": {
            "case_type": "edge_case",
            "match_mode": "exact",
            "notes": "Famous person query — high hallucination risk for biographical facts not in the memory.",
            "version": DATASET_TAG,
        },
    },
]
