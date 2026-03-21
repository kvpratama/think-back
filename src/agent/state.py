"""Agent state definition for LangGraph.

The AgentState is a TypedDict that represents the state of the agent
throughout the graph execution. It inherits from MessagesState which
uses a reducer for the messages list. Other list fields like memories
are overwritten per query.
"""

from typing import Any, Literal, NotRequired, TypedDict

from langgraph.graph import MessagesState


class Memory(TypedDict):
    """A memory record used within the agent state.

    This type represents a memory either retrieved from the database
    or just saved to it. It includes database fields and optionally
    search similarity scores.
    """

    id: str | Any  # Can be UUID from DB or string from search metadata
    content: str
    similarity: NotRequired[float]
    metadata: NotRequired[dict[str, Any]]
    source: NotRequired[str | None]
    created_at: NotRequired[str | Any]
    last_reviewed_at: NotRequired[str | Any | None]
    review_count: NotRequired[int]
    test_score_avg: NotRequired[float]


class AgentState(MessagesState):
    """State of the ThinkBack agent.

    This TypedDict defines the structure of the agent's state throughout
    the LangGraph execution. Fields annotated with a reducer function
    will accumulate values across graph steps.

    Attributes:
        user_input: The raw user input from Telegram (including command prefix).
        cleaned_input: The user input with command prefix stripped.
        intent: The detected intent ('save', 'query', or None).
        memories: List of retrieved or saved memory records. Overwritten per query.
        response: The final response to send to the user.
        error: Any error message that occurred during processing.
    """

    user_input: str
    cleaned_input: str
    intent: Literal["save", "query"] | None
    memories: list[Memory]
    response: str
    error: str | None
