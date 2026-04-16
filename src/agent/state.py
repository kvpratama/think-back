"""State definitions for the ThinkBack agent.

Memory is a TypedDict used by the vector store and tools to represent
memory records. The agent itself uses create_agent's built-in MessagesState.
"""

import datetime
import uuid
from typing import Any, Literal, NotRequired, TypedDict


class Memory(TypedDict):
    """A memory record used within the agent state.

    This type represents a memory either retrieved from the database
    or just saved to it. It includes database fields and optionally
    search similarity scores.
    """

    id: NotRequired[uuid.UUID]
    content: str
    similarity: NotRequired[float]
    metadata: NotRequired[dict[str, Any]]
    source: NotRequired[str | None]
    created_at: NotRequired[datetime.datetime]
    last_reviewed_at: NotRequired[datetime.datetime | None]
    review_count: NotRequired[int]
    test_score_avg: NotRequired[float]


class DuplicateMatch(TypedDict):
    """A duplicate memory match returned by find_duplicates.

    Represents either an exact text match or a semantic similarity match.
    """

    content: str
    similarity: float
    match_type: Literal["exact", "semantic"]
