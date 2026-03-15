from __future__ import annotations

import datetime
import uuid
from typing import (
    Annotated,
    Any,
    List,
    Literal,
    NotRequired,
    Optional,
    TypeAlias,
    TypedDict,
)

from pydantic import BaseModel, Field, Json

NetRequestStatus: TypeAlias = Literal["PENDING", "SUCCESS", "ERROR"]

RealtimeEqualityOp: TypeAlias = Literal["eq", "neq", "lt", "lte", "gt", "gte", "in"]

RealtimeAction: TypeAlias = Literal["INSERT", "UPDATE", "DELETE", "TRUNCATE", "ERROR"]

StorageBuckettype: TypeAlias = Literal["STANDARD", "ANALYTICS", "VECTOR"]

AuthFactorType: TypeAlias = Literal["totp", "webauthn", "phone"]

AuthFactorStatus: TypeAlias = Literal["unverified", "verified"]

AuthAalLevel: TypeAlias = Literal["aal1", "aal2", "aal3"]

AuthCodeChallengeMethod: TypeAlias = Literal["s256", "plain"]

AuthOneTimeTokenType: TypeAlias = Literal["confirmation_token", "reauthentication_token", "recovery_token", "email_change_token_new", "email_change_token_current", "phone_change_token"]

AuthOauthRegistrationType: TypeAlias = Literal["dynamic", "manual"]

AuthOauthAuthorizationStatus: TypeAlias = Literal["pending", "approved", "denied", "expired"]

AuthOauthResponseType: TypeAlias = Literal["code"]

AuthOauthClientType: TypeAlias = Literal["public", "confidential"]

PublicTestSessionStatus: TypeAlias = Literal["active", "completed"]

class PublicMemories(BaseModel):
    content: str = Field(alias="content")
    created_at: datetime.datetime = Field(alias="created_at")
    embedding: list[Any] = Field(alias="embedding")
    id: uuid.UUID = Field(alias="id")
    last_reviewed_at: Optional[datetime.datetime] = Field(alias="last_reviewed_at")
    review_count: int = Field(alias="review_count")
    source: Optional[str] = Field(alias="source")
    summary: str = Field(alias="summary")
    test_score_avg: float = Field(alias="test_score_avg")

class PublicMemoriesInsert(TypedDict):
    content: Annotated[str, Field(alias="content")]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    embedding: Annotated[list[Any], Field(alias="embedding")]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    last_reviewed_at: NotRequired[Annotated[datetime.datetime, Field(alias="last_reviewed_at")]]
    review_count: NotRequired[Annotated[int, Field(alias="review_count")]]
    source: NotRequired[Annotated[str, Field(alias="source")]]
    summary: Annotated[str, Field(alias="summary")]
    test_score_avg: NotRequired[Annotated[float, Field(alias="test_score_avg")]]

class PublicMemoriesUpdate(TypedDict):
    content: NotRequired[Annotated[str, Field(alias="content")]]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    embedding: NotRequired[Annotated[list[Any], Field(alias="embedding")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    last_reviewed_at: NotRequired[Annotated[datetime.datetime, Field(alias="last_reviewed_at")]]
    review_count: NotRequired[Annotated[int, Field(alias="review_count")]]
    source: NotRequired[Annotated[str, Field(alias="source")]]
    summary: NotRequired[Annotated[str, Field(alias="summary")]]
    test_score_avg: NotRequired[Annotated[float, Field(alias="test_score_avg")]]

class PublicTags(BaseModel):
    id: int = Field(alias="id")
    name: str = Field(alias="name")

class PublicTagsInsert(TypedDict):
    id: NotRequired[Annotated[int, Field(alias="id")]]
    name: Annotated[str, Field(alias="name")]

class PublicTagsUpdate(TypedDict):
    id: NotRequired[Annotated[int, Field(alias="id")]]
    name: NotRequired[Annotated[str, Field(alias="name")]]

class PublicMemoryTags(BaseModel):
    memory_id: uuid.UUID = Field(alias="memory_id")
    tag_id: int = Field(alias="tag_id")

class PublicMemoryTagsInsert(TypedDict):
    memory_id: Annotated[uuid.UUID, Field(alias="memory_id")]
    tag_id: Annotated[int, Field(alias="tag_id")]

class PublicMemoryTagsUpdate(TypedDict):
    memory_id: NotRequired[Annotated[uuid.UUID, Field(alias="memory_id")]]
    tag_id: NotRequired[Annotated[int, Field(alias="tag_id")]]

class PublicTestSessions(BaseModel):
    completed_at: Optional[datetime.datetime] = Field(alias="completed_at")
    conversation: Json[Any] = Field(alias="conversation")
    created_at: datetime.datetime = Field(alias="created_at")
    id: uuid.UUID = Field(alias="id")
    memory_id: uuid.UUID = Field(alias="memory_id")
    question: str = Field(alias="question")
    score: Optional[int] = Field(alias="score")
    status: PublicTestSessionStatus = Field(alias="status")

class PublicTestSessionsInsert(TypedDict):
    completed_at: NotRequired[Annotated[datetime.datetime, Field(alias="completed_at")]]
    conversation: NotRequired[Annotated[Json[Any], Field(alias="conversation")]]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    memory_id: Annotated[uuid.UUID, Field(alias="memory_id")]
    question: Annotated[str, Field(alias="question")]
    score: NotRequired[Annotated[int, Field(alias="score")]]
    status: NotRequired[Annotated[PublicTestSessionStatus, Field(alias="status")]]

class PublicTestSessionsUpdate(TypedDict):
    completed_at: NotRequired[Annotated[datetime.datetime, Field(alias="completed_at")]]
    conversation: NotRequired[Annotated[Json[Any], Field(alias="conversation")]]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    memory_id: NotRequired[Annotated[uuid.UUID, Field(alias="memory_id")]]
    question: NotRequired[Annotated[str, Field(alias="question")]]
    score: NotRequired[Annotated[int, Field(alias="score")]]
    status: NotRequired[Annotated[PublicTestSessionStatus, Field(alias="status")]]

class PublicUserSettings(BaseModel):
    confirm_before_save: bool = Field(alias="confirm_before_save")
    created_at: datetime.datetime = Field(alias="created_at")
    id: uuid.UUID = Field(alias="id")
    reminder_time_1: datetime.time = Field(alias="reminder_time_1")
    reminder_time_2: datetime.time = Field(alias="reminder_time_2")
    reminders_enabled: bool = Field(alias="reminders_enabled")
    telegram_chat_id: str = Field(alias="telegram_chat_id")
    test_enabled: bool = Field(alias="test_enabled")
    test_time: datetime.time = Field(alias="test_time")
    timezone: str = Field(alias="timezone")
    updated_at: datetime.datetime = Field(alias="updated_at")

class PublicUserSettingsInsert(TypedDict):
    confirm_before_save: NotRequired[Annotated[bool, Field(alias="confirm_before_save")]]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    reminder_time_1: NotRequired[Annotated[datetime.time, Field(alias="reminder_time_1")]]
    reminder_time_2: NotRequired[Annotated[datetime.time, Field(alias="reminder_time_2")]]
    reminders_enabled: NotRequired[Annotated[bool, Field(alias="reminders_enabled")]]
    telegram_chat_id: Annotated[str, Field(alias="telegram_chat_id")]
    test_enabled: NotRequired[Annotated[bool, Field(alias="test_enabled")]]
    test_time: NotRequired[Annotated[datetime.time, Field(alias="test_time")]]
    timezone: NotRequired[Annotated[str, Field(alias="timezone")]]
    updated_at: NotRequired[Annotated[datetime.datetime, Field(alias="updated_at")]]

class PublicUserSettingsUpdate(TypedDict):
    confirm_before_save: NotRequired[Annotated[bool, Field(alias="confirm_before_save")]]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    reminder_time_1: NotRequired[Annotated[datetime.time, Field(alias="reminder_time_1")]]
    reminder_time_2: NotRequired[Annotated[datetime.time, Field(alias="reminder_time_2")]]
    reminders_enabled: NotRequired[Annotated[bool, Field(alias="reminders_enabled")]]
    telegram_chat_id: NotRequired[Annotated[str, Field(alias="telegram_chat_id")]]
    test_enabled: NotRequired[Annotated[bool, Field(alias="test_enabled")]]
    test_time: NotRequired[Annotated[datetime.time, Field(alias="test_time")]]
    timezone: NotRequired[Annotated[str, Field(alias="timezone")]]
    updated_at: NotRequired[Annotated[datetime.datetime, Field(alias="updated_at")]]

class PublicMemoriesWithTags(BaseModel):
    content: Optional[str] = Field(alias="content")
    created_at: Optional[datetime.datetime] = Field(alias="created_at")
    id: Optional[uuid.UUID] = Field(alias="id")
    last_reviewed_at: Optional[datetime.datetime] = Field(alias="last_reviewed_at")
    review_count: Optional[int] = Field(alias="review_count")
    source: Optional[str] = Field(alias="source")
    summary: Optional[str] = Field(alias="summary")
    tags: Optional[List[str]] = Field(alias="tags")
    test_score_avg: Optional[float] = Field(alias="test_score_avg")

class PublicSurfacingCandidates(BaseModel):
    days_since_reviewed: Optional[float] = Field(alias="days_since_reviewed")
    id: Optional[uuid.UUID] = Field(alias="id")
    is_novel: Optional[bool] = Field(alias="is_novel")
    last_reviewed_at: Optional[datetime.datetime] = Field(alias="last_reviewed_at")
    review_count: Optional[int] = Field(alias="review_count")
    reviewed_recently: Optional[bool] = Field(alias="reviewed_recently")
    source: Optional[str] = Field(alias="source")
    summary: Optional[str] = Field(alias="summary")
    test_score_avg: Optional[float] = Field(alias="test_score_avg")
