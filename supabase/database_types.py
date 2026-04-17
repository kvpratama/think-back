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

class PublicMemories(BaseModel):
    content: str = Field(alias="content")
    created_at: datetime.datetime = Field(alias="created_at")
    embedding: list[Any] = Field(alias="embedding")
    id: uuid.UUID = Field(alias="id")
    last_reviewed_at: Optional[datetime.datetime] = Field(alias="last_reviewed_at")
    metadata: Optional[Json[Any]] = Field(alias="metadata")
    review_count: int = Field(alias="review_count")
    source: Optional[str] = Field(alias="source")

class PublicMemoriesInsert(TypedDict):
    content: Annotated[str, Field(alias="content")]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    embedding: Annotated[list[Any], Field(alias="embedding")]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    last_reviewed_at: NotRequired[Annotated[Optional[datetime.datetime], Field(alias="last_reviewed_at")]]
    metadata: NotRequired[Annotated[Optional[Json[Any]], Field(alias="metadata")]]
    review_count: NotRequired[Annotated[int, Field(alias="review_count")]]
    source: NotRequired[Annotated[Optional[str], Field(alias="source")]]

class PublicMemoriesUpdate(TypedDict):
    content: NotRequired[Annotated[str, Field(alias="content")]]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    embedding: NotRequired[Annotated[list[Any], Field(alias="embedding")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    last_reviewed_at: NotRequired[Annotated[Optional[datetime.datetime], Field(alias="last_reviewed_at")]]
    metadata: NotRequired[Annotated[Optional[Json[Any]], Field(alias="metadata")]]
    review_count: NotRequired[Annotated[int, Field(alias="review_count")]]
    source: NotRequired[Annotated[Optional[str], Field(alias="source")]]

class PublicUserSettings(BaseModel):
    confirm_before_save: bool = Field(alias="confirm_before_save")
    created_at: datetime.datetime = Field(alias="created_at")
    id: uuid.UUID = Field(alias="id")
    reminder_time_1: datetime.time = Field(alias="reminder_time_1")
    reminder_time_2: datetime.time = Field(alias="reminder_time_2")
    reminders_enabled: bool = Field(alias="reminders_enabled")
    telegram_chat_id: str = Field(alias="telegram_chat_id")
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
    timezone: NotRequired[Annotated[str, Field(alias="timezone")]]
    updated_at: NotRequired[Annotated[datetime.datetime, Field(alias="updated_at")]]
