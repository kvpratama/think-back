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

class PublicUserSettings(BaseModel):
    created_at: datetime.datetime = Field(alias="created_at")
    id: uuid.UUID = Field(alias="id")
    telegram_chat_id: str = Field(alias="telegram_chat_id")
    timezone: str = Field(alias="timezone")
    updated_at: datetime.datetime = Field(alias="updated_at")

class PublicUserSettingsInsert(TypedDict):
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    telegram_chat_id: Annotated[str, Field(alias="telegram_chat_id")]
    timezone: NotRequired[Annotated[str, Field(alias="timezone")]]
    updated_at: NotRequired[Annotated[datetime.datetime, Field(alias="updated_at")]]

class PublicUserSettingsUpdate(TypedDict):
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    telegram_chat_id: NotRequired[Annotated[str, Field(alias="telegram_chat_id")]]
    timezone: NotRequired[Annotated[str, Field(alias="timezone")]]
    updated_at: NotRequired[Annotated[datetime.datetime, Field(alias="updated_at")]]

class PublicReminderTimes(BaseModel):
    created_at: datetime.datetime = Field(alias="created_at")
    id: uuid.UUID = Field(alias="id")
    time: datetime.time = Field(alias="time")
    user_settings_id: uuid.UUID = Field(alias="user_settings_id")

class PublicReminderTimesInsert(TypedDict):
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    time: Annotated[datetime.time, Field(alias="time")]
    user_settings_id: Annotated[uuid.UUID, Field(alias="user_settings_id")]

class PublicReminderTimesUpdate(TypedDict):
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    time: NotRequired[Annotated[datetime.time, Field(alias="time")]]
    user_settings_id: NotRequired[Annotated[uuid.UUID, Field(alias="user_settings_id")]]

class PublicMemories(BaseModel):
    content: str = Field(alias="content")
    created_at: datetime.datetime = Field(alias="created_at")
    embedding: list[Any] = Field(alias="embedding")
    id: uuid.UUID = Field(alias="id")
    last_reviewed_at: Optional[datetime.datetime] = Field(alias="last_reviewed_at")
    metadata: Optional[Json[Any]] = Field(alias="metadata")
    review_count: int = Field(alias="review_count")
    source: Optional[str] = Field(alias="source")
    user_settings_id: uuid.UUID = Field(alias="user_settings_id")

class PublicMemoriesInsert(TypedDict):
    content: Annotated[str, Field(alias="content")]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    embedding: Annotated[list[Any], Field(alias="embedding")]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    last_reviewed_at: NotRequired[Annotated[Optional[datetime.datetime], Field(alias="last_reviewed_at")]]
    metadata: NotRequired[Annotated[Optional[Json[Any]], Field(alias="metadata")]]
    review_count: NotRequired[Annotated[int, Field(alias="review_count")]]
    source: NotRequired[Annotated[Optional[str], Field(alias="source")]]
    user_settings_id: Annotated[uuid.UUID, Field(alias="user_settings_id")]

class PublicMemoriesUpdate(TypedDict):
    content: NotRequired[Annotated[str, Field(alias="content")]]
    created_at: NotRequired[Annotated[datetime.datetime, Field(alias="created_at")]]
    embedding: NotRequired[Annotated[list[Any], Field(alias="embedding")]]
    id: NotRequired[Annotated[uuid.UUID, Field(alias="id")]]
    last_reviewed_at: NotRequired[Annotated[Optional[datetime.datetime], Field(alias="last_reviewed_at")]]
    metadata: NotRequired[Annotated[Optional[Json[Any]], Field(alias="metadata")]]
    review_count: NotRequired[Annotated[int, Field(alias="review_count")]]
    source: NotRequired[Annotated[Optional[str], Field(alias="source")]]
    user_settings_id: NotRequired[Annotated[uuid.UUID, Field(alias="user_settings_id")]]
