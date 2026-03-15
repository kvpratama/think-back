-- ============================================================
-- Migration 003: test_sessions
-- Records every Socratic test session for ThinkBack
-- Persists mid-test state so bot restarts don't lose progress
-- (InMemorySaver is ephemeral — this table is the source of truth)
-- ============================================================


-- ------------------------------------------------------------
-- ENUM: test_session_status
-- Constrained set — safer than a free TEXT column
-- Skill ref: schema-data-types — use enum or text+check for constrained values
-- ------------------------------------------------------------
do $$ begin
  create type public.test_session_status as enum ('active', 'completed');
exception
  when duplicate_object then null;  -- idempotent: skip if already exists
end $$;


-- ------------------------------------------------------------
-- TABLE: test_sessions
-- ------------------------------------------------------------
create table if not exists public.test_sessions (
  -- Skill ref: schema-primary-keys — gen_random_uuid() (UUIDv4), sufficient at personal scale
  id            uuid                      primary key default gen_random_uuid(),

  -- FK to memories — cascade delete: if a memory is deleted, its test history goes too
  -- Skill ref: schema-foreign-key-indexes — index this FK explicitly below
  memory_id     uuid                      not null
                  references public.memories(id) on delete cascade,

  -- The opening question posed to the user (derived from memory content, not verbatim)
  question      text                      not null,

  -- Full Socratic conversation as a JSONB array of turns
  -- Shape of each element: {"role": "bot"|"user", "content": "..."}
  -- Skill ref: advanced-jsonb-indexing — add GIN index for future conversation search
  conversation  jsonb                     not null default '[]'::jsonb,

  -- Internal score 1–5; null until session completes
  -- Skill ref: schema-constraints — CHECK to enforce valid range
  score         smallint                  check (score between 1 and 5),

  -- Enum status — drives restart recovery in Python without touching InMemorySaver
  status        public.test_session_status not null default 'active',

  -- Skill ref: schema-data-types — timestamptz always
  created_at    timestamptz               not null default now(),
  completed_at  timestamptz               -- null while session is still active
);

alter table public.test_sessions enable row level security;


-- ============================================================
-- INDEXES: test_sessions
-- ============================================================

-- Skill ref: schema-foreign-key-indexes — index the FK column
create index if not exists test_sessions_memory_id_idx
  on public.test_sessions (memory_id);

-- Partial index for active sessions only
-- Used by the scheduler query that excludes memories with an active test:
--   WHERE id NOT IN (SELECT memory_id FROM test_sessions WHERE status = 'active')
-- Skill ref: query-partial-indexes — tiny index, only active rows, very fast
create index if not exists test_sessions_active_memory_idx
  on public.test_sessions (memory_id)
  where status = 'active';

-- GIN index on conversation JSONB for future full-text search across test history
-- Skill ref: advanced-jsonb-indexing — jsonb_path_ops is 2-3x smaller, suits our query pattern
create index if not exists test_sessions_conversation_gin_idx
  on public.test_sessions
  using gin (conversation jsonb_path_ops);