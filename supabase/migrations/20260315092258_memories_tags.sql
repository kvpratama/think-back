-- ============================================================
-- Migration 002: memories, tags, memory_tags
-- Core knowledge base tables for ThinkBack
-- ============================================================


-- ------------------------------------------------------------
-- TABLE: memories
-- Each row is one deliberately saved piece of knowledge
-- ------------------------------------------------------------
create table if not exists public.memories (
  -- Skill ref: schema-primary-keys — UUIDv4
  id              uuid        primary key default gen_random_uuid(),

  -- Skill ref: schema-data-types — use text not varchar(n), no artificial limits
  content         text        not null,
  metadata        jsonb       default '{}'::jsonb,
  source          text,                         -- nullable: user may not always specify

  -- pgvector column — 768 dims matches gemini-embedding-001
  -- Must match embedding model; changing model requires re-embedding all rows
  embedding       extensions.vector(768) not null,

  -- Skill ref: schema-data-types — always use timestamptz (timezone-aware)
  created_at      timestamptz not null default now(),
  last_reviewed_at timestamptz,                 -- null until first review

  -- Surfacing weight inputs — kept simple (rolling avg only, per PRD decision)
  review_count    int         not null default 0,
  test_score_avg  float       not null default 0.0  -- 0.0 = never tested
);

-- Skill ref: security-rls-basics — enable RLS on all tables in Supabase
-- Single-user app: service role key bypasses RLS; anon key is blocked
alter table public.memories enable row level security;


-- ------------------------------------------------------------
-- TABLE: tags
-- Normalised tag lookup table — avoids TEXT[] duplication issues
-- ------------------------------------------------------------
create table if not exists public.tags (
  -- Skill ref: schema-primary-keys — bigint identity for simple lookup table
  id    bigint  generated always as identity primary key,

  -- Lowercase enforced via CHECK constraint — prevents "Habits" vs "habits" duplicates
  -- Skill ref: schema-constraints — use CHECK for data integrity
  name  text    not null,

  constraint tags_name_unique unique (name),
  constraint tags_name_lowercase check (name = lower(name))
);

alter table public.tags enable row level security;


-- ------------------------------------------------------------
-- TABLE: memory_tags
-- Join table: many memories ↔ many tags
-- ------------------------------------------------------------
create table if not exists public.memory_tags (
  memory_id   uuid    not null references public.memories(id) on delete cascade,
  tag_id      bigint  not null references public.tags(id)     on delete cascade,

  primary key (memory_id, tag_id)
);

alter table public.memory_tags enable row level security;


-- ============================================================
-- INDEXES: memories
-- ============================================================

-- HNSW index for fast approximate nearest-neighbour vector search
-- Skill ref: query-index-types — GiST/HNSW for KNN queries
-- vector_cosine_ops matches cosine similarity used in RAG queries
-- m=16, ef_construction=64 are safe defaults for a personal-scale collection
create index if not exists memories_embedding_hnsw_idx
  on public.memories
  using hnsw (embedding extensions.vector_cosine_ops)
  with (m = 16, ef_construction = 64);

-- B-tree on last_reviewed_at for surfacing score queries
-- NULLS FIRST so never-reviewed memories sort to the top naturally
-- Skill ref: query-missing-indexes — index columns used in ORDER BY / WHERE
create index if not exists memories_last_reviewed_at_idx
  on public.memories (last_reviewed_at asc nulls first);

-- Partial index for surfacing score query:
-- only memories NOT currently under an active test session need fast lookup
-- Skill ref: query-partial-indexes — smaller index, faster for filtered queries
-- (The active-test exclusion is handled in the query WHERE clause)
create index if not exists memories_never_reviewed_idx
  on public.memories (created_at)
  where last_reviewed_at is null;


-- ============================================================
-- INDEXES: memory_tags
-- ============================================================

-- Skill ref: schema-foreign-key-indexes — Postgres does NOT auto-index FK columns
-- Both directions needed: memory→tags (for /list filtering) and tag→memories (for cascade)
create index if not exists memory_tags_tag_id_idx
  on public.memory_tags (tag_id);

create index if not exists memory_tags_memory_id_idx
  on public.memory_tags (memory_id);