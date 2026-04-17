-- ============================================================
-- Migration 004: user_settings + updated_at trigger
-- Stores per-user configuration for ThinkBack
-- Single-user app but designed cleanly with telegram_chat_id as the user key
-- ============================================================


-- ------------------------------------------------------------
-- FUNCTION: set_updated_at()
-- Reusable trigger function — can be attached to any table with updated_at
-- Skill ref: schema-data-types — timestamptz for all time columns
-- ------------------------------------------------------------
create or replace function public.set_updated_at()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;


-- ------------------------------------------------------------
-- TABLE: user_settings
-- ------------------------------------------------------------
create table if not exists public.user_settings (
  -- Skill ref: schema-primary-keys — gen_random_uuid() (UUIDv4), sufficient at personal scale
  id                  uuid        primary key default gen_random_uuid(),

  -- The user's Telegram chat ID — fixed for the lifetime of their account
  -- UNIQUE ensures only one settings row per user
  telegram_chat_id    text        not null unique,

  -- Scheduler times — stored as TIME (no date, no timezone — timezone applied at runtime)
  reminder_time_1     time        not null default '08:00',
  reminder_time_2     time        not null default '20:00',

  -- Timezone string e.g. 'Asia/Jakarta', 'America/New_York', 'UTC'
  -- Applied by APScheduler when scheduling jobs
  timezone            text        not null default 'UTC',

  -- Feature toggles
  reminders_enabled   boolean     not null default true,
  confirm_before_save boolean     not null default true,

  -- Skill ref: schema-data-types — timestamptz always
  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now()
);

alter table public.user_settings enable row level security;


-- ------------------------------------------------------------
-- TRIGGER: auto-update updated_at on every UPDATE
-- Skill ref: agreed design decision — Postgres trigger, set and forget
-- ------------------------------------------------------------
create or replace trigger user_settings_set_updated_at
  before update on public.user_settings
  for each row
  execute function public.set_updated_at();


-- ============================================================
-- INDEXES: user_settings
-- ============================================================

-- Fast lookup by telegram_chat_id on every bot message
-- UNIQUE constraint already creates an index but we name it explicitly for clarity
-- (The unique constraint index is sufficient — no separate index needed)
-- The telegram_chat_id unique constraint covers the lookup pattern:
--   SELECT * FROM user_settings WHERE telegram_chat_id = $1


-- ============================================================
-- SEED: default settings row
-- Insert a placeholder row that the bot will upsert on first run
-- Replace 'YOUR_TELEGRAM_CHAT_ID' in .env — bot auto-upserts on startup
-- ============================================================
-- (Intentionally left empty — the Python bot upserts settings on first message)