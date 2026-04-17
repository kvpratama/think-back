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
  id                  uuid        primary key default gen_random_uuid(),
  telegram_chat_id    text        not null unique,
  timezone            text        not null default 'UTC',
  created_at          timestamptz not null default now(),
  updated_at          timestamptz not null default now()
);

alter table public.user_settings enable row level security;


-- ------------------------------------------------------------
-- TRIGGER: auto-update updated_at on every UPDATE
-- ------------------------------------------------------------
create or replace trigger user_settings_set_updated_at
  before update on public.user_settings
  for each row
  execute function public.set_updated_at();


-- ------------------------------------------------------------
-- TABLE: reminder_times
-- Normalized reminder schedule — one row per reminder per user.
-- Max 5 per user enforced at the app layer.
-- ------------------------------------------------------------
create table if not exists public.reminder_times (
  id                uuid        primary key default gen_random_uuid(),
  user_settings_id  uuid        not null references public.user_settings(id) on delete cascade,
  time              time        not null,
  created_at        timestamptz not null default now(),

  constraint uq_user_reminder_time unique (user_settings_id, time)
);

alter table public.reminder_times enable row level security;