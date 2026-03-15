-- ============================================================
-- Migration 005: RLS Policies
-- ThinkBack is single-user — service role key (used by the Python bot)
-- bypasses RLS entirely. These policies block accidental anon/public access.
-- Skill ref: security-rls-basics — enable RLS on all Supabase tables
-- ============================================================

-- ------------------------------------------------------------
-- POLICY STRATEGY
-- The Python backend connects with the service role key, which
-- bypasses RLS — no policy needed for the app itself.
-- These policies exist to:
--   1. Prevent accidental data exposure via the anon key
--   2. Make the intent explicit and auditable
--   3. Be ready if we ever add Supabase Auth in a future version
-- All policies below use USING (false) to deny all anon access.
-- ------------------------------------------------------------


-- memories
create policy "deny anon access to memories"
  on public.memories
  for all
  to anon
  using (false);

-- tags
create policy "deny anon access to tags"
  on public.tags
  for all
  to anon
  using (false);

-- memory_tags
create policy "deny anon access to memory_tags"
  on public.memory_tags
  for all
  to anon
  using (false);

-- test_sessions
create policy "deny anon access to test_sessions"
  on public.test_sessions
  for all
  to anon
  using (false);

-- user_settings
create policy "deny anon access to user_settings"
  on public.user_settings
  for all
  to anon
  using (false);