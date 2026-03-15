-- ============================================================
-- Migration 006: Helper Views
-- Convenience views used by the Python memory_db.py layer
-- Avoids repeating complex JOINs in application code
-- ============================================================


-- ------------------------------------------------------------
-- VIEW: memories_with_tags
-- Returns each memory with its tags aggregated as a text array
-- Used by: /list command, /stats, and save confirmation display
-- ------------------------------------------------------------
create or replace view public.memories_with_tags as
select
  m.id,
  m.content,
  m.summary,
  m.source,
  m.created_at,
  m.last_reviewed_at,
  m.review_count,
  m.test_score_avg,
  -- Aggregate tag names into an array; empty array if no tags
  coalesce(
    array_agg(t.name order by t.name) filter (where t.name is not null),
    array[]::text[]
  ) as tags
from public.memories m
left join public.memory_tags mt on mt.memory_id = m.id
left join public.tags t         on t.id = mt.tag_id
group by m.id;


-- ------------------------------------------------------------
-- VIEW: surfacing_candidates
-- Returns memories eligible for spaced repetition surfacing
-- Excludes memories that have an active test session in progress
-- Used by: scheduler reminder job and /remind command
-- ------------------------------------------------------------
create or replace view public.surfacing_candidates as
select
  m.id,
  m.summary,
  m.source,
  m.review_count,
  m.test_score_avg,
  m.last_reviewed_at,
  -- Surfacing score components — Python multiplies these together
  -- days_since_reviewed: null (never reviewed) treated as a large number (999)
  coalesce(
    extract(epoch from (now() - m.last_reviewed_at)) / 86400.0,
    999.0
  )::float as days_since_reviewed,
  -- Flag: reviewed in last 24h → Python applies 0.1x recency penalty
  case
    when m.last_reviewed_at > now() - interval '24 hours' then true
    else false
  end as reviewed_recently,
  -- Flag: never reviewed → Python applies 1.5x novelty bonus
  case
    when m.last_reviewed_at is null then true
    else false
  end as is_novel
from public.memories m
where
  -- Core exclusion: skip memories with an active test session
  -- Skill ref: query-partial-indexes — backed by test_sessions_active_memory_idx
  m.id not in (
    select memory_id
    from public.test_sessions
    where status = 'active'
  );