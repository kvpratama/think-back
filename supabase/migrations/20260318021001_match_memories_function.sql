-- ============================================================
-- Migration 007: match_memories function
-- Required by LangChain's SupabaseVectorStore
-- SupabaseVectorStore calls this function for similarity search
-- instead of querying the table directly
-- ============================================================

-- The function signature matches what SupabaseVectorStore expects:
--   query_embedding  — the vector to search against
--   filter           — optional JSONB metadata filter (passed by LangChain)
--   match_count      — max results to return
--
-- We map LangChain's generic "content" / "metadata" concepts to
-- our memories table columns:
--   content   → memories.content (the full saved text)
--   metadata  → a JSONB object we build from our columns
--              (source, last_reviewed_at, review_count)
--              plus any additional memories.metadata stored in the table
--
-- The similarity score is: 1 - cosine_distance (higher = more similar)

CREATE OR REPLACE FUNCTION "public"."match_memories"(
    "query_embedding" "extensions"."vector",
    "filter" "jsonb" DEFAULT '{}'::"jsonb",
    "match_count" integer DEFAULT 5,
    "p_user_settings_id" uuid DEFAULT NULL
)
RETURNS TABLE(
    "id" "uuid",
    "content" "text",
    "metadata" "jsonb",
    "similarity" double precision
)
LANGUAGE "plpgsql" SECURITY DEFINER
SET "search_path" TO 'public', 'extensions'
AS $$
#variable_conflict use_column
begin
  return query
  select
    memories.id,
    memories.content,
    jsonb_build_object(
      'source',           memories.source,
      'last_reviewed_at', memories.last_reviewed_at,
      'review_count',     memories.review_count
    ) || COALESCE(memories.metadata, '{}'::jsonb) as metadata,
    1 - (memories.embedding <=> query_embedding) as similarity
  from memories
  where
    (p_user_settings_id IS NULL OR memories.user_settings_id = p_user_settings_id)
    and case
      when filter = '{}'::jsonb then true
      else memories.id::text = filter->>'id'
    end
  order by memories.embedding <=> query_embedding
  limit match_count;
end;
$$;
