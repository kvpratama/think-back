


SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;


CREATE EXTENSION IF NOT EXISTS "pg_net" WITH SCHEMA "extensions";






COMMENT ON SCHEMA "public" IS 'standard public schema';



CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";






CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";






CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";






CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA "extensions";






CREATE TYPE "public"."test_session_status" AS ENUM (
    'active',
    'completed'
);


ALTER TYPE "public"."test_session_status" OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."match_memories"("query_embedding" "extensions"."vector", "filter" "jsonb" DEFAULT '{}'::"jsonb", "match_count" integer DEFAULT 5) RETURNS TABLE("id" "uuid", "content" "text", "metadata" "jsonb", "similarity" double precision)
    LANGUAGE "plpgsql"
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
      'review_count',     memories.review_count,
      'test_score_avg',   memories.test_score_avg
    ) || COALESCE(memories.metadata, '{}'::jsonb) as metadata,
    1 - (memories.embedding <=> query_embedding) as similarity
  from memories
  -- Apply optional metadata filter if passed by LangChain
  -- For ThinkBack we rarely use this but it keeps the interface standard
  where
    case
      when filter = '{}'::jsonb then true
      else memories.id::text = filter->>'id'  -- basic id filter support
    end
  order by memories.embedding <=> query_embedding
  limit match_count;
end;
$$;


ALTER FUNCTION "public"."match_memories"("query_embedding" "extensions"."vector", "filter" "jsonb", "match_count" integer) OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."set_updated_at"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
begin
  new.updated_at = now();
  return new;
end;
$$;


ALTER FUNCTION "public"."set_updated_at"() OWNER TO "postgres";

SET default_tablespace = '';

SET default_table_access_method = "heap";


CREATE TABLE IF NOT EXISTS "public"."memories" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "content" "text" NOT NULL,
    "metadata" "jsonb" DEFAULT '{}'::"jsonb",
    "source" "text",
    "embedding" "extensions"."vector"(768) NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "last_reviewed_at" timestamp with time zone,
    "review_count" integer DEFAULT 0 NOT NULL,
    "test_score_avg" double precision DEFAULT 0.0 NOT NULL
);


ALTER TABLE "public"."memories" OWNER TO "postgres";


CREATE OR REPLACE VIEW "public"."memories_with_tags" AS
SELECT
    NULL::"uuid" AS "id",
    NULL::"text" AS "content",
    NULL::"text" AS "source",
    NULL::timestamp with time zone AS "created_at",
    NULL::timestamp with time zone AS "last_reviewed_at",
    NULL::integer AS "review_count",
    NULL::double precision AS "test_score_avg",
    NULL::"text"[] AS "tags";


ALTER VIEW "public"."memories_with_tags" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."memory_tags" (
    "memory_id" "uuid" NOT NULL,
    "tag_id" bigint NOT NULL
);


ALTER TABLE "public"."memory_tags" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."test_sessions" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "memory_id" "uuid" NOT NULL,
    "question" "text" NOT NULL,
    "conversation" "jsonb" DEFAULT '[]'::"jsonb" NOT NULL,
    "score" smallint,
    "status" "public"."test_session_status" DEFAULT 'active'::"public"."test_session_status" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "completed_at" timestamp with time zone,
    CONSTRAINT "test_sessions_score_check" CHECK ((("score" >= 1) AND ("score" <= 5)))
);


ALTER TABLE "public"."test_sessions" OWNER TO "postgres";


CREATE OR REPLACE VIEW "public"."surfacing_candidates" AS
 SELECT "id",
    "source",
    "review_count",
    "test_score_avg",
    "last_reviewed_at",
    (COALESCE((EXTRACT(epoch FROM ("now"() - "last_reviewed_at")) / 86400.0), 999.0))::double precision AS "days_since_reviewed",
        CASE
            WHEN ("last_reviewed_at" > ("now"() - '24:00:00'::interval)) THEN true
            ELSE false
        END AS "reviewed_recently",
        CASE
            WHEN ("last_reviewed_at" IS NULL) THEN true
            ELSE false
        END AS "is_novel"
   FROM "public"."memories" "m"
  WHERE (NOT ("id" IN ( SELECT "test_sessions"."memory_id"
           FROM "public"."test_sessions"
          WHERE ("test_sessions"."status" = 'active'::"public"."test_session_status"))));


ALTER VIEW "public"."surfacing_candidates" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."tags" (
    "id" bigint NOT NULL,
    "name" "text" NOT NULL,
    CONSTRAINT "tags_name_lowercase" CHECK (("name" = "lower"("name")))
);


ALTER TABLE "public"."tags" OWNER TO "postgres";


ALTER TABLE "public"."tags" ALTER COLUMN "id" ADD GENERATED ALWAYS AS IDENTITY (
    SEQUENCE NAME "public"."tags_id_seq"
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1
);



CREATE TABLE IF NOT EXISTS "public"."user_settings" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "telegram_chat_id" "text" NOT NULL,
    "reminder_time_1" time without time zone DEFAULT '08:00:00'::time without time zone NOT NULL,
    "reminder_time_2" time without time zone DEFAULT '20:00:00'::time without time zone NOT NULL,
    "test_time" time without time zone DEFAULT '09:00:00'::time without time zone NOT NULL,
    "timezone" "text" DEFAULT 'UTC'::"text" NOT NULL,
    "reminders_enabled" boolean DEFAULT true NOT NULL,
    "test_enabled" boolean DEFAULT true NOT NULL,
    "confirm_before_save" boolean DEFAULT true NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."user_settings" OWNER TO "postgres";


ALTER TABLE ONLY "public"."memories"
    ADD CONSTRAINT "memories_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."memory_tags"
    ADD CONSTRAINT "memory_tags_pkey" PRIMARY KEY ("memory_id", "tag_id");



ALTER TABLE ONLY "public"."tags"
    ADD CONSTRAINT "tags_name_unique" UNIQUE ("name");



ALTER TABLE ONLY "public"."tags"
    ADD CONSTRAINT "tags_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."test_sessions"
    ADD CONSTRAINT "test_sessions_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."user_settings"
    ADD CONSTRAINT "user_settings_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."user_settings"
    ADD CONSTRAINT "user_settings_telegram_chat_id_key" UNIQUE ("telegram_chat_id");



CREATE INDEX "memories_embedding_hnsw_idx" ON "public"."memories" USING "hnsw" ("embedding" "extensions"."vector_cosine_ops") WITH ("m"='16', "ef_construction"='64');



CREATE INDEX "memories_last_reviewed_at_idx" ON "public"."memories" USING "btree" ("last_reviewed_at" NULLS FIRST);



CREATE INDEX "memories_never_reviewed_idx" ON "public"."memories" USING "btree" ("created_at") WHERE ("last_reviewed_at" IS NULL);



CREATE INDEX "memory_tags_memory_id_idx" ON "public"."memory_tags" USING "btree" ("memory_id");



CREATE INDEX "memory_tags_tag_id_idx" ON "public"."memory_tags" USING "btree" ("tag_id");



CREATE INDEX "test_sessions_active_memory_idx" ON "public"."test_sessions" USING "btree" ("memory_id") WHERE ("status" = 'active'::"public"."test_session_status");



CREATE INDEX "test_sessions_conversation_gin_idx" ON "public"."test_sessions" USING "gin" ("conversation" "jsonb_path_ops");



CREATE INDEX "test_sessions_memory_id_idx" ON "public"."test_sessions" USING "btree" ("memory_id");



CREATE OR REPLACE VIEW "public"."memories_with_tags" AS
 SELECT "m"."id",
    "m"."content",
    "m"."source",
    "m"."created_at",
    "m"."last_reviewed_at",
    "m"."review_count",
    "m"."test_score_avg",
    COALESCE("array_agg"("t"."name" ORDER BY "t"."name") FILTER (WHERE ("t"."name" IS NOT NULL)), ARRAY[]::"text"[]) AS "tags"
   FROM (("public"."memories" "m"
     LEFT JOIN "public"."memory_tags" "mt" ON (("mt"."memory_id" = "m"."id")))
     LEFT JOIN "public"."tags" "t" ON (("t"."id" = "mt"."tag_id")))
  GROUP BY "m"."id";



CREATE OR REPLACE TRIGGER "user_settings_set_updated_at" BEFORE UPDATE ON "public"."user_settings" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



ALTER TABLE ONLY "public"."memory_tags"
    ADD CONSTRAINT "memory_tags_memory_id_fkey" FOREIGN KEY ("memory_id") REFERENCES "public"."memories"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."memory_tags"
    ADD CONSTRAINT "memory_tags_tag_id_fkey" FOREIGN KEY ("tag_id") REFERENCES "public"."tags"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."test_sessions"
    ADD CONSTRAINT "test_sessions_memory_id_fkey" FOREIGN KEY ("memory_id") REFERENCES "public"."memories"("id") ON DELETE CASCADE;



CREATE POLICY "deny anon access to memories" ON "public"."memories" TO "anon" USING (false);



CREATE POLICY "deny anon access to memory_tags" ON "public"."memory_tags" TO "anon" USING (false);



CREATE POLICY "deny anon access to tags" ON "public"."tags" TO "anon" USING (false);



CREATE POLICY "deny anon access to test_sessions" ON "public"."test_sessions" TO "anon" USING (false);



CREATE POLICY "deny anon access to user_settings" ON "public"."user_settings" TO "anon" USING (false);



ALTER TABLE "public"."memories" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."memory_tags" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."tags" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."test_sessions" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."user_settings" ENABLE ROW LEVEL SECURITY;




ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";





GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";
























































































































































































































































































































































































































































































































GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "anon";
GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."set_updated_at"() TO "service_role";






























GRANT ALL ON TABLE "public"."memories" TO "anon";
GRANT ALL ON TABLE "public"."memories" TO "authenticated";
GRANT ALL ON TABLE "public"."memories" TO "service_role";



GRANT ALL ON TABLE "public"."memories_with_tags" TO "anon";
GRANT ALL ON TABLE "public"."memories_with_tags" TO "authenticated";
GRANT ALL ON TABLE "public"."memories_with_tags" TO "service_role";



GRANT ALL ON TABLE "public"."memory_tags" TO "anon";
GRANT ALL ON TABLE "public"."memory_tags" TO "authenticated";
GRANT ALL ON TABLE "public"."memory_tags" TO "service_role";



GRANT ALL ON TABLE "public"."test_sessions" TO "anon";
GRANT ALL ON TABLE "public"."test_sessions" TO "authenticated";
GRANT ALL ON TABLE "public"."test_sessions" TO "service_role";



GRANT ALL ON TABLE "public"."surfacing_candidates" TO "anon";
GRANT ALL ON TABLE "public"."surfacing_candidates" TO "authenticated";
GRANT ALL ON TABLE "public"."surfacing_candidates" TO "service_role";



GRANT ALL ON TABLE "public"."tags" TO "anon";
GRANT ALL ON TABLE "public"."tags" TO "authenticated";
GRANT ALL ON TABLE "public"."tags" TO "service_role";



GRANT ALL ON SEQUENCE "public"."tags_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."tags_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."tags_id_seq" TO "service_role";



GRANT ALL ON TABLE "public"."user_settings" TO "anon";
GRANT ALL ON TABLE "public"."user_settings" TO "authenticated";
GRANT ALL ON TABLE "public"."user_settings" TO "service_role";









ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS TO "service_role";






ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES TO "service_role";































