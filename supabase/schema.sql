


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






CREATE OR REPLACE FUNCTION "public"."match_memories"("query_embedding" "extensions"."vector", "p_user_settings_id" "uuid", "match_count" integer DEFAULT 5, "filter" "jsonb" DEFAULT '{}'::"jsonb") RETURNS TABLE("id" "uuid", "content" "text", "metadata" "jsonb", "similarity" double precision)
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
    memories.user_settings_id = p_user_settings_id
    and case
      when filter = '{}'::jsonb then true
      else memories.id::text = filter->>'id'
    end
  order by memories.embedding <=> query_embedding
  limit match_count;
end;
$$;


ALTER FUNCTION "public"."match_memories"("query_embedding" "extensions"."vector", "p_user_settings_id" "uuid", "match_count" integer, "filter" "jsonb") OWNER TO "postgres";


CREATE OR REPLACE FUNCTION "public"."set_updated_at"() RETURNS "trigger"
    LANGUAGE "plpgsql" SECURITY DEFINER
    SET "search_path" TO 'public'
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
    "user_settings_id" "uuid" NOT NULL,
    "content" "text" NOT NULL,
    "metadata" "jsonb" DEFAULT '{}'::"jsonb",
    "source" "text",
    "embedding" "extensions"."vector"(768) NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "last_reviewed_at" timestamp with time zone,
    "review_count" integer DEFAULT 0 NOT NULL
);


ALTER TABLE "public"."memories" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."reminder_times" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "user_settings_id" "uuid" NOT NULL,
    "time" time without time zone NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."reminder_times" OWNER TO "postgres";


CREATE TABLE IF NOT EXISTS "public"."user_settings" (
    "id" "uuid" DEFAULT "gen_random_uuid"() NOT NULL,
    "telegram_chat_id" "text" NOT NULL,
    "timezone" "text" DEFAULT 'UTC'::"text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT "now"() NOT NULL,
    "updated_at" timestamp with time zone DEFAULT "now"() NOT NULL
);


ALTER TABLE "public"."user_settings" OWNER TO "postgres";


ALTER TABLE ONLY "public"."memories"
    ADD CONSTRAINT "memories_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."reminder_times"
    ADD CONSTRAINT "reminder_times_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."reminder_times"
    ADD CONSTRAINT "uq_user_reminder_time" UNIQUE ("user_settings_id", "time");



ALTER TABLE ONLY "public"."user_settings"
    ADD CONSTRAINT "user_settings_pkey" PRIMARY KEY ("id");



ALTER TABLE ONLY "public"."user_settings"
    ADD CONSTRAINT "user_settings_telegram_chat_id_key" UNIQUE ("telegram_chat_id");



CREATE INDEX "memories_embedding_hnsw_idx" ON "public"."memories" USING "hnsw" ("embedding" "extensions"."vector_cosine_ops") WITH ("m"='16', "ef_construction"='64');



CREATE INDEX "memories_last_reviewed_at_idx" ON "public"."memories" USING "btree" ("last_reviewed_at" NULLS FIRST);



CREATE INDEX "memories_never_reviewed_idx" ON "public"."memories" USING "btree" ("created_at") WHERE ("last_reviewed_at" IS NULL);



CREATE INDEX "memories_user_settings_id_idx" ON "public"."memories" USING "btree" ("user_settings_id");



CREATE OR REPLACE TRIGGER "user_settings_set_updated_at" BEFORE UPDATE ON "public"."user_settings" FOR EACH ROW EXECUTE FUNCTION "public"."set_updated_at"();



ALTER TABLE ONLY "public"."memories"
    ADD CONSTRAINT "memories_user_settings_id_fkey" FOREIGN KEY ("user_settings_id") REFERENCES "public"."user_settings"("id") ON DELETE CASCADE;



ALTER TABLE ONLY "public"."reminder_times"
    ADD CONSTRAINT "reminder_times_user_settings_id_fkey" FOREIGN KEY ("user_settings_id") REFERENCES "public"."user_settings"("id") ON DELETE CASCADE;



CREATE POLICY "deny anon access to memories" ON "public"."memories" TO "anon" USING (false);



CREATE POLICY "deny anon access to reminder_times" ON "public"."reminder_times" TO "anon" USING (false);



CREATE POLICY "deny anon access to user_settings" ON "public"."user_settings" TO "anon" USING (false);



ALTER TABLE "public"."memories" ENABLE ROW LEVEL SECURITY;


ALTER TABLE "public"."reminder_times" ENABLE ROW LEVEL SECURITY;


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



GRANT ALL ON TABLE "public"."reminder_times" TO "anon";
GRANT ALL ON TABLE "public"."reminder_times" TO "authenticated";
GRANT ALL ON TABLE "public"."reminder_times" TO "service_role";



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































