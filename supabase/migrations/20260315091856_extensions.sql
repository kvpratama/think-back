-- ============================================================
-- Migration 001: Extensions
-- Enable required Postgres extensions before any table creation
-- ============================================================

-- pgvector: VECTOR type and HNSW / IVFFlat index support
create extension if not exists vector
  with schema extensions;

-- Note: pg_uuidv7 is not available in the standard Supabase Docker image.
-- We use gen_random_uuid() (UUIDv4) instead. Index fragmentation from UUIDv4
-- only becomes a real concern at millions of rows — irrelevant for a personal tool.