# Initial Setup

```bash
supabase init
supabase login
supabase link --project-ref <project-ref>

# Create a migration file in supabase/migrations/ that matches your remote schema
supabase db pull

# Launch the local Supabase Docker containers
supabase start
supabase status # to view your local URLs and API keys.
supabase stop # to shut down the Docker containers

# Export your database's structure (schema) into a single SQL file
supabase db dump > supabase/schemas/schema.sql
# Export data from your linked Supabase project and save it as a local seed file
supabase db dump --data-only > supabase/seed.sql
# Generates TypeScript definitions
supabase gen types typescript --local > src/lib/supabase/database.types.ts

# To reset your local database completely (Destructive):
supabase db reset
```

## Declarative Workflow:

```bash

# 1. Modify declarative files in supabase/schema.sql

# 2. Generate a migration file
supabase db diff -f <migration_name>

# 3. Verify that the generated migration contain a single incremental change in supabase/migrations/ Ensure it doesn't suggest something destructive (like dropping a column to rename it)

# 4. Apply the migration manually to see your schema changes in the local Dashboard
supabase start
supabase migration up

# 5. Push your changes to the remote database.
supabase db push

# 6. Remember to regenerate your TypeScript definitions after each schema change.
supabase gen types typescript --local > src/lib/supabase/database.types.ts
```

## Migration-First Workflow:

```bash

# 1. Generate a new migration to store the SQL needed
# This creates a new migration: supabase/migrations/<timestamp>_<migration_name>.sql
supabase migration new <migration_name>

# 2. Edit the generated migration file

# 3. Apply the migration manually to see your schema changes in the local Dashboard
supabase start
supabase migration up # Good for quickly applying a new file without losing local data.
supabase db reset # Better for "quality assurance." It wipes the local DB, runs all migrations in order, and re-runs your seed.sql. This guarantees that a new developer joining your team (or your CI/CD pipeline) can set up the project without errors.
supabase db reset --linked # Same as above but for the linked remote database.

# Check which database migrations have been applied by querying the schema_migrations table
SELECT version FROM supabase_migrations.schema_migrations;

# 4. Push your changes to the remote database.
supabase db push

# 5. Remember to regenerate Python definitions after each schema change
supabase gen types --lang=python --local > supabase/database_types.py

# 6. Remember to regenerate schema.sql
supabase db dump --local > supabase/schema.sql
```
