#!/usr/bin/env bash
set -e

PROJECT_DIR="$HOME/think-back"
BACKUP_DIR="$HOME/db-backup/think-back"
DATE=$(date +%Y-%m-%d)

cd "$PROJECT_DIR"

echo "[$(date)] Starting backup..."
supabase db dump --data-only > supabase/seed.sql

mkdir -p "$BACKUP_DIR"
cp supabase/seed.sql "$BACKUP_DIR/seed-$DATE.sql"

echo "[$(date)] Backup complete: $BACKUP_DIR/seed-$DATE.sql"
