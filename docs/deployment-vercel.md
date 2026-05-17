# Deploying ThinkBack to Vercel

ThinkBack's Telegram bot runs as a single Vercel Function on the Hobby plan
with Fluid Compute. Reminders run separately on GitHub Actions.

## Prerequisites

- A Vercel account and the [`vercel` CLI](https://vercel.com/docs/cli).
- A Supabase project with the `pgvector` extension and existing
  ThinkBack tables (same as Railway setup).
- All API keys (Telegram, OpenAI, Gemini).

## Project layout (relevant bits)

```
api/
  __init__.py
  _runtime.py       # singletons
  index.py          # FastAPI app (Vercel auto-detects this)
src/                # existing app code
scripts/
  set_bot_commands.py
pyproject.toml      # Vercel installs from this + uv.lock
uv.lock
```

## Environment variables

Set these in the Vercel project settings (all environments — Production,
Preview, Development unless noted):

| Name | Notes |
|---|---|
| `SUPABASE_URL` | Same as Railway. |
| `SUPABASE_KEY` | Same as Railway. |
| `TELEGRAM_BOT_TOKEN` | Same as Railway. |
| `OPENAI_API_KEY` | Same as Railway. |
| `GEMINI_API_KEY` | Same as Railway. |
| `DATABASE_URL` | **Supabase transaction pooler URL (port 6543).** Required for the LangGraph checkpointer in serverless. |
| `WEBHOOK_SECRET` | Random opaque string (e.g. `openssl rand -hex 32`). |

Do **not** set `WEBHOOK_URL` or `PORT` — Vercel handles routing.

## First deploy

1. Link the repo:
   ```bash
   vercel link
   ```
2. Add env vars (or paste them into the Vercel dashboard):
   ```bash
   vercel env add WEBHOOK_SECRET production
   # ...repeat for the others
   ```
3. Deploy:
   ```bash
   vercel --prod
   ```
4. Note the production URL, e.g. `https://thinkback.vercel.app`.

## Post-deploy steps

1. **Register slash-commands** (one-shot; re-run only when the list changes):
   ```bash
   uv run python scripts/set_bot_commands.py
   ```
2. **Register the Telegram webhook**:
   ```bash
   curl -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook" \
     -d "url=https://thinkback.vercel.app/api/webhook" \
     -d "secret_token=${WEBHOOK_SECRET}"
   ```
3. **Health check**:
   ```bash
   curl https://thinkback.vercel.app/api/health
   # → {"status":"ok"}
   ```
4. **End-to-end smoke**: open Telegram, send `/start`, send a memory
   message, tap the Save button. Confirm a row appears in Supabase.

## Operational notes

- **Function timeout**: 300s via Fluid Compute (default on Hobby for
  FastAPI). Typical graph runs are 5–30s.
- **Cold starts**: ~1–3s extra on the first request per container, mostly
  spent opening the Postgres pool.
- **Logs**: Vercel dashboard → Project → Functions → Logs.
- **Reminders**: continue to run via the existing GitHub Actions workflow;
  no Vercel cron needed.

## Switching back to Railway

The polling and `run_webhook` paths in `src/api/bot.py:main()` are
preserved. Self-hosting still works by setting `WEBHOOK_URL` + `PORT` and
running `uv run python -m src.api.bot`.
