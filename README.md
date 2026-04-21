<div align="center">

<img src="assets/think-back-transparent.png" alt="ThinkBack Logo" width="200">

# ThinkBack

</div>

**ThinkBack** is a memory-enhanced AI assistant that helps you capture, organize, and retrieve your personal knowledge through conversation. Built with **LangGraph** and **Supabase**, it provides long-term memory for your interactions via a Telegram bot interface.

---

## ✨ Key Features

- **🧠 Long-Term Memory**: Saves and retrieves information using vector embeddings
- **🔍 Semantic Search**: Finds relevant memories based on meaning, not just keywords
- **🤖 Telegram Bot Interface**: Interact with your knowledge base anywhere, anytime
- **🔔 Spaced Repetition Reminders**: Scheduled memory reviews with AI-generated insights
- **🛡️ Duplicate Detection**: Prevents redundant memories with semantic similarity matching
- **🔌 Flexible LLM Support**: Works with OpenAI or any OpenAI-compatible provider

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) + [LangChain](https://www.langchain.com/) |
| **Database & Vector Store** | [Supabase](https://supabase.com/) (PostgreSQL + pgvector) |
| **LLM** | OpenAI / OpenAI-compatible |
| **Embeddings** | Google Gemini (gemini-embedding-001) |
| **Bot Interface** | [python-telegram-bot](https://python-telegram-bot.org/) |
| **Package Management** | [uv](https://github.com/astral-sh/uv) |
| **Testing** | pytest + pytest-asyncio |
| **Linting** | ruff |

> **Note**: Vector dimensions are fixed at 768. Changing this requires a database migration.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager
- A [Supabase](https://supabase.com/) project with `pgvector` extension enabled
- API keys:
  - **Telegram bot token**: Open Telegram, search for [@BotFather](https://t.me/botfather), send `/newbot`, follow the prompts to name your bot, and copy the token provided
  - **OpenAI API key**: Get from [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
  - **Google Gemini API key**: Get from [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd think-back
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Set up Supabase**:
   - Create a new Supabase project
   - Enable the `pgvector` extension
   - Run migrations from `supabase/migrations/`

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your credentials
   ```

   Required variables:
   - `SUPABASE_URL` - Your Supabase project URL
   - `SUPABASE_KEY` - Your Supabase service role key
   - `TELEGRAM_BOT_TOKEN` - Bot token from @BotFather
   - `OPENAI_API_KEY` - OpenAI API key
   - `GEMINI_API_KEY` - Google Gemini API key

5. **Run the bot**:
   ```bash
   uv run python -m src.api.bot
   ```

6. **Configure via Telegram**:
   - Start a chat with your bot on Telegram
   - Send `/start` to initialize your account
   - Set your timezone when prompted
   - Use `/reminders` to add reminder times
   - You're ready to start saving memories!

---

## 📖 Usage

### Bot Commands

- `/start` - Initialize your account and set timezone (interactive setup)
- `/help` - Show available commands
- `/reminders` - Add or remove reminder times (up to 3 per day)
- `/timezone` - Update your timezone

All configuration is done through the bot interface — no manual database setup required.

### Natural Interaction

Just chat naturally with the bot:

- **Save memories**: Share insights, lessons, or facts you want to remember
  - The bot will extract the core insight and ask for confirmation before saving
  - Duplicate detection prevents redundant entries

- **Search memories**: Ask questions about your saved knowledge
  - The bot searches semantically and returns relevant memories
  - Answers are based only on your saved memories (no external knowledge)

---

## ⚠️ Important Notes

- **Private chats only**: ThinkBack is designed for one-on-one conversations. It does not support group chats or channels.

---

## ⏰ Automated Reminders

ThinkBack includes a spaced repetition system that automatically surfaces your memories via Telegram.

### How It Works

The reminder job (`src/jobs/remind.py`) runs hourly via GitHub Actions:

1. **User Selection**: Finds users whose configured reminder times match the current hour (respects timezones)
2. **Memory Selection**: Uses weighted random selection to pick a memory:
   - Novel memories (never reviewed): weighted by age since creation
   - Reviewed memories: weighted by `days_since_review / review_count`
   - This creates natural spaced repetition — neglected memories surface more often
3. **Insight Generation**: LLM generates a personalized insight and reflective question
4. **Delivery**: Sends formatted message via Telegram with the memory, insight, and question
5. **Tracking**: Updates `last_reviewed_at` and increments `review_count`

### Running Locally

```bash
# Run the reminder job manually
uv run python -m src.jobs.remind
```

### GitHub Actions Setup

The reminder runs automatically every hour via `.github/workflows/remind.yml`. Required secrets:
- `SUPABASE_URL`, `SUPABASE_KEY`
- `OPENAI_API_KEY`, `GEMINI_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- Optional: `LLM_MODEL`, `LLM_PROVIDER`, `LLM_PROVIDER_BASE_URL`, `EMBEDDING_MODEL`

---

## 🛠️ Development

### Common Commands

```bash
# Run all tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run ty check

# Add a dependency
uv add <package>

# Add a dev dependency
uv add --dev <package>
```

### Running Evaluations

```bash
# Seed evaluation dataset
uv run seed-dataset

# Run evaluation pipeline
uv run run-evals
```

### Database Operations

```bash
# Seed memories from JSON (get user_settings_id from Supabase user_settings table)
uv run python -m src.db.seed_memories <user_settings_id>

# Generate type definitions from schema
supabase gen types python --local > supabase/database_types.py
```

---

## 📂 Project Structure

```
think-back/
├── src/
│   ├── agent/          # LangGraph agent (graph, tools, state)
│   │   ├── graph.py    # Agent assembly and system prompt
│   │   ├── tools.py    # save_memory_tool, search_memories_tool
│   │   └── state.py    # State schemas (Memory, DuplicateMatch)
│   ├── api/            # Telegram bot interface
│   │   ├── bot.py      # Main bot application
│   │   ├── bot_commands.py      # Command handlers
│   │   ├── bot_callbacks.py     # Callback query handlers
│   │   ├── bot_keyboards.py     # Inline keyboard builders
│   │   └── bot_helpers.py       # Message formatting utilities
│   ├── core/           # Configuration
│   │   └── config.py   # Pydantic settings
│   ├── db/             # Database layer
│   │   ├── client.py   # Supabase client singleton
│   │   ├── vector_store.py      # Vector operations
│   │   └── user_settings.py     # User preferences
│   ├── jobs/           # Background jobs
│   │   └── remind.py   # Spaced repetition reminder job
│   └── evals/          # Evaluation framework
│       ├── evaluators/ # Custom evaluators
│       ├── dataset_examples.py  # Test cases
│       └── run_evals.py         # Evaluation runner
├── supabase/
│   ├── migrations/     # Database migrations
│   ├── schema.sql      # Generated schema (DO NOT EDIT)
│   └── database_types.py        # Generated types (DO NOT EDIT)
├── docs/
│   └── superpowers/    # Design docs and implementation plans
├── pyproject.toml      # Project configuration
├── langgraph.json      # LangGraph CLI configuration
└── .env.example        # Environment variable template
```

---

## 🧪 Testing

Tests are co-located with their modules using the `<module>_test.py` naming convention:

- Unit tests mock all external services (Supabase, OpenAI)
- Integration tests are marked with `@pytest.mark.integration`
- Async tests use `pytest-asyncio` with `asyncio_mode = "auto"`

Example:
```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src

# Run only unit tests (skip integration)
uv run pytest -m "not integration"
```

---

## 🔧 Configuration

### LLM Configuration

Configure your LLM provider in `.env`:

```bash
# Default: OpenAI GPT-4o-mini
LLM_MODEL=gpt-4o-mini
LLM_PROVIDER=openai
LLM_PROVIDER_BASE_URL=https://api.openai.com/v1

# Or use a local LLM
LLM_MODEL=llama3
LLM_PROVIDER=openai
LLM_PROVIDER_BASE_URL=http://localhost:11434/v1
```

### Embedding Configuration

```bash
# Default: Google Gemini (768 dimensions)
EMBEDDING_MODEL=gemini-embedding-001
```

### Evaluation Configuration

```bash
# LLM for evaluation tasks
EVAL_LLM_MODEL=gpt-4o
EVAL_LLM_PROVIDER=openai

# Multi-judge evaluation (JSON array)
EVAL_JURY_JUDGES='[{"model":"gpt-4o","provider":"openai","api_key_field":"openai_api_key","base_url":""}]'
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Follow the code conventions in `AGENTS.md`
2. Write tests for new features (TDD approach)
3. Ensure all tests pass: `uv run pytest`
4. Format code: `uv run ruff format .`
5. Check types: `uv run ty check`

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

Built with:
- [LangChain](https://www.langchain.com/) & [LangGraph](https://langchain-ai.github.io/langgraph/)
- [Supabase](https://supabase.com/)
- [python-telegram-bot](https://python-telegram-bot.org/)
