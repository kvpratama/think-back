# ThinkBack 🧠

**ThinkBack** is a memory-enhanced AI assistant designed to help you capture, organize, and retrieve your personal knowledge seamlessly. Built with **LangGraph** and **Supabase**, it provides a long-term memory for your interactions, allowing you to build a personal knowledge base that grows with you.

---

## ✨ Key Features

- **🧠 Long-Term Memory**: Automatically saves and retrieves information using vector embeddings.
- **🔍 Semantic Search**: Find what you need based on meaning, not just keywords.
- **🤖 Interactive Telegram Bot**: A user-friendly interface to interact with your knowledge base anywhere.
- **⚡ Streaming Responses**: Real-time feedback as the AI "thinks" and generates answers.
- **🏗️ Robust Architecture**: Powered by LangGraph for complex agentic workflows.
- **🔌 Flexible LLM Support**: Supports OpenAI, Google Gemini, and any OpenAI-compatible provider (e.g., Local LLMs).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Agent Framework** | [LangGraph](https://langchain-ai.github.io/langgraph/) / [LangChain](https://www.langchain.com/) |
| **Database & Vector Store** | [Supabase](https://supabase.com/) (PostgreSQL + pgvector) |
| **LLM & Embeddings** | Google Gemini / OpenAI / OpenAI-compatible |
| **Bot Interface** | [python-telegram-bot](https://python-telegram-bot.org/) |
| **Package Management** | [uv](https://github.com/astral-sh/uv) |

> **Note**: Vector dimensions are fixed at 768. Changing this requires a full database migration.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) installed.
- A Supabase project with `pgvector` enabled.
- API keys for Telegram (from [@BotFather](https://t.me/botfather)) and your chosen LLM provider (Google Gemini or OpenAI).

### Quick Start

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd think-back
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

3. **Configure Environment Variables**:
   Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   # Edit .env with your keys
   ```

4. **Run the Telegram Bot**:
   ```bash
   uv run python -m src.api.bot
   ```

---

## 📖 Usage

Once the bot is running, you can interact with it on Telegram using the following commands:

- `/start`: Initialize the assistant and see available commands.
- `/save <text>`: Save a piece of information to your long-term memory.
- `/ask <question>`: Query your saved knowledge base.
- `/query <question>`: Alternative for querying your knowledge.

---

## 🛠️ Development

Common commands for development tasks:

```bash
# Run tests
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format .

# Type checking
uv run ty check
```

---

## 📂 Project Structure

- `src/agent/`: Core agent logic (graph assembly, nodes, state).
- `src/api/`: Bot interface and message handling.
- `src/core/`: Application configuration and settings.
- `src/db/`: Database and vector store clients.
- `supabase/`: SQL migrations and database schema.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.