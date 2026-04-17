# Seed Memories Script

Import memories from a JSON file into the database with rate limiting and retry logic.

## Usage

```bash
# Use default file (src/db/seed.json)
uv run python -m src.db.seed_memories

# Use custom file
uv run python -m src.db.seed_memories path/to/your/memories.json
```

## JSON Format

The JSON file should contain an array of memory objects with `content` and `summary` fields:

```json
[
  {
    "content": "Full text of the memory or quote...",
    "summary": "Short summary or title"
  },
  {
    "content": "Another memory...",
    "summary": "Another summary"
  }
]
```

See `src/db/seed.example.json` for a complete example.

## Features

- **Rate Limiting**: Adds 2 second delay between API calls to respect rate limits
- **Retry Logic**: On error, waits 2 minutes and retries once before marking as failed
- **Progress Display**: Shows real-time progress with ✓/✗ indicators
- **Summary Report**: Displays total successes and failures at the end

## Example Output

```
Loading memories from src/db/seed.json...
Found 120 entries to import

[1/120] ✓ The way to get started is to quit talking and...
[2/120] ✓ Life is what happens when you're busy making...
[3/120] ✗ Failed: Some memory that couldn't be saved...
...
[120/120] ✓ Whoever is happy will make others happy too.

Summary:
✓ Successfully imported: 118
✗ Failed: 2
```

## Testing

Run the test suite:

```bash
uv run pytest src/db/seed_memories_test.py -v
```
