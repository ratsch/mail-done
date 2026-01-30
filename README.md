# mail-done

AI-powered email processing system with automatic classification, semantic search, and IMAP actions.

## Features

- **AI Classification**: Automatic email categorization using LLMs (OpenAI/Anthropic)
- **Semantic Search**: Vector-based email search using pgvector
- **IMAP Actions**: Automatic folder moves, labels, and flagging based on classification
- **VIP Detection**: Priority handling for important senders
- **Email Preprocessing**: Extract original senders from forwarded emails
- **MCP Server**: Integration with Claude Code and other AI assistants
- **Multi-Account Support**: Handle multiple email accounts

## Requirements

- Python 3.11 (required - not 3.10 or 3.12+)
- PostgreSQL with pgvector extension
- Poetry for dependency management

## Quick Start

```bash
# Install dependencies
poetry install

# Copy example configs
cp .env.example .env
cp config/*.example.yaml config/  # Remove .example suffix

# Edit configs with your settings
vim .env
vim config/vip_senders.yaml

# Process emails (dry run first)
python3.11 process_inbox.py --dry-run --limit 10

# Process for real
python3.11 process_inbox.py --new-only
```

## Configuration

### Environment Variables (.env)

Required variables:
- `DATABASE_URL` - PostgreSQL connection string
- `OPENAI_API_KEY` - For AI classification and embeddings
- `IMAP_HOST/PORT/USERNAME/PASSWORD` - Email server credentials
- `DB_ENCRYPTION_KEY` - Fernet key for field-level encryption

Optional overlay directories:
- `CONFIG_DIR` - Custom config directory (overrides ./config)
- `PROMPTS_DIR` - Custom prompts directory (overrides ./backend/core/ai/prompts)

### Config Files (config/)

- `vip_senders.yaml` - VIP sender definitions with priority levels
- `ai_category_actions.yaml` - IMAP actions for each AI category
- `classification_rules.yaml` - Rule-based pre-classification
- `preprocessing_rules.yaml` - Email preprocessing (forwarding detection)
- `clients.yaml` - API client authentication (optional)

## Commands

```bash
# Process inbox
python3.11 process_inbox.py --dry-run --limit 10    # Preview
python3.11 process_inbox.py --new-only              # Process unseen
python3.11 process_inbox.py --reprocess             # Re-run AI on processed
python3.11 process_inbox.py --actions-only          # Execute pending actions

# Run API server
poetry run uvicorn backend.api.main:app --reload

# Run MCP server (for Claude Code integration)
./run_mcp_server.sh

# Run tests
poetry run pytest
poetry run pytest --cov=backend
```

## Architecture

```
backend/
  api/          - FastAPI routes and authentication
  core/
    ai/         - LLM classification (classifier.py, prompts/)
    email/      - IMAP processing (processor.py, imap_actions.py)
    database/   - SQLAlchemy models and repository
    search/     - Vector search with pgvector
    auth/       - OAuth 2.0, JWT, API keys

config/         - Configuration files (YAML)
mcp_server/     - MCP server for AI assistant integration
```

## Private Configuration

For personal/institutional configurations, use the `CONFIG_DIR` and `PROMPTS_DIR` environment variables to overlay custom configs without modifying this repo:

```bash
export CONFIG_DIR=/path/to/private-config
export PROMPTS_DIR=/path/to/private-config/prompts
```

This allows keeping sensitive VIP lists, custom prompts, and institutional rules in a separate private repository.

## License

MIT
