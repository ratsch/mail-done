# mail-done

AI-powered email processing system with automatic classification, semantic search, and IMAP actions.

## Features

- **AI Classification**: Automatic email categorization using LLMs (OpenAI/Azure OpenAI/Anthropic)
- **Semantic Search**: Vector-based email search using pgvector embeddings
- **IMAP Actions**: Automatic folder moves, labels, and flagging based on classification
- **VIP Detection**: Priority handling for important senders
- **Email Preprocessing**: Extract original senders from forwarded emails
- **MCP Server**: Integration with Claude Code, Claude Desktop, and Cursor
- **Web UI**: Lightweight web interface for search and management
- **Multi-Account Support**: Handle multiple email accounts
- **Cost Tracking**: Monitor LLM API usage and costs

## Requirements

- Python 3.11 (required - not 3.10 or 3.12+)
- PostgreSQL 16+ with pgvector extension
- Poetry for dependency management

## Quick Start

### 1. Install Dependencies

```bash
poetry install
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
- `DATABASE_URL` - PostgreSQL connection string
- `OPENAI_API_KEY` - For AI classification and embeddings
- `IMAP_HOST/PORT/USERNAME/PASSWORD` - Email server credentials
- `DB_ENCRYPTION_KEY` - Fernet key for field-level encryption

### 3. Set Up Database

```bash
# Run migrations
poetry run alembic upgrade head
```

### 4. Configure Rules

```bash
# Copy example configs
cp config/vip_senders.example.yaml config/vip_senders.yaml
cp config/classification_rules.example.yaml config/classification_rules.yaml

# Edit with your settings
vim config/vip_senders.yaml
```

### 5. Process Emails

```bash
# Dry run first (safe preview)
python3.11 process_inbox.py --dry-run --limit 10

# Process new emails
python3.11 process_inbox.py --new-only
```

## Components

### Backend API

FastAPI-based REST API for email access and search:

```bash
# Start development server
poetry run uvicorn backend.api.main:app --reload --port 8000

# API documentation
open http://localhost:8000/docs
```

See [docs/API.md](docs/API.md) for full API documentation.

### MCP Server

Integration with AI assistants (Claude, Cursor):

```bash
# Start MCP server
./run_mcp_server.sh
```

Configure in your AI assistant to enable natural language email queries:
- "Search my emails for machine learning papers"
- "Find emails from collaborators about the project"
- "Show PhD applications about genomics"

See [docs/MCP.md](docs/MCP.md) for setup instructions.

### Web UI

Lightweight web interface:

```bash
cd web-ui
./start.sh  # Docker
# or
python app.py  # Standalone
```

Features:
- Semantic email search
- Inbox processing triggers
- System statistics
- Cost overview

See [web-ui/README.md](web-ui/README.md) for details.

### Email Processor

Command-line email processing:

```bash
python3.11 process_inbox.py --dry-run --limit 10    # Preview
python3.11 process_inbox.py --new-only              # Process unseen
python3.11 process_inbox.py --reprocess             # Re-run AI
python3.11 process_inbox.py --actions-only          # Execute pending
```

See [docs/PROCESS_INBOX.md](docs/PROCESS_INBOX.md) for full guide.

## Deployment

Deploy to a self-hosted server using Docker or Podman:

```bash
# Standard Docker deployment
./deploy/deploy.sh

# Raspberry Pi with Podman
./deploy/deploy.sh --pi
```

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for complete deployment guide.

## Configuration

### Environment Variables (.env)

| Variable | Description | Required |
|----------|-------------|----------|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `DB_ENCRYPTION_KEY` | Fernet encryption key | Yes |
| `OPENAI_API_KEY` | OpenAI API key | Yes* |
| `IMAP_HOST` | IMAP server hostname | Yes |
| `IMAP_PORT` | IMAP port (usually 993) | Yes |
| `IMAP_USERNAME` | Email address | Yes |
| `IMAP_PASSWORD` | Email password | Yes |
| `API_KEY` | Backend API authentication | Recommended |
| `DRY_RUN` | Disable IMAP modifications | No (default: false) |
| `CONFIG_DIR` | Custom config overlay directory | No |
| `PROMPTS_DIR` | Custom prompts overlay directory | No |

*Or use Azure OpenAI with `AZURE_OPENAI_*` variables.

### Config Files (config/)

| File | Description |
|------|-------------|
| `vip_senders.yaml` | VIP sender definitions with priority levels |
| `ai_category_actions.yaml` | IMAP actions for each AI category |
| `classification_rules.yaml` | Rule-based pre-classification |
| `preprocessing_rules.yaml` | Forwarding detection and sender extraction |
| `clients.yaml` | API client authentication |

### Private Configuration Overlay

For sensitive configurations (VIP lists, custom prompts), use overlay directories:

```bash
export CONFIG_DIR=/path/to/private-config
export PROMPTS_DIR=/path/to/private-config/prompts
```

This allows keeping institutional rules in a separate private repository.

## Architecture

```
mail-done/
├── backend/
│   ├── api/              # FastAPI routes and middleware
│   │   ├── main.py       # App entry point
│   │   ├── routes/       # API endpoints
│   │   └── signed_auth.py # Authentication
│   └── core/
│       ├── ai/           # LLM classification
│       │   ├── classifier.py
│       │   ├── prompts/  # System prompts
│       │   └── providers/ # LLM provider abstraction
│       ├── email/        # Email processing
│       │   ├── processor.py
│       │   └── imap_actions.py
│       ├── database/     # SQLAlchemy models
│       │   ├── models.py
│       │   ├── repository.py
│       │   └── encryption.py
│       ├── search/       # Vector search
│       └── auth/         # OAuth, JWT, API keys
├── mcp_server/           # MCP server for AI assistants
├── config/               # Configuration files
├── web-ui/               # Web interface
├── deploy/               # Deployment scripts
└── docs/                 # Documentation
```

## Development

For full local development setup, see [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md).

```bash
# Run tests
poetry run pytest
poetry run pytest --cov=backend --cov-report=html

# Watch mode
poetry run ptw

# Linting
poetry run ruff check .
poetry run black backend/
poetry run mypy backend/
```

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md) - Self-hosted deployment (Docker/Podman)
- [Development Guide](docs/DEVELOPMENT.md) - Local development setup
- [API Documentation](docs/API.md) - FastAPI endpoints
- [MCP Server Setup](docs/MCP.md) - Claude/Cursor integration
- [Database Schema](docs/DATABASE.md) - PostgreSQL and pgvector
- [Email Processing](docs/PROCESS_INBOX.md) - Command-line processor
- [Web UI](web-ui/README.md) - Web interface

## License

MIT
