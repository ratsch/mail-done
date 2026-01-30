# Local Development Guide

> **Related:** [Deployment Guide](DEPLOYMENT.md), [API Reference](API.md)

This guide covers setting up mail-done for local development without Docker.

## Prerequisites

- Python 3.11 (required - not 3.10 or 3.12+)
- PostgreSQL 16+ with pgvector extension
- Poetry for dependency management
- Git

## Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/YOUR-ORG/mail-done.git
cd mail-done
```

### 2. Install Python Dependencies

```bash
# Install Poetry if needed
pip install poetry

# Install dependencies
poetry install

# Verify installation
poetry run python --version  # Should show 3.11.x
```

### 3. Set Up PostgreSQL

#### Option A: Use Docker for Database Only

```bash
# Start just the database
docker run -d \
    --name mail-done-dev-db \
    -p 5432:5432 \
    -e POSTGRES_USER=postgres \
    -e POSTGRES_PASSWORD=devpassword \
    -e POSTGRES_DB=email_processor \
    -v mail-done-dev-data:/var/lib/postgresql/data \
    pgvector/pgvector:pg16

# Initialize extensions
docker exec mail-done-dev-db psql -U postgres -d email_processor -c "
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
"
```

#### Option B: Local PostgreSQL Installation

```bash
# macOS with Homebrew
brew install postgresql@16
brew services start postgresql@16

# Install pgvector
brew install pgvector

# Create database
createdb email_processor
psql email_processor -c "CREATE EXTENSION IF NOT EXISTS vector;"
psql email_processor -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"
psql email_processor -c "CREATE EXTENSION IF NOT EXISTS pg_trgm;"
```

### 4. Configure Environment

```bash
# Copy example environment
cp .env.example .env

# Edit for local development
vim .env
```

Key settings for local development:

```bash
# Local database
DATABASE_URL=postgresql://postgres:devpassword@localhost:5432/email_processor

# Generate encryption key
DB_ENCRYPTION_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")

# API key for testing
API_KEY=dev-api-key-for-testing

# LLM (required for AI features)
OPENAI_API_KEY=sk-...

# Safety: Keep dry run enabled during development
DRY_RUN=true
```

### 5. Set Up Config Files

```bash
# Copy example configs
for f in config/*.example.yaml; do
    cp "$f" "${f%.example.yaml}.yaml"
done

# Edit accounts.yaml with your test email
vim config/accounts.yaml
```

### 6. Initialize Database

```bash
# Create tables
poetry run python -c "
from backend.core.database.connection import init_db, create_tables
init_db()
create_tables()
print('Tables created')
"

# Or use Alembic if migrations exist
poetry run alembic upgrade head
```

### 7. Start Development Server

```bash
# With auto-reload
poetry run uvicorn backend.api.main:app --reload --port 8000

# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## Development Workflow

### Running Tests

```bash
# All tests
poetry run pytest

# With coverage
poetry run pytest --cov=backend --cov-report=html

# Specific test file
poetry run pytest backend/tests/unit/test_classifier.py -v

# Watch mode (re-run on changes)
poetry run ptw
```

### Linting and Formatting

```bash
# Check code style
poetry run ruff check .

# Auto-format
poetry run black backend/

# Type checking
poetry run mypy backend/
```

### Processing Test Emails

```bash
# Always use dry-run during development
poetry run python process_inbox.py --dry-run --limit 5

# With verbose output
poetry run python process_inbox.py --dry-run --limit 5 --verbose
```

### Testing MCP Server

```bash
# Start MCP server
./run_mcp_server.sh

# In another terminal, test the API
curl -H "X-API-Key: dev-api-key-for-testing" http://localhost:8000/api/emails?limit=1
```

## Project Structure

```
mail-done/
├── backend/
│   ├── api/              # FastAPI routes
│   │   ├── main.py       # App entry point
│   │   └── routes/       # Endpoint modules
│   └── core/
│       ├── ai/           # LLM classification
│       ├── email/        # IMAP processing
│       ├── database/     # SQLAlchemy models
│       └── search/       # Vector search
├── mcp_server/           # MCP server for AI assistants
├── config/               # YAML configuration
├── deploy/               # Deployment scripts
├── docs/                 # Documentation
└── tests/                # Test files
```

## Common Development Tasks

### Adding a New API Endpoint

1. Create route in `backend/api/routes/`
2. Register in `backend/api/main.py`
3. Add tests in `backend/tests/`
4. Update `docs/API.md`

### Adding a New Email Category

1. Add to `config/categories.yaml`
2. Add to `backend/core/email/models.py` (EmailCategory enum)
3. Add actions in `config/ai_category_actions.yaml`
4. Update classifier prompts if needed

### Modifying Database Schema

```bash
# Create migration
poetry run alembic revision --autogenerate -m "description"

# Review generated migration
vim alembic/versions/xxx_description.py

# Apply migration
poetry run alembic upgrade head
```

## Troubleshooting

### Poetry Install Fails

```bash
# Clear cache and retry
poetry cache clear . --all
poetry install
```

### Database Connection Errors

```bash
# Check PostgreSQL is running
pg_isready -h localhost -p 5432

# Check connection string
poetry run python -c "
from backend.core.database.connection import engine
print(engine.url)
"
```

### Import Errors

```bash
# Ensure you're in the project root
cd /path/to/mail-done

# Use poetry run for correct environment
poetry run python -c "from backend.api.main import app; print('OK')"
```

## Environment Variables Reference

| Variable | Development Value | Purpose |
|----------|-------------------|---------|
| `DATABASE_URL` | `postgresql://postgres:devpassword@localhost:5432/email_processor` | Local database |
| `DRY_RUN` | `true` | Prevent IMAP modifications |
| `API_KEY` | `dev-api-key` | Simple test key |
| `LOG_LEVEL` | `DEBUG` | Verbose logging |
| `OPENAI_API_KEY` | Your key | Required for AI features |

## VS Code Configuration

Recommended `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["backend/tests"],
    "editor.formatOnSave": true,
    "python.formatting.provider": "black",
    "python.linting.ruffEnabled": true
}
```
