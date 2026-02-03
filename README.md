# mail-done

An AI-powered email and document processing system designed for academics and professionals who need intelligent organization, semantic search, and workflow automation across their communications and files.

**mail-done** helps you manage two related problems:
1. **Email overload** - Hundreds of emails needing classification, organization, and search
2. **Document sprawl** - Files scattered across email attachments, local folders, and remote servers

## What This Tool Does

**mail-done** connects to your email accounts and document folders, then uses large language models (GPT-4, Claude, etc.) to:

**For Email:**
1. **Classify emails** into categories (PhD applications, speaking invitations, collaboration requests, newsletters, etc.)
2. **Score applications** with detailed AI analysis (scientific excellence, research fit, technical skills)
3. **Organize your inbox** by moving emails to folders, adding labels, and flagging important messages
4. **Enable semantic search** so you can find emails by meaning ("papers about single-cell RNA sequencing") rather than exact keywords

**For Documents:**
5. **Index files** from email attachments, local folders, and remote servers (via SSH)
6. **Extract text** from PDFs, Office documents, images (OCR), and other formats
7. **Deduplicate** files across sources (same PDF attached to 10 emails = indexed once)
8. **Unified search** across emails and documents simultaneously

### Primary Use Cases

- **Academic PIs** receiving hundreds of PhD/postdoc applications who need AI-assisted screening
- **Researchers** who want to search their email archive semantically ("find collaboration requests about CRISPR")
- **Professionals** drowning in email who want automatic organization and priority detection
- **Anyone** who wants their AI assistant (Claude, Cursor) to understand and search their email

### What This Tool Does NOT Do

- **Send emails** - This is a read-only system for processing incoming mail
- **Replace your email client** - You still use Gmail/Outlook/Apple Mail; this runs alongside
- **Modify your files** - Documents are indexed but never changed; originals stay where they are
- **Require cloud hosting** - Designed for self-hosting on your own server or Raspberry Pi
- **Store data in the cloud** - Your emails and document index stay in your PostgreSQL database under your control

## Features

### Email Processing & Classification

When emails arrive, the system:

1. Fetches new messages via IMAP from one or more email accounts
2. Extracts the sender, subject, body, and attachments
3. Applies rule-based pre-classification (e.g., "emails from @nature.com go to 'journals'")
4. Sends unclassified emails to an LLM for AI categorization
5. Executes actions based on classification (move to folder, add label, flag)
6. Generates vector embeddings for semantic search
7. Stores everything in PostgreSQL with pgvector for fast similarity search

**Supported email providers:** Any IMAP server (Gmail, Outlook/Office365, institutional email, self-hosted)

**Supported LLMs:** OpenAI (GPT-4o, GPT-4o-mini), Azure OpenAI, Anthropic Claude

### Application Review System

Built specifically for academic labs reviewing PhD, postdoc, and intern applications:

**AI Analysis** - Each application is scored on a 1-10 scale for:
- Scientific excellence (publications, research experience, academic trajectory)
- Research fit (alignment with lab's research areas)
- Overall recommendation
- Technical skills (coding, omics/genomics, medical data, image analysis, etc.)

**Automatic Extraction** - The AI extracts:
- Applicant name, email, institution, nationality
- Current degree and situation
- Online profiles (GitHub, LinkedIn, Google Scholar)
- Key strengths and concerns
- Suggested next steps

**Reference Letter Matching** - Automatically finds reference letters sent separately and links them to the application using semantic similarity search.

**Google Drive Export** - Uploads all application materials (email text, CV, transcripts, letters) to organized Google Drive folders for collaborative review.

**Review Portal** - Web interface where multiple reviewers can rate applications, leave comments, and make decisions.

### Document Indexing & Multi-Source Search

Search across three sources simultaneously:

1. **Email text** - The body and subject of your emails
2. **Email attachments** - PDFs, Word docs, spreadsheets attached to emails
3. **Folder files** - Documents from local directories or remote servers (via SSH)

**How it works:**
- Documents are hashed for deduplication (same file attached to 10 emails = indexed once)
- Text is extracted from PDFs, Office documents, RTF, plain text, and CSV files
- OCR can extract text from images and scanned documents
- Vector embeddings enable semantic search ("find invoices from 2024" finds invoices even if the word "invoice" isn't in the filename)

**Example queries:**
- "machine learning papers from collaborators" (searches emails)
- "CV from the Yale applicant" (searches attachments)
- "grant proposal about cancer genomics" (searches folder documents)
- "everything about project X" (searches all three sources)

### MCP Server for AI Assistants

The MCP (Model Context Protocol) server lets AI assistants like Claude and Cursor search your email directly. Instead of copy-pasting emails into ChatGPT, you can ask:

- "Find emails from Dr. Smith about the collaboration"
- "Show me PhD applications with research fit score above 8"
- "What's in my Archive/2024 folder?"
- "Download the transcript from that ETH applicant"

The server exposes 25+ tools organized by function:

**Email Search Tools:**
- `semantic_search` - Find emails by meaning/concept
- `search_by_sender` - Find all emails from someone
- `search_by_topic` - Search by research topic
- `find_similar_emails` - Find emails similar to a reference
- `get_email_details` - Get full email content
- `list_categories` - See all email categories
- `list_top_senders` - See who emails you most

**Document Tools:**
- `semantic_document_search` - Search indexed files
- `semantic_search_unified` - Search emails AND documents together
- `get_document_details` - Get file metadata and text preview
- `find_similar_documents` - Find related documents
- `search_document_by_name` - Search by filename
- `get_document_index_stats` - See indexing progress
- `list_indexed_folder` - Browse folder contents
- `download_document` - Retrieve files to local cache

**IMAP Tools (live server access):**
- `list_imap_folders` - List all folders on mail server
- `get_imap_folder_status` - Get message counts
- `list_imap_folder_emails` - List emails in a folder

**Application Review Tools:**
- `list_applications` - Search and filter applications
- `get_application_details` - Full application with AI scores
- `get_application_tags` - Available profile tags
- `get_application_collections` - Application groupings

**Attachment Tools (when enabled):**
- `list_attachments` - List email attachments
- `download_attachment` - Download to local cache
- `clear_attachment_cache` - Manage disk usage

### Web Interface

A lightweight web UI for common tasks:

- **Semantic Search** - Search emails with natural language queries
- **Multi-Source Search** - Toggle between emails, attachments, and documents
- **Cost Tracking** - Monitor LLM API costs by day, model, and task type
- **Processing Triggers** - Start email processing jobs from the browser
- **Statistics** - View database stats and category breakdowns

### Privacy & Security

- **Self-hosted** - Runs on your own infrastructure (server, NAS, Raspberry Pi)
- **No cloud dependency** - Emails stored in your PostgreSQL database
- **Encrypted storage** - Sensitive fields encrypted at rest with Fernet
- **API authentication** - Multiple auth methods (API keys, HMAC signatures, OAuth, JWT)
- **Audit logging** - All MCP tool calls logged for security review

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
# Dry run first (safe preview - no changes made)
python3.11 process_inbox.py --dry-run --limit 10

# Process new emails
python3.11 process_inbox.py --new-only
```

## Components

### Backend API

FastAPI-based REST API providing programmatic access to all functionality:

```bash
# Start development server
poetry run uvicorn backend.api.main:app --reload --port 8000

# API documentation (interactive)
open http://localhost:8000/docs
```

Endpoints include:
- `/api/emails` - Email CRUD and search
- `/api/search` - Semantic and hybrid search
- `/api/documents` - Document indexing and search
- `/api/imap` - Direct IMAP folder access
- `/api/review` - Application review portal
- `/api/stats` - Usage statistics and costs

See [docs/API.md](docs/API.md) for full API documentation.

### MCP Server

Connect your AI assistant to your email:

```bash
# Start MCP server
./run_mcp_server.sh
```

Then configure in Claude Desktop, Cursor, or Claude Code:

```json
{
  "mcpServers": {
    "email-search": {
      "command": "/path/to/run_mcp_server.sh",
      "env": {
        "MCP_API_KEY": "your-secure-key"
      }
    }
  }
}
```

Now you can ask your AI assistant:
- "Search my emails for machine learning papers"
- "Find PhD applications about genomics with score > 8"
- "Show documents similar to this invoice"
- "List emails in my Archive/2024 folder"

See [docs/MCP.md](docs/MCP.md) for detailed setup instructions.

### Web UI

Lightweight browser interface:

```bash
cd web-ui
./start.sh  # Docker (recommended)
# or
python app.py  # Standalone
```

Access at `http://localhost:5000`

See [web-ui/README.md](web-ui/README.md) for details.

### Email Processor

Command-line tool for batch email processing:

```bash
# Basic processing
python3.11 process_inbox.py --dry-run --limit 10    # Preview changes
python3.11 process_inbox.py --new-only              # Process unseen emails
python3.11 process_inbox.py --reprocess             # Re-run AI on existing
python3.11 process_inbox.py --actions-only          # Execute pending moves/labels

# Application processing
python3.11 process_inbox.py --export-files          # Upload to Google Drive
python3.11 process_inbox.py --find-reference-letters # Match reference letters

# Attachment indexing
python3.11 process_inbox.py --index-attachments     # Index attachments as documents
python3.11 process_inbox.py --backfill-attachments  # Index existing email attachments
```

See [docs/PROCESS_INBOX.md](docs/PROCESS_INBOX.md) for all options.

### Document Indexer

Index documents from folders for unified search:

```bash
# Scan and index a local folder
python3.11 -m backend.core.documents.folder_scanner \
    --path /path/to/documents \
    --host my-machine

# Scan a remote folder via SSH
python3.11 -m backend.core.documents.folder_scanner \
    --path /home/user/papers \
    --host remote-server \
    --ssh-host server.example.com

# Generate embeddings for all indexed documents
python3.11 -m backend.core.documents.embeddings --generate-missing
```

## Deployment

Deploy to your own server using Docker or Podman:

```bash
# Standard Docker deployment
./deploy/deploy.sh

# Raspberry Pi with Podman
./deploy/deploy.sh --pi
```

The system is designed to run on modest hardware - a Raspberry Pi 4 can handle email processing for a typical academic inbox.

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
| `IMAP_PASSWORD` | Email password or app password | Yes |
| `API_KEY` | Backend API authentication | Recommended |
| `DRY_RUN` | Disable IMAP modifications | No (default: false) |
| `CONFIG_DIR` | Custom config overlay directory | No |
| `PROMPTS_DIR` | Custom prompts overlay directory | No |
| `MCP_API_KEY` | MCP server authentication | For MCP |
| `MCP_ENABLE_FILE_DOWNLOADS` | Enable attachment downloads | No |
| `MCP_FILE_CACHE_DIR` | Attachment cache directory | If downloads enabled |
| `GOOGLE_DRIVE_FOLDER_ID` | Root folder for exports | For GDrive |

*Or use Azure OpenAI with `AZURE_OPENAI_*` variables.

### Config Files (config/)

| File | Description |
|------|-------------|
| `vip_senders.yaml` | VIP sender definitions with priority levels |
| `ai_category_actions.yaml` | IMAP actions (move, label, flag) for each AI category |
| `classification_rules.yaml` | Rule-based pre-classification before AI |
| `preprocessing_rules.yaml` | Forwarding detection and sender extraction |
| `accounts.yaml` | Multi-account email configuration |
| `model_config.yaml` | LLM model selection per email category |
| `clients.yaml` | API client authentication |

### Private Configuration Overlay

For sensitive configurations (VIP lists, custom prompts), use overlay directories:

```bash
export CONFIG_DIR=/path/to/private-config
export PROMPTS_DIR=/path/to/private-config/prompts
```

This allows keeping institutional rules in a separate private repository while using the main codebase.

## Architecture

```
mail-done/
├── backend/
│   ├── api/                    # FastAPI REST API
│   │   ├── main.py             # App entry point
│   │   ├── routes/             # API endpoints
│   │   │   ├── emails.py       # Email CRUD
│   │   │   ├── search.py       # Semantic search
│   │   │   ├── documents.py    # Document indexing
│   │   │   ├── imap.py         # Live IMAP access
│   │   │   ├── attachments.py  # Attachment handling
│   │   │   ├── review_*.py     # Application review portal
│   │   │   └── ...
│   │   ├── signed_auth.py      # HMAC request signing
│   │   └── review_auth.py      # Review portal OAuth
│   └── core/
│       ├── ai/                 # LLM integration
│       │   ├── classifier.py   # Email classification
│       │   ├── prompts/        # System prompts (customizable)
│       │   └── providers/      # OpenAI, Azure, Anthropic
│       ├── email/              # Email processing
│       │   ├── processor.py    # Main processing pipeline
│       │   └── imap_actions.py # Folder moves, labels
│       ├── database/           # Data layer
│       │   ├── models.py       # SQLAlchemy models
│       │   ├── repository.py   # Data access
│       │   └── encryption.py   # Field-level encryption
│       ├── documents/          # Document indexing
│       │   ├── folder_scanner.py      # Scan directories
│       │   ├── attachment_indexer.py  # Index email attachments
│       │   ├── processor.py           # Text extraction
│       │   ├── embeddings.py          # Vector generation
│       │   └── search.py              # Document search
│       ├── search/             # Email vector search
│       └── auth/               # OAuth, JWT, API keys
├── mcp_server/                 # MCP server for AI assistants
│   ├── server.py               # Tool definitions (25+ tools)
│   └── api_client.py           # Backend API client
├── config/                     # Configuration files
├── web-ui/                     # Browser interface
├── deploy/                     # Docker/Podman deployment
└── docs/                       # Documentation
```

## Development

```bash
# Run tests
poetry run pytest
poetry run pytest --cov=backend --cov-report=html

# Watch mode (re-run tests on file changes)
poetry run ptw

# Linting
poetry run ruff check .
poetry run black backend/
poetry run mypy backend/
```

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for full local development setup.

## Documentation

- [Deployment Guide](docs/DEPLOYMENT.md) - Self-hosted deployment (Docker/Podman)
- [Review Portal Deployment](docs/DEPLOYMENT_REVIEW.md) - Application review portal setup
- [Development Guide](docs/DEVELOPMENT.md) - Local development setup
- [API Documentation](docs/API.md) - REST API endpoints
- [MCP Server Setup](docs/MCP.md) - Claude/Cursor integration
- [Database Schema](docs/DATABASE.md) - PostgreSQL and pgvector
- [Email Processing](docs/PROCESS_INBOX.md) - Command-line processor options
- [LLM Configuration](docs/LLM_CONFIGURATION.md) - Model selection per category
- [Gmail Setup](docs/GMAIL_SETUP.md) - Gmail OAuth configuration
- [Outlook OAuth](docs/OUTLOOK_OAUTH2.md) - Microsoft 365 OAuth setup
- [Web UI](web-ui/README.md) - Browser interface

## License

MIT
