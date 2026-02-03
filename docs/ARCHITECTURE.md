# Architecture Overview

This document describes the high-level architecture, key components, and data flow of mail-done.

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER INTERFACES                                 │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│   Web UI        │   MCP Server    │   CLI Tools     │   Review Portal       │
│   (Flask)       │   (stdio)       │   (Python)      │   (Next.js)           │
│   :5000         │                 │                 │   :3000               │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                     │
         └─────────────────┴────────┬────────┴─────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           BACKEND API (FastAPI)                              │
│                              :8000                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  Authentication: API Key, HMAC Signed Requests, OAuth2, JWT                  │
│  Rate Limiting: Per-client request throttling                                │
│  Audit Logging: All requests logged for security review                      │
└────────┬────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CORE MODULES                                    │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│   AI            │   Email         │   Documents     │   Search              │
│   Classification│   Processing    │   Indexing      │   (Vector)            │
├─────────────────┼─────────────────┼─────────────────┼───────────────────────┤
│   Database      │   Auth          │   Google        │   Signing             │
│   (Repository)  │   (OAuth/JWT)   │   (Drive)       │   (HMAC)              │
└────────┬────────┴────────┬────────┴────────┬────────┴───────────┬───────────┘
         │                 │                 │                     │
         ▼                 ▼                 ▼                     ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   PostgreSQL    │ │   IMAP Server   │ │   Google Drive  │ │   LLM APIs      │
│   + pgvector    │ │   (Gmail, etc)  │ │   API           │ │   (OpenAI, etc) │
└─────────────────┘ └─────────────────┘ └─────────────────┘ └─────────────────┘
```

## Directory Structure

```
mail-done/
├── backend/                    # Python backend
│   ├── api/                    # FastAPI application
│   │   ├── main.py             # App entry point, middleware setup
│   │   ├── routes/             # API endpoint handlers
│   │   ├── auth.py             # API key authentication
│   │   ├── signed_auth.py      # HMAC request signing
│   │   ├── review_auth.py      # Review portal OAuth
│   │   ├── rate_limiting.py    # Request throttling
│   │   ├── security_monitor.py # Threat detection
│   │   └── schemas.py          # Pydantic models
│   │
│   ├── core/                   # Business logic modules
│   │   ├── ai/                 # LLM integration
│   │   ├── email/              # Email processing
│   │   ├── documents/          # Document indexing
│   │   ├── database/           # Data access layer
│   │   ├── search/             # Vector search
│   │   ├── auth/               # Authentication helpers
│   │   ├── google/             # Google Drive integration
│   │   ├── signing/            # Request signing
│   │   ├── accounts/           # Multi-account management
│   │   ├── tracking/           # Link/pixel tracking
│   │   ├── replies/            # Reply suggestions
│   │   └── analytics/          # Usage analytics
│   │
│   ├── tests/                  # Test suite
│   └── utils/                  # Shared utilities
│
├── mcp_server/                 # MCP server for AI assistants
│   ├── server.py               # Tool definitions
│   └── api_client.py           # Backend API client
│
├── web-ui/                     # Lightweight web interface
│   ├── app.py                  # Flask application
│   ├── static/                 # JS, CSS
│   └── templates/              # HTML templates
│
├── config/                     # Configuration files
│   ├── vip_senders.yaml        # VIP definitions
│   ├── classification_rules.yaml
│   ├── ai_category_actions.yaml
│   ├── accounts.yaml           # Multi-account config
│   └── model_config.yaml       # LLM model selection
│
├── deploy/                     # Deployment scripts
│   ├── docker-compose.yml
│   └── deploy.sh
│
├── docs/                       # Documentation
├── alembic/                    # Database migrations
└── process_inbox.py            # Main CLI processor
```

## Core Modules

### 1. AI Module (`backend/core/ai/`)

Handles all LLM interactions for email classification and analysis.

```
ai/
├── classifier.py           # Main classification logic
├── two_stage_classifier.py # Fast model → detailed model pipeline
├── config_loader.py        # Model selection per category
├── providers/              # LLM provider abstraction
│   ├── base.py             # Abstract provider interface
│   ├── openai_provider.py  # OpenAI/Azure OpenAI
│   └── anthropic_provider.py
├── prompts/                # System prompts (customizable)
│   ├── classification.py   # General email classification
│   └── application.py      # Application-specific analysis
└── config/                 # Model configuration
```

**Key Classes:**
- `EmailClassifier` - Classifies emails into categories
- `TwoStageClassifier` - Uses fast model for triage, detailed model for complex cases
- `LLMProvider` - Abstract interface for LLM providers

**Data Flow:**
```
Email → Classifier → LLM Provider → Response Parser → Classification Result
                         ↓
                    OpenAI / Azure / Anthropic
```

### 2. Email Module (`backend/core/email/`)

Handles IMAP connections, email parsing, and inbox actions.

```
email/
├── processor.py            # Email processing pipeline
├── imap_monitor.py         # IMAP connection management
├── imap_actions.py         # Folder moves, labels, flags
├── attachment_extractor.py # Extract and parse attachments
├── preprocessing.py        # Forwarding detection, sender extraction
├── models.py               # Email data models
└── oauth2_imap.py          # OAuth2 IMAP authentication
```

**Key Classes:**
- `EmailProcessor` - Main processing pipeline
- `IMAPMonitor` - Manages IMAP connections per account
- `IMAPActions` - Executes folder moves, labels, flags
- `AttachmentExtractor` - Extracts text from attachments

**Processing Pipeline:**
```
IMAP Fetch → Parse → Preprocess → Classify → Actions → Store → Embed
     ↓           ↓          ↓          ↓         ↓        ↓       ↓
  Raw email   Extract    Detect     LLM      Move/     DB    Vector
              parts     forwarding          Label           embedding
```

### 3. Documents Module (`backend/core/documents/`)

Indexes documents from multiple sources for unified search.

```
documents/
├── folder_scanner.py       # Scan local/remote directories
├── attachment_indexer.py   # Index email attachments
├── processor.py            # Text extraction (PDF, Office, etc.)
├── embeddings.py           # Generate vector embeddings
├── search.py               # Document search
├── repository.py           # Data access
├── models.py               # Document data models
├── retrieval.py            # File retrieval from origins
└── config.py               # Indexing configuration
```

**Key Classes:**
- `FolderScanner` - Scans directories, tracks changes
- `AttachmentIndexer` - Indexes email attachments as documents
- `DocumentProcessor` - Extracts text from various formats
- `DocumentEmbeddingService` - Generates embeddings

**Document Sources:**
```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Local Folders  │     │ Email Attachments│     │  Remote (SSH)   │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Document Repository    │
                    │   (deduplicated by hash) │
                    └────────────┬────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │   Vector Embeddings     │
                    │   (semantic search)     │
                    └─────────────────────────┘
```

### 4. Database Module (`backend/core/database/`)

Data access layer using SQLAlchemy with PostgreSQL/pgvector.

```
database/
├── models.py               # SQLAlchemy ORM models
├── repository.py           # Data access methods
├── encryption.py           # Field-level encryption (Fernet)
├── session.py              # Session management
└── migrations/             # Alembic migrations (in /alembic)
```

**Key Tables:**
- `emails` - Email metadata and content
- `documents` - Indexed document files
- `document_origins` - Where documents were found
- `embeddings` - Vector embeddings for emails
- `document_embeddings` - Vector embeddings for documents
- `applications` - Application-specific metadata
- `reviews` - Application reviews

**Encryption:**
Sensitive fields (email body, sender) are encrypted at rest using Fernet symmetric encryption.

### 5. Search Module (`backend/core/search/`)

Vector similarity search using pgvector.

```
search/
├── vector_search.py        # Email vector search
├── embeddings.py           # Embedding generation
├── unified_search.py       # Cross-source search
└── hybrid_search.py        # Combine vector + keyword
```

**Search Modes:**
- **Semantic** - Pure vector similarity (finds conceptually related content)
- **Keyword** - Traditional text matching
- **Hybrid** - Combines both (recommended for most use cases)

### 6. Auth Module (`backend/core/auth/`)

Authentication and authorization.

```
auth/
├── oauth2.py               # OAuth2 flows (Google, Microsoft)
├── jwt_handler.py          # JWT token management
└── api_keys.py             # API key validation
```

**Authentication Methods:**
1. **API Key** - Simple header-based auth (`X-API-Key`)
2. **HMAC Signing** - Request signing for sensitive operations
3. **OAuth2** - For review portal user authentication
4. **JWT** - Session tokens for web interfaces

### 7. Google Module (`backend/core/google/`)

Google Drive integration for application exports.

```
google/
├── drive_client.py         # Drive API wrapper
└── auth.py                 # Service account authentication
```

**Features:**
- Create folders per applicant
- Upload email text, LLM analysis, attachments
- Organize by application type (PhD, postdoc, intern)

## API Routes

The FastAPI application exposes these endpoint groups:

| Route Prefix | Module | Purpose |
|--------------|--------|---------|
| `/api/emails` | `routes/emails.py` | Email CRUD, listing |
| `/api/search` | `routes/search.py` | Semantic and hybrid search |
| `/api/documents` | `routes/documents.py` | Document indexing and search |
| `/api/imap` | `routes/imap.py` | Direct IMAP folder access |
| `/api/attachments` | `routes/attachments.py` | Attachment download |
| `/api/stats` | `routes/stats.py` | Database statistics |
| `/api/costs` | `routes/costs.py` | LLM cost tracking |
| `/api/review/*` | `routes/review_*.py` | Application review portal |

## Data Flow Examples

### Email Processing Flow

```
1. process_inbox.py starts
2. IMAPMonitor connects to configured accounts
3. For each new email:
   a. Fetch from IMAP server
   b. Parse headers, body, attachments
   c. Check classification rules (rule-based)
   d. If not matched, send to LLM classifier
   e. Execute IMAP actions (move, label, flag)
   f. Store in database
   g. Generate embedding for search
   h. If application: extract applicant info, scores
   i. If --export-files: upload to Google Drive
```

### Search Flow

```
1. User query arrives (API or MCP)
2. Generate embedding for query text
3. Search modes:
   - Semantic: pgvector similarity search
   - Keyword: PostgreSQL full-text search
   - Hybrid: Combine and re-rank results
4. Apply filters (date, category, sender)
5. Return ranked results with snippets
```

### Document Indexing Flow

```
1. FolderScanner walks directory tree
2. For each file:
   a. Calculate SHA-256 hash
   b. Check if already indexed (deduplication)
   c. Extract text (PDF, Office, etc.)
   d. Create/update Document record
   e. Add origin record (where file was found)
3. DocumentEmbeddingService generates embeddings
4. Documents searchable via unified search
```

## Configuration System

Configuration is loaded from multiple sources with overlay support:

```
Priority (highest to lowest):
1. Environment variables
2. CONFIG_DIR overlay (private repo)
3. config/ directory (default)
```

This allows:
- Default configs in the main repo
- Sensitive configs (VIP lists, custom prompts) in a private overlay repo
- Environment-specific overrides

## Security Model

### Authentication Layers

1. **Transport**: HTTPS (Caddy/nginx reverse proxy)
2. **API Auth**: API key or HMAC signature required
3. **Review Portal**: OAuth2 + JWT sessions
4. **MCP Server**: Separate MCP_API_KEY

### Data Protection

1. **At Rest**: Fernet encryption for sensitive fields
2. **In Transit**: TLS for all connections
3. **Secrets**: Environment variables, never in code
4. **Audit**: All API calls logged

### Rate Limiting

- Per-client request limits
- Configurable thresholds
- Automatic blocking on abuse

## Deployment Options

1. **Docker Compose** - Standard deployment with all services
2. **Podman** - Rootless containers for Raspberry Pi
3. **Manual** - Direct Python execution for development

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed instructions.
