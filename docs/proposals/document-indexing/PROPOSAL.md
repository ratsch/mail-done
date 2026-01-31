# Document Indexing Feature Proposal

**Status:** Draft (Revised after code review)
**Author:** Claude (with human guidance)
**Date:** 2026-01-31
**Review:** Validated against mail-done codebase

---

## Executive Summary

Extend mail-done to index documents from folders (NAS, laptop, Pi) and email attachments, enabling semantic search across all documents. This proposal is structured in incremental phases, starting with the highest-value, lowest-risk changes.

---

## Architectural Principle: Parity with Email Storage Model

The document indexing system follows the **same architectural pattern as email storage**, ensuring consistent privacy guarantees and operational simplicity.

### Storage Model Equivalence

| Concept | Email System | Document System |
|---------|--------------|-----------------|
| **Source of truth** | IMAP server | File system (NAS/laptop/Pi) |
| **Raw content location** | Stays on IMAP | Stays on disk (reference only) |
| **Extracted text** | Body text → encrypted in DB | OCR/parsed text → encrypted in DB |
| **Plaintext metadata** | Subject (keyword searchable) | Filename + summary (keyword searchable) |
| **Semantic search** | Embedding from body+subject | Embedding from extracted text |
| **Binary storage** | Not copied locally | Not copied locally |
| **Content retrieval** | Fetch from IMAP on demand | Fetch from origin path on demand |

### Why This Matters

1. **Privacy consistency**: Sensitive content (email bodies, document text) is always encrypted at rest in the database. Only non-sensitive metadata (subject/filename) is stored in plaintext for keyword search.

2. **No binary duplication**: Just as we don't store `.eml` files locally, we don't copy document binaries. The original location (IMAP/filesystem) remains the source of truth.

3. **Deduplication via checksum**: Documents with identical SHA-256 checksums are stored once in the DB, with multiple origins tracked (same file found on NAS + laptop + email attachment).

4. **Operational simplicity**: No sync logic, no storage growth, no consistency issues between "cached copy" and original.

### Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                         EMAIL FLOW (existing)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  IMAP Server ──────► Extract body ──────► Encrypt ──────► DB        │
│  (source of truth)        │                   │                      │
│                           │                   ▼                      │
│                           │            encrypted_body                │
│                           │                                          │
│                           ▼                                          │
│                      Embed (body + subject) ──────► email_embeddings │
│                                                                      │
│  Plaintext in DB: subject, sender, date                              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                       DOCUMENT FLOW (proposed)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  File System ──────► Extract text ──────► Encrypt ──────► DB        │
│  (source of truth)   (OCR/parse)              │                      │
│       │                   │                   ▼                      │
│       │                   │            extracted_text_encrypted      │
│       │                   │                                          │
│       │                   ▼                                          │
│       │              Embed (text) ──────► document_embeddings        │
│       │                                                              │
│       ▼                                                              │
│  Track origin path (for later retrieval)                             │
│                                                                      │
│  Plaintext in DB: filename, summary, document_type, tags             │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Critical Design Decisions

### Decision 1: Content Encryption Policy

**Current state:** Email bodies and AI summaries are encrypted at rest using `EncryptedText` column types.

**Options for `documents.extracted_text`:**

| Option | Pros | Cons |
|--------|------|------|
| **A) Encrypted (recommended)** | Consistent with email security model | No keyword/tsvector search; semantic search only |
| **B) Plaintext** | Full-text search via tsvector/GIN | New sensitive plaintext corpus; requires compensating controls |
| **C) Separate index** | Best of both (encrypted DB + search index) | Added operational complexity |

**Recommendation:** Option A (encrypted) for initial implementation. Semantic search via embeddings is sufficient for most use cases. Add plaintext option later if needed with explicit opt-in.

### Decision 2: Document Lifecycle States

Documents go through clear extraction states:

```
┌──────────┐    ┌────────────┐    ┌───────────┐    ┌───────────────┐
│ pending  │───►│ processing │───►│ completed │    │  no_content   │
└──────────┘    └────────────┘    └───────────┘    └───────────────┘
                      │                                    ▲
                      │           ┌──────────┐             │
                      └──────────►│  failed  │             │
                                  └──────────┘             │
                                       │                   │
                                       └───── retry ───────┘
```

| Status | `extraction_status` | `extracted_text` | `extraction_quality` | Meaning |
|--------|---------------------|------------------|---------------------|---------|
| Registered | `pending` | NULL | NULL | Queued for extraction |
| Processing | `processing` | NULL | NULL | Worker is extracting |
| Success | `completed` | non-NULL | 0.0-1.0 | Has text, quality scored |
| No text | `no_content` | empty | 1.0 | Processed, no extractable text (photo, empty file) |
| Failed | `failed` | NULL | NULL | Extraction failed (will retry) |

**Note:** `no_content` documents still get embeddings generated from metadata (filename, path, tags). A photo named `receipt_restaurant.jpg` in `/expenses/` is searchable even without OCR.

This enables:
- Clear monitoring ("how many pending extractions?")
- Retry logic for failed extractions
- Distinguishing "not yet processed" from "processed but empty"

---

### Decision 3: Binary Storage Strategy

**Architectural principle:** Follow email pattern - source of truth stays at origin.

**Decision: Reference only (Option A)**

| Aspect | Email | Document |
|--------|-------|----------|
| Binary storage | None (IMAP is source) | None (filesystem is source) |
| What DB stores | Encrypted text + embedding | Encrypted text + embedding |
| Retrieval | Fetch from IMAP | Fetch from origin path |

**Why reference only:**
- Consistent with email architecture (we don't store `.eml` files)
- No storage overhead or sync complexity
- Origin tracking handles "same file in multiple places"
- If original is deleted, we still have: extracted text, embedding, metadata (searchable)

**Trade-off accepted:** If original file is deleted/moved, we can't retrieve binary. But:
- Extracted text and embedding remain (search still works)
- Origin tracking shows where it was last seen
- Same limitation exists for emails (delete from IMAP = can't retrieve)

### Decision 4: Embedding Model Consistency

**Current state:** Email embeddings use `text-embedding-3-large` (3072 dimensions).

**Requirement:** Document embeddings MUST use the same model/dimension to enable unified search.

```python
# Current email embedding config (backend/core/search/vector_search.py)
DEFAULT_MODEL = "text-embedding-3-large"  # 3072 dimensions
```

---

## Email ↔ Document Relationship

Documents and emails are connected bidirectionally via the `document_origins` table.

### How Attachments Become Documents

When processing an email with attachments:

```
Email (id: email-123)
├── Attachment 0: report.pdf (checksum: abc...)
├── Attachment 1: data.xlsx (checksum: def...)
└── Attachment 2: report.pdf (checksum: abc...)  ← Same as attachment 0!
```

Results in:

```
documents:
├── Document A (checksum: abc...) ← report.pdf (deduplicated)
└── Document B (checksum: def...) ← data.xlsx

document_origins:
├── Document A ← email-123, attachment_index=0
├── Document A ← email-123, attachment_index=2  ← Same doc, different attachment
└── Document B ← email-123, attachment_index=1
```

### Bidirectional Queries

| Query | SQL |
|-------|-----|
| **Document → Email(s)** | `SELECT email_id FROM document_origins WHERE document_id = ? AND email_id IS NOT NULL` |
| **Email → Document(s)** | `SELECT document_id FROM document_origins WHERE email_id = ?` |
| **Document → All origins** | `SELECT * FROM document_origins WHERE document_id = ?` |

### Search Scenarios

| User Wants | How It Works |
|------------|--------------|
| Search emails, find by attachment content | **Phase 0**: Attachment text included in email embedding |
| Search documents, see source email | Follow `document_origins.email_id` to show "From email: [subject]" |
| Find all emails with same document | Query `document_origins` by `document_id`, get all `email_id`s |
| "Show me the PDF that was in John's email" | Search emails → find email → get documents via origins |

### API Response Example

```json
// GET /api/documents/{id}
{
  "document": {
    "id": "doc-456",
    "title": "Q4 Financial Report",
    "summary": "Quarterly financial summary for Q4 2025",
    "checksum": "abc123...",
    "extraction_status": "completed",
    "extraction_quality": 0.95
  },
  "origins": [
    {
      "origin_type": "email_attachment",
      "email_id": "email-123",
      "email_subject": "Q4 Report - Please Review",  // Joined from emails table
      "email_from": "cfo@company.com",
      "attachment_index": 0,
      "discovered_at": "2025-10-15T14:30:00Z"
    },
    {
      "origin_type": "folder",
      "origin_host": "nas.local",
      "origin_path": "/documents/finance/2025/Q4-report.pdf",
      "discovered_at": "2025-10-20T09:00:00Z"
    }
  ]
}
```

This shows the same document was found both as an email attachment and on the NAS.

---

## Document Retrieval

Since binaries stay at origin (reference-only model), we need a retrieval mechanism.

### Origin Types and Retrieval Methods

| Origin Type | Stored Info | Retrieval Method |
|-------------|-------------|------------------|
| `folder` | `origin_host`, `origin_path` | File system access (local or network mount) |
| `email_attachment` | `email_id`, `attachment_index` | Fetch from IMAP via existing attachment API |
| `google_drive` | `origin_path` (Drive file ID) | Google Drive API |

### Database Schema for Retrieval

```sql
-- document_origins already has:
origin_type VARCHAR(50),      -- 'folder', 'email_attachment', 'google_drive'
origin_host VARCHAR(255),     -- 'nas.local', 'laptop', 'nvme-pi', 'imap.gmail.com'
origin_path TEXT,             -- Full path or Drive ID

-- For email attachments:
email_id UUID,                -- FK to emails table
attachment_index INTEGER,     -- Which attachment (0-indexed)
```

### Multi-Host Access Strategy

**Assumption:** The API/MCP server runs on a host with SSH access to all document source hosts.

```
┌─────────────────────────────────────────────────────────────┐
│                    API/MCP Server (nvme-pi)                  │
│                                                              │
│   Has SSH access to:                                         │
│   ├── nas.local (network mount or SSH)                       │
│   ├── laptop.local (SSH)                                     │
│   ├── backup-server (SSH)                                    │
│   └── IMAP server (for email attachments)                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

| Host Type | Access Method |
|-----------|---------------|
| **Local filesystem** | Direct file read (API server's local files) |
| **Network mount (NAS)** | Mount point on API server, read like local |
| **Remote host** | SSH/SCP from API server to remote host |
| **IMAP (email attachments)** | Existing `AttachmentExtractor` fetches from IMAP |
| **Google Drive** | Drive API with OAuth |

### Configuration

```yaml
# config/document_hosts.yaml

# The API/MCP server (e.g., nvme-pi) has SSH access to all hosts
default_ssh_user: raetsch          # Default SSH user if not specified per host

hosts:
  nvme-pi:
    type: local                     # API server runs here - direct file access
    base_path: /

  nas.local:
    type: network_mount
    mount_point: /mnt/nas           # NAS mounted on API server

  laptop:
    type: ssh
    ssh_host: laptop.local
    ssh_user: raetsch               # Or uses default_ssh_user
    # ssh_key: ~/.ssh/id_ed25519   # Optional, defaults to SSH agent

  macbook:
    type: ssh
    ssh_host: macbook.local

  imap:
    type: imap                      # Email attachments via IMAP
    # Uses credentials from accounts.yaml
```

### API Endpoints

```python
# GET /api/documents/{id}/content
# Returns the actual file binary

@router.get("/{document_id}/content")
async def get_document_content(
    document_id: UUID,
    origin_index: int = 0,          # Which origin to fetch from (if multiple)
):
    """Retrieve original document binary."""

    doc = await db.get(Document, document_id)
    origins = await db.query(DocumentOrigin).filter_by(
        document_id=document_id,
        is_deleted=False
    ).order_by(DocumentOrigin.is_primary.desc()).all()

    if not origins:
        raise HTTPException(404, "No accessible origins for document")

    origin = origins[origin_index]

    # Dispatch to appropriate retrieval method
    if origin.origin_type == 'folder':
        content = await retrieve_from_filesystem(origin)
    elif origin.origin_type == 'email_attachment':
        content = await retrieve_from_imap(origin)
    elif origin.origin_type == 'google_drive':
        content = await retrieve_from_drive(origin)

    return Response(
        content=content,
        media_type=doc.mime_type,
        headers={
            "Content-Disposition": f'attachment; filename="{doc.original_filename}"'
        }
    )


async def retrieve_from_filesystem(origin: DocumentOrigin) -> bytes:
    """Retrieve file from local, network-mounted, or remote filesystem."""

    host_config = get_host_config(origin.origin_host)

    if host_config.type == 'local':
        path = origin.origin_path
        if not os.path.exists(path):
            raise HTTPException(404, f"File not found: {path}")
        with open(path, 'rb') as f:
            return f.read()

    elif host_config.type == 'network_mount':
        # Translate path: /nas/documents/file.pdf → /mnt/nas/documents/file.pdf
        path = translate_path(origin.origin_path, host_config.mount_point)
        if not os.path.exists(path):
            raise HTTPException(404, f"File not found at mount: {path}")
        with open(path, 'rb') as f:
            return f.read()

    elif host_config.type == 'ssh':
        return await fetch_via_ssh(
            host=host_config.get('ssh_host', origin.origin_host),
            user=host_config.get('ssh_user', get_default_ssh_user()),
            remote_path=origin.origin_path,
            key_file=host_config.get('ssh_key')
        )

    raise HTTPException(400, f"Unknown host type: {host_config.type}")


async def fetch_via_ssh(host: str, user: str, remote_path: str, key_file: str = None) -> bytes:
    """Fetch file from remote host via SSH/SCP."""
    import asyncio
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Build scp command
        cmd = ['scp', '-q']
        if key_file:
            cmd.extend(['-i', key_file])
        cmd.append(f'{user}@{host}:{remote_path}')
        cmd.append(tmp_path)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        _, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise HTTPException(404, f"SSH fetch failed: {stderr.decode()}")

        with open(tmp_path, 'rb') as f:
            return f.read()
    finally:
        os.unlink(tmp_path)


async def retrieve_from_imap(origin: DocumentOrigin) -> bytes:
    """Retrieve attachment from IMAP (reuse existing logic)."""

    # Reuse existing attachment retrieval
    from backend.core.email.attachment_extractor import AttachmentExtractor

    extractor = AttachmentExtractor()
    return await extractor.download_attachment(
        email_id=origin.email_id,
        attachment_index=origin.attachment_index
    )
```

### MCP Tool for Document Retrieval

```python
# MCP tool definition
{
    "name": "get_document_content",
    "description": "Retrieve the original binary content of an indexed document",
    "parameters": {
        "document_id": {"type": "string", "description": "Document UUID"},
        "save_to": {"type": "string", "description": "Optional: local path to save file"}
    }
}
```

### Handling Unavailable Origins

```python
async def get_document_with_availability(document_id: UUID):
    """Check which origins are currently accessible."""

    doc = await db.get(Document, document_id)
    origins = await db.query(DocumentOrigin).filter_by(document_id=document_id).all()

    result = []
    for origin in origins:
        accessible = await check_origin_accessible(origin)
        result.append({
            **origin.to_dict(),
            "accessible": accessible,
            "last_verified": origin.last_verified_at
        })

    return {"document": doc, "origins": result}
```

This allows the UI/client to show which copies of the document are currently reachable.

---

## Implementation Phases

### Phase 0: Attachment Text in Email Embeddings (Fastest Value)

**Goal:** Make attachment content searchable via existing email semantic search—no new schema required.

**Current gap:** `backend/core/search/embeddings.py` builds embedding text from subject + body + AI summary, but ignores `ProcessedEmail.attachment_texts` even though extraction already happens.

**Changes:**

```python
# backend/core/search/embeddings.py - prepare_email_for_embedding()
def prepare_email_for_embedding(email: ProcessedEmail) -> str:
    parts = [
        f"Subject: {email.subject}",
        f"From: {email.sender}",
        email.body_text[:MAX_BODY_CHARS],
    ]

    # NEW: Include attachment text (bounded)
    if email.attachment_texts:
        for i, att_text in enumerate(email.attachment_texts[:MAX_ATTACHMENTS]):
            if att_text:
                parts.append(f"Attachment {i+1}: {att_text[:MAX_ATTACHMENT_CHARS]}")

    if email.ai_summary:
        parts.append(f"Summary: {email.ai_summary}")

    return "\n\n".join(parts)
```

**Optional enhancement:** Persist attachment text encrypted for later re-embedding:

```sql
-- Option: New table for attachment text persistence
CREATE TABLE email_attachment_texts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email_id UUID NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    attachment_index INTEGER NOT NULL,
    extracted_text_encrypted BYTEA,  -- Encrypted via Fernet
    extraction_method VARCHAR(50),
    extraction_quality FLOAT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(email_id, attachment_index)
);
```

**Effort:** 1-2 days
**Value:** Immediate semantic search over attachment content

---

### Phase 1: Documents as First-Class Objects

**Goal:** Add document storage with checksum-based deduplication and origin tracking.

#### 1.1 Database Schema

```sql
-- Core document table (one row per unique document by checksum)
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Identity (deduplication key)
    checksum VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256
    checksum_algorithm VARCHAR(20) DEFAULT 'sha256',

    -- File metadata
    file_size BIGINT NOT NULL,
    mime_type VARCHAR(127),
    original_filename VARCHAR(500),
    page_count INTEGER,

    -- Extracted text (ENCRYPTED - consistent with email body treatment)
    extracted_text_encrypted BYTEA,        -- Fernet encrypted

    -- Extraction state machine
    extraction_status VARCHAR(20) DEFAULT 'pending',  -- pending/processing/completed/no_content/failed

    -- Extraction metadata (not sensitive - plaintext OK)
    extraction_version VARCHAR(20),
    extraction_method VARCHAR(50),         -- 'sandboxed', 'pdftotext', 'tesseract', 'claude'
    extraction_model VARCHAR(100),         -- LLM model if used
    extraction_quality FLOAT,              -- 0.0-1.0
    extraction_cost FLOAT,                 -- API cost in USD
    extracted_at TIMESTAMP WITH TIME ZONE,

    -- Plaintext metadata (like email subject - keyword searchable)
    title VARCHAR(500),                    -- Extracted or filename-derived
    summary VARCHAR(1000),                 -- One-line description ("subject" equivalent)
    document_date DATE,
    document_type VARCHAR(100),            -- 'invoice', 'contract', 'letter'
    language VARCHAR(10),
    ai_category VARCHAR(100),
    ai_tags TEXT[],

    -- NOTE: No binary storage columns - files stay at origin (like emails stay on IMAP)
    -- Use document_origins.origin_path to retrieve original file

    -- Lifecycle
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_deleted BOOLEAN DEFAULT FALSE,

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes on plaintext metadata (not on encrypted content)
CREATE INDEX idx_documents_checksum ON documents(checksum);
CREATE INDEX idx_documents_mime_type ON documents(mime_type);
CREATE INDEX idx_documents_document_type ON documents(document_type);
CREATE INDEX idx_documents_document_date ON documents(document_date);
CREATE INDEX idx_documents_extraction_quality ON documents(extraction_quality);
CREATE INDEX idx_documents_ai_category ON documents(ai_category);
CREATE INDEX idx_documents_ai_tags ON documents USING GIN(ai_tags);

-- Trigram index for keyword search on title/summary (like email subject search)
CREATE INDEX idx_documents_title_trgm ON documents USING GIN(title gin_trgm_ops);
CREATE INDEX idx_documents_summary_trgm ON documents USING GIN(summary gin_trgm_ops);

-- Index for extraction status monitoring
CREATE INDEX idx_documents_extraction_status ON documents(extraction_status);

-- Partial index for documents needing processing
CREATE INDEX idx_documents_needs_extraction ON documents(id)
    WHERE extraction_status IN ('pending', 'failed');
```

```sql
-- Track where each document was found (and how to retrieve it)
-- This is the equivalent of IMAP folder/UID for emails
CREATE TABLE document_origins (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Origin identification (used to retrieve original file)
    origin_type VARCHAR(50) NOT NULL,      -- 'folder', 'email_attachment', 'google_drive'
    origin_host VARCHAR(255),              -- 'nas.local', 'laptop', 'nvme-pi'
    origin_path TEXT,                      -- Full path to file (for retrieval)
    origin_filename VARCHAR(500),          -- Filename at this location

    -- For email attachments specifically
    email_id UUID REFERENCES emails(id) ON DELETE SET NULL,
    attachment_index INTEGER,

    -- Discovery metadata
    file_modified_at TIMESTAMP WITH TIME ZONE,
    discovered_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_verified_at TIMESTAMP WITH TIME ZONE,
    is_primary BOOLEAN DEFAULT FALSE,

    -- Status
    is_deleted BOOLEAN DEFAULT FALSE,
    deleted_at TIMESTAMP WITH TIME ZONE,

    UNIQUE(document_id, origin_type, origin_host, origin_path)
);

CREATE INDEX idx_document_origins_document ON document_origins(document_id);
CREATE INDEX idx_document_origins_email ON document_origins(email_id);
CREATE INDEX idx_document_origins_host ON document_origins(origin_host);
```

```sql
-- Vector embeddings (MUST match email embedding dimensions)
CREATE TABLE document_embeddings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Embedding (same dimension as email_embeddings!)
    embedding vector(3072),                -- text-embedding-3-large

    -- Page-level granularity (especially useful for scanned docs)
    page_number INTEGER,                   -- NULL for non-paginated, 1-indexed for pages

    -- Within-page chunking (for very long pages)
    chunk_index INTEGER DEFAULT 0,         -- 0 = first/only chunk on page
    chunk_start INTEGER,                   -- Character offset within page
    chunk_end INTEGER,
    chunk_text_encrypted BYTEA,            -- Encrypted chunk text

    -- Model info (must match email embedding model)
    model VARCHAR(100) NOT NULL DEFAULT 'text-embedding-3-large',
    model_version VARCHAR(50),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Unique per document + page + chunk
    UNIQUE(document_id, page_number, chunk_index)
);

-- Vector index (same type as email_embeddings)
CREATE INDEX idx_document_embeddings_vector
    ON document_embeddings
    USING diskann (embedding vector_cosine_ops)
    WITH (num_neighbors = 50);
```

```sql
-- Processing queue for async extraction/embedding
CREATE TABLE document_processing_queue (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,

    task_type VARCHAR(50) NOT NULL,        -- 'extract_text', 'generate_embedding', 'classify'
    priority INTEGER DEFAULT 5,

    status VARCHAR(20) DEFAULT 'pending',  -- 'pending', 'processing', 'completed', 'failed'
    attempts INTEGER DEFAULT 0,
    max_attempts INTEGER DEFAULT 3,
    last_error TEXT,

    scheduled_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    worker_id VARCHAR(100),

    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_processing_queue_pending
    ON document_processing_queue(priority, scheduled_at)
    WHERE status = 'pending';
```

#### 1.2 SQLAlchemy Models

```python
# backend/core/database/models.py additions

class Document(Base):
    __tablename__ = "documents"

    id = Column(UUID, primary_key=True, default=uuid.uuid4)
    checksum = Column(String(64), unique=True, nullable=False, index=True)
    checksum_algorithm = Column(String(20), default="sha256")

    file_size = Column(BigInteger, nullable=False)
    mime_type = Column(String(127))
    original_filename = Column(String(500))
    page_count = Column(Integer)

    # Encrypted content (reuse existing EncryptedText type)
    extracted_text = Column(EncryptedText)

    # Extraction state
    extraction_status = Column(String(20), default="pending")  # pending/processing/completed/no_content/failed

    # Extraction metadata
    extraction_version = Column(String(20))
    extraction_method = Column(String(50))
    extraction_model = Column(String(100))
    extraction_quality = Column(Float)
    extraction_cost = Column(Float)
    extracted_at = Column(DateTime(timezone=True))

    # Plaintext metadata (keyword searchable, like email subject)
    title = Column(String(500))
    summary = Column(String(1000))         # One-line description
    document_date = Column(Date)
    document_type = Column(String(100))
    language = Column(String(10))
    ai_category = Column(String(100))
    ai_tags = Column(ARRAY(String))

    # NOTE: No storage columns - files stay at origin (reference only)
    # Retrieve via: origins[0].origin_path

    # Lifecycle
    first_seen_at = Column(DateTime(timezone=True), default=func.now())
    last_seen_at = Column(DateTime(timezone=True), default=func.now())
    is_deleted = Column(Boolean, default=False)

    # Relationships
    origins = relationship("DocumentOrigin", back_populates="document")
    embeddings = relationship("DocumentEmbedding", back_populates="document")
```

#### 1.3 Reuse Existing Extractors

**Key insight from review:** mail-done already has sandboxed extraction for attachments.

```python
# Reuse existing extraction (backend/core/email/sandboxed_extractor.py)
from backend.core.email.sandboxed_extractor import SandboxedExtractor

class DocumentProcessor:
    def __init__(self):
        self.extractor = SandboxedExtractor()  # Already handles PDF, DOCX, XLSX, etc.

    async def extract_text(self, file_path: str, mime_type: str) -> ExtractionResult:
        """Use existing sandboxed extractor for supported types."""

        # Read file bytes
        with open(file_path, 'rb') as f:
            content = f.read()

        # Reuse existing extraction logic
        text = await self.extractor.extract_text(
            content=content,
            content_type=mime_type,
            filename=os.path.basename(file_path)
        )

        return ExtractionResult(
            text=text,
            method='sandboxed',
            quality_score=self._score_quality(text)
        )
```

---

### Phase 2: Folder Scanning

**Goal:** Implement `process_folders.py` for indexing local/network folders.

```python
# process_folders.py - CLI tool

"""
Usage:
    python process_folders.py /nas/documents --recursive --host nas.local
    python process_folders.py ~/Downloads --extensions pdf,docx --dry-run
    python process_folders.py --config config/document_folders.yaml
"""

@click.command()
@click.argument('path', required=False)
@click.option('--recursive/--no-recursive', default=True)
@click.option('--host', default=socket.gethostname())
@click.option('--extensions', help='Comma-separated: pdf,docx,xlsx')
@click.option('--config', help='YAML config file')
@click.option('--dry-run', is_flag=True)
@click.option('--limit', type=int, help='Max files to process')
def scan_folder(path, recursive, host, extensions, config, dry_run, limit):
    """Scan folder and index documents."""

    processor = FolderProcessor(db_session, dry_run=dry_run)

    for file_path in discover_files(path, recursive, extensions):
        checksum = calculate_sha256(file_path)

        existing = db.query(Document).filter_by(checksum=checksum).first()

        if existing:
            # Document exists - just add origin
            processor.add_origin(existing, file_path, host)
            logger.info(f"Duplicate: {file_path} -> existing document")
        else:
            # New document
            doc = processor.create_document(file_path, checksum)
            processor.add_origin(doc, file_path, host)
            processor.queue_extraction(doc)
            logger.info(f"New: {file_path}")
```

**Configuration file:**

```yaml
# config/document_folders.yaml
folders:
  - path: /nas/documents
    host: nas.local
    recursive: true
    extensions: [pdf, docx, xlsx, pptx, txt, md]
    exclude_patterns:
      - "*.tmp"
      - ".DS_Store"
      - "~$*"

  - path: /home/user/Scans
    host: laptop
    extensions: [pdf, jpg, png]
    scan_interval: 30m

defaults:
  max_file_size: 100MB
  skip_hidden: true
```

---

### Phase 3: OCR Processing (Optional, Opt-in)

**Goal:** Add OCR capability for scanned documents and images.

**Important:** This phase is **optional** and requires explicit configuration. It adds deployment dependencies (Tesseract, poppler-utils) and API costs (Claude vision).

#### 3.1 Enable OCR

```bash
# Environment variable to enable OCR
DOCUMENT_OCR_ENABLED=true
DOCUMENT_OCR_QUALITY_THRESHOLD=0.7
DOCUMENT_OCR_USE_CLAUDE_FALLBACK=true
```

#### 3.2 OCR Pipeline

```python
# backend/core/documents/ocr.py

class HybridOCRPipeline:
    """
    OCR pipeline with cost controls.

    Strategy:
    1. pdftotext for text-based PDFs (free, fast)
    2. Tesseract for images (free, local)
    3. Claude Vision for low-quality results (paid, optional)
    """

    def __init__(self, config: OCRConfig):
        self.quality_threshold = config.quality_threshold
        self.use_claude_fallback = config.use_claude_fallback
        self.max_cost_per_doc = config.max_cost_per_doc

    async def extract(self, file_path: str, mime_type: str) -> ExtractionResult:
        if mime_type == 'application/pdf':
            return await self._extract_pdf(file_path)
        elif mime_type.startswith('image/'):
            return await self._extract_image(file_path)
        else:
            raise ValueError(f"OCR not supported for {mime_type}")

    async def _extract_pdf(self, file_path: str) -> ExtractionResult:
        # Try pdftotext first
        text = await self._pdftotext(file_path)
        quality = self._score_quality(text)

        if quality >= self.quality_threshold:
            return ExtractionResult(text=text, method='pdftotext', quality_score=quality)

        # Fall back to Claude if enabled
        if self.use_claude_fallback:
            return await self._claude_ocr(file_path, existing_text=text)

        return ExtractionResult(text=text, method='pdftotext', quality_score=quality)
```

#### 3.3 System Dependencies (OCR Only)

**Important:** These dependencies are ONLY needed if `DOCUMENT_OCR_ENABLED=true`. The base system uses the existing sandboxed Python extractors and requires no additional system packages.

```dockerfile
# Dockerfile additions for OCR support (optional)
# Only add if DOCUMENT_OCR_ENABLED=true

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-deu \
    tesseract-ocr-eng \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Note: LibreOffice for .doc/.xls/.ppt is NOT included by default
# due to large image size and security surface. Use Python libraries
# (python-docx, openpyxl, python-pptx) for modern Office formats.
```

**Deployment impact of enabling OCR:**

| Aspect | Without OCR | With OCR |
|--------|-------------|----------|
| Docker image size | ~500MB | ~800MB |
| System packages | None additional | tesseract, poppler-utils |
| API costs | $0 | Claude Vision for fallback |
| Security surface | Python libraries only | External binaries (sandboxed) |

#### 3.4 Cost Tracking

```python
# Track extraction costs
@dataclass
class ExtractionResult:
    text: str
    method: str
    quality_score: float
    model_used: Optional[str] = None
    cost_usd: float = 0.0
    tokens_used: int = 0
```

---

### Phase 3.5: Embedding Generation Strategy

**Page-level embeddings** for multi-page documents:

```python
async def generate_embeddings(doc: Document, pages: list[PageContent]) -> list[Embedding]:
    """Generate embeddings per page for granular search."""

    embeddings = []

    for page in pages:
        text = prepare_page_for_embedding(doc, page)

        if token_count(text) <= MAX_TOKENS:
            # Single embedding for page
            embeddings.append(create_embedding(
                text=text,
                page_number=page.number,
                chunk_index=0
            ))
        else:
            # Split page into chunks
            for i, chunk in enumerate(chunk_text(text, MAX_TOKENS)):
                embeddings.append(create_embedding(
                    text=chunk,
                    page_number=page.number,
                    chunk_index=i
                ))

    return embeddings
```

**Embedding from metadata** (for `no_content` documents):

Even documents without extractable text get useful embeddings from metadata:

```python
def prepare_document_for_embedding(doc: Document) -> str:
    """Build embedding text from whatever we have."""
    parts = []

    # Always have filename (often semantic: "Q4_Financial_Report.pdf")
    parts.append(f"Filename: {doc.original_filename}")

    # Path contains semantic info ("/finance/invoices/2025/")
    if doc.origins:
        primary = next((o for o in doc.origins if o.is_primary), doc.origins[0])
        if primary.origin_path:
            parts.append(f"Location: {primary.origin_path}")

    # Extracted text if available
    if doc.extracted_text:
        parts.append(doc.extracted_text)

    # Summary if generated (AI or manual)
    if doc.summary:
        parts.append(f"Summary: {doc.summary}")

    # Classification metadata
    if doc.document_type:
        parts.append(f"Type: {doc.document_type}")
    if doc.ai_tags:
        parts.append(f"Tags: {', '.join(doc.ai_tags)}")

    return "\n\n".join(parts)
```

This means:
- `receipt_restaurant_zurich.jpg` in `/expenses/meals/` → searchable by "restaurant expense zurich"
- Empty PDF with meaningful filename → still findable
- Scanned image before OCR runs → searchable by filename/path

---

### Phase 4: Document Search API

**Goal:** Add document-specific search endpoints. Unified search comes later.

```python
# backend/api/routes/documents.py

router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.get("/search/semantic")
async def semantic_search(
    q: str,
    limit: int = 20,
    min_quality: float = 0.5,
    document_type: Optional[str] = None,
):
    """Semantic search over documents (uses same embedding model as emails)."""

    query_embedding = await embedding_service.embed(q)

    results = await document_search.semantic_search(
        embedding=query_embedding,
        limit=limit,
        filters={
            "extraction_quality": {"gte": min_quality},
            "document_type": document_type,
        }
    )

    return {"results": results}


@router.get("/{document_id}")
async def get_document(document_id: UUID):
    """Get document with all origins."""
    doc = await db.get(Document, document_id)
    origins = await db.query(DocumentOrigin).filter_by(document_id=document_id).all()
    return {"document": doc, "origins": origins}


@router.get("/{document_id}/similar")
async def find_similar(document_id: UUID, limit: int = 10):
    """Find documents similar to this one."""
    doc = await db.get(Document, document_id)
    embedding = await db.query(DocumentEmbedding).filter_by(document_id=doc.id).first()

    return await document_search.find_similar(embedding.embedding, limit=limit)
```

---

### Phase 5: Unified Search (Future)

**Deferred until document search is validated.**

When ready, add:
- `/api/search?types=email,document` unified endpoint
- Cross-corpus ranking strategy
- Union response types
- Pagination across corpora

---

## Configuration Reference

### Environment Variables

```bash
# Core document indexing (reference-only model - no binary storage)
DOCUMENT_MAX_FILE_SIZE=100MB                # Skip files larger than this
DOCUMENT_SUPPORTED_EXTENSIONS=pdf,docx,xlsx,pptx,txt,md

# Processing
DOCUMENT_PROCESS_WORKERS=4
DOCUMENT_EXTRACTION_TIMEOUT=60

# OCR (Phase 3 - optional)
DOCUMENT_OCR_ENABLED=false                  # Must opt-in
DOCUMENT_OCR_QUALITY_THRESHOLD=0.7
DOCUMENT_OCR_USE_CLAUDE_FALLBACK=false
DOCUMENT_OCR_MAX_COST_PER_DOC=0.50
DOCUMENT_OCR_DAILY_BUDGET=50.00
```

### Python Dependencies

```toml
# pyproject.toml - Phase 1 (uses existing deps where possible)
# No new dependencies for basic extraction - reuses sandboxed_extractor

# Remote host access (optional - can use system scp instead)
paramiko = { version = "^3.4", optional = true }  # Pure Python SSH

# Phase 3 OCR additions (optional)
pytesseract = { version = "^0.3.10", optional = true }
pdf2image = { version = "^1.16.3", optional = true }

[tool.poetry.extras]
ssh = ["paramiko"]
ocr = ["pytesseract", "pdf2image"]
```

---

## Migration Path

### For Existing Deployments

```bash
# 1. Run database migration
./scripts/db_migrate.sh

# 2. (Optional) Backfill attachment text into email embeddings
python -m backend.scripts.backfill_attachment_embeddings --since 2024-01-01

# 3. Initial folder scan
python process_folders.py /path/to/documents --host $(hostname)

# 4. (Optional) Backfill email attachments as documents
python process_inbox.py --backfill-attachments --since 2024-01-01
```

---

## Open Questions Resolved

| Question | Decision |
|----------|----------|
| Encryption vs keyword search | **Encrypted text** - semantic search on content; keyword search on title/summary only |
| Binary storage | **Reference only** - files stay at origin (like emails stay on IMAP) |
| Keyword searchable fields | **Title + summary** - equivalent to email subject |
| Embedding model | **text-embedding-3-large (3072)** - must match emails |
| Unified search timing | **Phase 5** - after document search is proven |
| OCR dependencies | **Optional opt-in** - Phase 3 with explicit config |
| Document lifecycle states | **Explicit enum** - pending/processing/completed/no_content/failed |
| Multi-page documents | **Page-level embeddings** - one embedding per page (+ chunking for long pages) |
| Documents without text | **Embed from metadata** - filename, path, tags still create useful embeddings |
| Email ↔ Document link | **Bidirectional via origins** - document_origins.email_id links both ways |
| Document retrieval | **Via API/MCP** - uses origin_host config (local, mount, SSH, IMAP) |

---

## Implementation Notes

Based on final review feedback:

### Migration Safety (Phase 1)

Ensure Alembic migration includes pgvector index creation, following the pattern in `001_initial_schema.py`:

```python
# In alembic/versions/002_documents.py
def upgrade():
    # Create tables first
    op.create_table('documents', ...)
    op.create_table('document_origins', ...)
    op.create_table('document_embeddings', ...)

    # Then create vector index (requires pgvector extension)
    op.execute("""
        CREATE INDEX idx_document_embeddings_vector
        ON document_embeddings
        USING diskann (embedding vector_cosine_ops)
        WITH (num_neighbors = 50)
    """)
```

### Rate Limiting (Phase 2+)

Background workers for folder scanning and OCR must respect existing rate limiting:

```python
# In document_worker.py
from backend.api.rate_limiting import RateLimiter

class DocumentWorker:
    def __init__(self):
        self.ocr_limiter = RateLimiter(
            max_requests=100,
            window_seconds=3600,  # 100/hour for Claude API
            key="document_ocr"
        )

    async def process_with_ocr(self, doc):
        await self.ocr_limiter.acquire()
        # ... OCR processing
```

### Testing (Phase 0)

Start with unit test verifying attachment text inclusion:

```python
# backend/tests/unit/test_embeddings.py
def test_attachment_text_included_in_embedding():
    """Phase 0: Verify attachment text is in embedding input."""
    email = ProcessedEmail(
        subject="Contract Review",
        body_text="Please review attached.",
        attachment_texts=["This is the contract content..."]
    )

    embedding_text = prepare_email_for_embedding(email)

    assert "contract content" in embedding_text.lower()
    assert "Attachment 1:" in embedding_text
```

---

## Success Metrics

- **Phase 0:** Attachment content appears in email semantic search results
- **Phase 1:** Documents indexed with deduplication working across origins
- **Phase 2:** Folder scanning completes incrementally (mtime+checksum cache)
- **Phase 3:** OCR quality >0.7 for 90%+ of scanned documents
- **Phase 4:** Document search latency <500ms
- **Phase 5:** Unified search ranks results meaningfully across corpora
