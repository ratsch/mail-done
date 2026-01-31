# Document Indexing Implementation Plan

**Status:** Ready for Implementation
**Created:** 2026-01-31
**Based on:** `docs/proposals/DOCUMENT_INDEXING.md` (approved)

---

## Overview

This plan breaks the document indexing feature into concrete implementation tasks organized by phase. Each task includes files to modify, acceptance criteria, and dependencies.

**Estimated Total Effort:** 4-6 weeks (one developer)

---

## Phase 0: Attachment Text in Email Embeddings

**Goal:** Make attachment content searchable via existing email semantic search
**Effort:** 1-2 days
**Risk:** Low (minimal code change, high value)

### Task 0.1: Modify Embedding Text Preparation

**Files to modify:**
- `backend/core/search/embeddings.py`

**Changes:**
```python
# Add constants
MAX_ATTACHMENT_TEXT_CHARS = 5000  # Per attachment
MAX_ATTACHMENTS_FOR_EMBEDDING = 5

# Modify prepare_email_for_embedding()
def prepare_email_for_embedding(email: ProcessedEmail) -> str:
    parts = [
        f"Subject: {email.subject}",
        f"From: {email.sender}",
        email.body_text[:MAX_BODY_CHARS],
    ]

    # NEW: Include attachment text
    if hasattr(email, 'attachment_texts') and email.attachment_texts:
        for i, att_text in enumerate(email.attachment_texts[:MAX_ATTACHMENTS_FOR_EMBEDDING]):
            if att_text and att_text.strip():
                truncated = att_text[:MAX_ATTACHMENT_TEXT_CHARS]
                parts.append(f"Attachment {i+1}: {truncated}")

    if email.ai_summary:
        parts.append(f"Summary: {email.ai_summary}")

    return "\n\n".join(parts)
```

**Acceptance criteria:**
- [ ] Unit test passes: attachment text appears in embedding input
- [ ] Integration test: email with PDF attachment is findable by PDF content
- [ ] No regression in existing email search

### Task 0.2: Add Unit Test

**Files to create:**
- `backend/tests/unit/search/test_embeddings_attachments.py`

**Test cases:**
```python
def test_attachment_text_included():
    """Attachment text should be in embedding input."""

def test_attachment_text_truncated():
    """Long attachment text should be truncated."""

def test_multiple_attachments_limited():
    """Only first N attachments included."""

def test_no_attachments_still_works():
    """Emails without attachments work as before."""
```

**Acceptance criteria:**
- [ ] All tests pass
- [ ] Coverage for new code paths

### Task 0.3: Verify ProcessedEmail Has attachment_texts

**Files to check:**
- `backend/core/email/processor.py`
- `backend/core/email/models.py` (ProcessedEmail dataclass)

**Verify:**
- [ ] `ProcessedEmail.attachment_texts` field exists
- [ ] Field is populated during email processing
- [ ] If not populated, add extraction call

### Task 0.4: Backfill Script (Optional)

**Files to create:**
- `backend/scripts/backfill_attachment_embeddings.py`

**Functionality:**
```bash
# Re-generate embeddings for emails with attachments
python -m backend.scripts.backfill_attachment_embeddings \
    --since 2024-01-01 \
    --batch-size 100 \
    --dry-run
```

**Acceptance criteria:**
- [ ] Script identifies emails with attachments lacking updated embeddings
- [ ] Regenerates embeddings with attachment text included
- [ ] Respects rate limits

---

## Phase 1: Documents as First-Class Objects

**Goal:** Create document storage with deduplication and origin tracking
**Effort:** 1-2 weeks
**Risk:** Medium (new schema, but follows existing patterns)
**Depends on:** Phase 0 complete (for consistency)

### Task 1.1: Create SQLAlchemy Models

**Files to create:**
- `backend/core/documents/__init__.py`
- `backend/core/documents/models.py`

**Models:**
```python
class Document(Base):
    __tablename__ = "documents"
    # As defined in proposal

class DocumentOrigin(Base):
    __tablename__ = "document_origins"
    # As defined in proposal

class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"
    # As defined in proposal

class DocumentProcessingQueue(Base):
    __tablename__ = "document_processing_queue"
    # As defined in proposal
```

**Acceptance criteria:**
- [ ] Models match proposal schema exactly
- [ ] `EncryptedText` used for `extracted_text`
- [ ] Relationships defined (Document → Origins, Embeddings)
- [ ] Models importable from `backend.core.documents`

### Task 1.2: Create Alembic Migration

**Files to create:**
- `alembic/versions/002_documents.py`

**Migration includes:**
```python
def upgrade():
    # Create enum for extraction_status
    extraction_status_enum = sa.Enum(
        'pending', 'processing', 'completed', 'no_content', 'failed',
        name='extraction_status'
    )

    # Create documents table
    op.create_table('documents', ...)

    # Create document_origins table
    op.create_table('document_origins', ...)

    # Create document_embeddings table
    op.create_table('document_embeddings', ...)

    # Create document_processing_queue table
    op.create_table('document_processing_queue', ...)

    # Create indexes (including vector index)
    op.execute("""
        CREATE INDEX idx_document_embeddings_vector
        ON document_embeddings
        USING diskann (embedding vector_cosine_ops)
        WITH (num_neighbors = 50)
    """)

    # Create trigram indexes for title/summary search
    op.execute("""
        CREATE INDEX idx_documents_title_trgm
        ON documents USING GIN(title gin_trgm_ops)
    """)

def downgrade():
    op.drop_table('document_processing_queue')
    op.drop_table('document_embeddings')
    op.drop_table('document_origins')
    op.drop_table('documents')
    op.execute("DROP TYPE extraction_status")
```

**Acceptance criteria:**
- [ ] Migration runs successfully on fresh DB
- [ ] Migration runs successfully on existing DB (nvme-pi)
- [ ] Downgrade works cleanly
- [ ] All indexes created including vector index

### Task 1.3: Create Document Repository

**Files to create:**
- `backend/core/documents/repository.py`

**Methods:**
```python
class DocumentRepository:
    async def create_document(self, checksum: str, ...) -> Document
    async def get_by_checksum(self, checksum: str) -> Optional[Document]
    async def get_by_id(self, document_id: UUID) -> Optional[Document]
    async def add_origin(self, document_id: UUID, origin: DocumentOriginCreate) -> DocumentOrigin
    async def get_origins(self, document_id: UUID) -> list[DocumentOrigin]
    async def update_extraction(self, document_id: UUID, text: str, quality: float, ...)
    async def queue_task(self, document_id: UUID, task_type: str, priority: int = 5)
    async def get_pending_tasks(self, task_type: str, limit: int) -> list[DocumentProcessingQueue]
    async def mark_task_processing(self, task_id: UUID, worker_id: str)
    async def mark_task_completed(self, task_id: UUID)
    async def mark_task_failed(self, task_id: UUID, error: str)
```

**Acceptance criteria:**
- [ ] All CRUD operations work
- [ ] Checksum uniqueness enforced
- [ ] Origin deduplication works (same doc, same origin = no duplicate)
- [ ] Task queue operations atomic

### Task 1.4: Create Document Processor

**Files to create:**
- `backend/core/documents/processor.py`

**Functionality:**
```python
class DocumentProcessor:
    def __init__(self, repository: DocumentRepository):
        self.repository = repository
        self.extractor = SandboxedExtractor()  # Reuse existing

    async def register_document(
        self,
        file_path: str,
        origin_type: str,
        origin_host: str,
        email_id: Optional[UUID] = None,
        attachment_index: Optional[int] = None
    ) -> tuple[Document, bool]:
        """
        Register a document. Returns (document, is_new).
        If document exists (by checksum), just adds origin.
        """

    async def extract_text(self, document: Document, file_content: bytes) -> ExtractionResult:
        """Extract text using sandboxed extractor."""

    def calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA-256 checksum."""
```

**Acceptance criteria:**
- [ ] Checksum calculation correct (matches `sha256sum` command)
- [ ] Deduplication works (same file registered twice = one document, two origins)
- [ ] Extraction reuses `SandboxedExtractor`
- [ ] Proper error handling for unreadable files

### Task 1.5: Create Document Embedding Service

**Files to create:**
- `backend/core/documents/embeddings.py`

**Functionality:**
```python
class DocumentEmbeddingService:
    async def generate_embeddings(self, document: Document) -> list[DocumentEmbedding]:
        """Generate embeddings for document (page-level for multi-page)."""

    def prepare_document_for_embedding(self, document: Document) -> str:
        """Build embedding text from content + metadata."""

    async def generate_page_embeddings(
        self,
        document: Document,
        pages: list[PageContent]
    ) -> list[DocumentEmbedding]:
        """Generate per-page embeddings."""
```

**Acceptance criteria:**
- [ ] Uses same embedding model as email (text-embedding-3-large)
- [ ] Handles documents with no text (embeds metadata only)
- [ ] Page-level embeddings work for multi-page docs
- [ ] Chunking works for very long pages

### Task 1.6: Basic API Endpoints

**Files to create:**
- `backend/api/routes/documents.py`

**Endpoints:**
```python
router = APIRouter(prefix="/api/documents", tags=["documents"])

@router.get("/{document_id}")
async def get_document(document_id: UUID) -> DocumentResponse

@router.get("/{document_id}/origins")
async def get_document_origins(document_id: UUID) -> list[DocumentOriginResponse]

@router.get("/by-checksum/{checksum}")
async def get_by_checksum(checksum: str) -> Optional[DocumentResponse]

@router.get("/stats")
async def get_stats() -> DocumentStats
```

**Acceptance criteria:**
- [ ] Endpoints follow existing API patterns (auth, error handling)
- [ ] Response schemas defined in `backend/api/schemas/`
- [ ] OpenAPI docs generated correctly

### Task 1.7: Integration Tests

**Files to create:**
- `backend/tests/integration/test_documents.py`

**Test cases:**
```python
async def test_register_new_document():
    """New document is created with origin."""

async def test_register_duplicate_document():
    """Same checksum = same document, new origin."""

async def test_extraction_workflow():
    """Document registered → extracted → embedded."""

async def test_api_get_document():
    """API returns document with origins."""
```

**Acceptance criteria:**
- [ ] All integration tests pass
- [ ] Tests use test database (not production)
- [ ] Cleanup after tests

---

## Phase 2: Folder Scanning

**Goal:** Index documents from local/network folders
**Effort:** 1 week
**Risk:** Medium (file system operations, multi-host)
**Depends on:** Phase 1 complete

### Task 2.1: Create Folder Scanner

**Files to create:**
- `backend/core/documents/folder_scanner.py`

**Functionality:**
```python
class FolderScanner:
    def __init__(self, config: FolderScanConfig):
        self.config = config

    async def scan(
        self,
        path: str,
        host: str,
        recursive: bool = True,
        extensions: Optional[list[str]] = None
    ) -> ScanResult:
        """Scan folder and register documents."""

    def discover_files(
        self,
        path: str,
        recursive: bool,
        extensions: list[str],
        exclude_patterns: list[str]
    ) -> Iterator[Path]:
        """Yield files matching criteria."""
```

**Acceptance criteria:**
- [ ] Recursive scanning works
- [ ] Extension filtering works
- [ ] Exclusion patterns work (*.tmp, .DS_Store, etc.)
- [ ] Hidden files skipped by default
- [ ] Max file size respected

### Task 2.2: Create CLI Tool

**Files to create:**
- `process_folders.py` (root level, like process_inbox.py)

**CLI interface:**
```bash
python process_folders.py /path/to/folder \
    --host hostname \
    --recursive \
    --extensions pdf,docx,xlsx \
    --exclude "*.tmp" \
    --dry-run \
    --limit 100
```

**Acceptance criteria:**
- [ ] CLI arguments parsed correctly
- [ ] Dry-run mode shows what would be indexed
- [ ] Progress output during scanning
- [ ] Summary at end (new, duplicates, errors)

### Task 2.3: Host Configuration

**Files to create:**
- `config/document_hosts.yaml.example`

**Files to modify:**
- `backend/core/documents/config.py`

**Configuration loading:**
```python
@dataclass
class HostConfig:
    type: str  # 'local', 'network_mount', 'ssh'
    mount_point: Optional[str] = None
    ssh_host: Optional[str] = None
    ssh_user: Optional[str] = None

def load_host_config(host: str) -> HostConfig:
    """Load host config from document_hosts.yaml."""
```

**Acceptance criteria:**
- [ ] Config file loaded from CONFIG_DIR
- [ ] Default config works (local access only)
- [ ] SSH config validated

### Task 2.4: Document Retrieval Service

**Files to create:**
- `backend/core/documents/retrieval.py`

**Functionality:**
```python
class DocumentRetrievalService:
    async def get_content(
        self,
        document: Document,
        origin_index: int = 0
    ) -> bytes:
        """Retrieve document binary from origin."""

    async def retrieve_from_filesystem(self, origin: DocumentOrigin) -> bytes
    async def retrieve_from_ssh(self, origin: DocumentOrigin) -> bytes
    async def retrieve_from_imap(self, origin: DocumentOrigin) -> bytes

    async def check_origin_accessible(self, origin: DocumentOrigin) -> bool
```

**Acceptance criteria:**
- [ ] Local file retrieval works
- [ ] Network mount retrieval works
- [ ] SSH retrieval works (uses scp)
- [ ] IMAP attachment retrieval works (reuses existing)
- [ ] Graceful handling of inaccessible origins

### Task 2.5: API Content Endpoint

**Files to modify:**
- `backend/api/routes/documents.py`

**New endpoint:**
```python
@router.get("/{document_id}/content")
async def get_document_content(
    document_id: UUID,
    origin_index: int = 0
) -> Response:
    """Retrieve original document binary."""
```

**Acceptance criteria:**
- [ ] Returns binary with correct Content-Type
- [ ] Content-Disposition header for download
- [ ] 404 if origin not accessible
- [ ] Falls back to next origin if primary fails

### Task 2.6: Incremental Scanning

**Files to modify:**
- `backend/core/documents/folder_scanner.py`

**Add caching for incremental scans:**
```python
class ScanCache:
    """Track file mtime+size to skip unchanged files."""

    def should_process(self, path: Path) -> bool:
        """Check if file changed since last scan."""

    def mark_processed(self, path: Path):
        """Record file as processed."""
```

**Acceptance criteria:**
- [ ] Second scan of same folder is fast (skips unchanged)
- [ ] Changed files are re-processed
- [ ] New files are processed
- [ ] Deleted files marked as deleted in origins

---

## Phase 3: OCR Processing (Optional)

**Goal:** Extract text from scanned documents and images
**Effort:** 1 week
**Risk:** Medium (external dependencies, API costs)
**Depends on:** Phase 2 complete
**Prerequisite:** `DOCUMENT_OCR_ENABLED=true`

### Task 3.1: OCR Configuration

**Files to create:**
- `backend/core/documents/ocr_config.py`

**Configuration:**
```python
@dataclass
class OCRConfig:
    enabled: bool = False
    quality_threshold: float = 0.7
    use_tesseract: bool = True
    use_claude_fallback: bool = False
    max_cost_per_doc: float = 0.50
    daily_budget: float = 50.00
    rate_limit_per_hour: int = 100
```

**Acceptance criteria:**
- [ ] Config loaded from environment variables
- [ ] Disabled by default
- [ ] Validation of settings

### Task 3.2: Quality Scorer

**Files to create:**
- `backend/core/documents/quality_scorer.py`

**Functionality:**
```python
class QualityScorer:
    def score(self, text: str) -> tuple[float, list[str]]:
        """
        Score text quality 0.0-1.0.
        Returns (score, list_of_issues).
        """

    # Heuristics:
    # - Garbled text patterns
    # - Character diversity
    # - Word validity
    # - Language detection
```

**Acceptance criteria:**
- [ ] Good text scores > 0.8
- [ ] Garbled OCR output scores < 0.5
- [ ] Empty text scores 0.0
- [ ] Issues list is informative

### Task 3.3: Tesseract Integration

**Files to create:**
- `backend/core/documents/ocr_tesseract.py`

**Functionality:**
```python
class TesseractOCR:
    def __init__(self, languages: str = "eng+deu"):
        self.languages = languages

    async def extract(self, image_path: str) -> str:
        """Extract text using Tesseract."""

    async def extract_from_pdf(self, pdf_path: str) -> list[PageContent]:
        """Extract text from PDF pages via Tesseract."""
```

**Acceptance criteria:**
- [ ] Works with PNG, JPG, TIFF
- [ ] Works with multi-page PDF (via pdf2image)
- [ ] Timeout handling
- [ ] Graceful failure if Tesseract not installed

### Task 3.4: Claude Vision Integration

**Files to create:**
- `backend/core/documents/ocr_claude.py`

**Functionality:**
```python
class ClaudeOCR:
    async def extract(self, image_path: str) -> str:
        """Full OCR via Claude Vision."""

    async def refine(self, image_path: str, tesseract_text: str) -> str:
        """Refine Tesseract output using Claude."""
```

**Acceptance criteria:**
- [ ] Uses existing Anthropic client
- [ ] Cost tracking per request
- [ ] Rate limiting respected
- [ ] Handles images and PDFs

### Task 3.5: Hybrid OCR Pipeline

**Files to create:**
- `backend/core/documents/ocr_pipeline.py`

**Functionality:**
```python
class HybridOCRPipeline:
    async def extract(self, file_path: str, mime_type: str) -> ExtractionResult:
        """
        1. Try pdftotext/Tesseract
        2. Score quality
        3. If low quality and Claude enabled, refine/retry
        """
```

**Acceptance criteria:**
- [ ] pdftotext tried first for PDFs
- [ ] Tesseract tried first for images
- [ ] Claude only used if quality below threshold
- [ ] Cost tracked and budget respected

### Task 3.6: OCR Worker

**Files to create:**
- `backend/core/documents/ocr_worker.py`

**Functionality:**
```python
class OCRWorker:
    async def run(self, batch_size: int = 10, continuous: bool = False):
        """Process documents needing OCR from queue."""

    async def process_document(self, task: DocumentProcessingQueue):
        """Run OCR pipeline on single document."""
```

**Acceptance criteria:**
- [ ] Pulls from processing queue
- [ ] Updates document with extraction results
- [ ] Queues for embedding after successful extraction
- [ ] Respects rate limits and budget
- [ ] Handles failures gracefully

---

## Phase 4: Document Search

**Goal:** Enable semantic search over documents
**Effort:** 3-5 days
**Risk:** Low (follows email search patterns)
**Depends on:** Phase 1 complete (Phase 2-3 optional)

### Task 4.1: Document Search Service

**Files to create:**
- `backend/core/documents/search.py`

**Functionality:**
```python
class DocumentSearchService:
    async def semantic_search(
        self,
        query: str,
        limit: int = 20,
        filters: Optional[DocumentFilters] = None
    ) -> list[DocumentSearchResult]:
        """Semantic search over document embeddings."""

    async def find_similar(
        self,
        document_id: UUID,
        limit: int = 10
    ) -> list[DocumentSearchResult]:
        """Find documents similar to given document."""
```

**Acceptance criteria:**
- [ ] Uses same embedding model as query
- [ ] Filters work (document_type, date range, quality)
- [ ] Results include similarity score
- [ ] Performance acceptable (<500ms)

### Task 4.2: Search API Endpoints

**Files to modify:**
- `backend/api/routes/documents.py`

**New endpoints:**
```python
@router.get("/search/semantic")
async def semantic_search(
    q: str,
    limit: int = 20,
    document_type: Optional[str] = None,
    min_quality: float = 0.5
) -> DocumentSearchResponse

@router.get("/{document_id}/similar")
async def find_similar(
    document_id: UUID,
    limit: int = 10
) -> list[DocumentSearchResult]
```

**Acceptance criteria:**
- [ ] Search returns relevant results
- [ ] Pagination works
- [ ] Filters applied correctly

### Task 4.3: Search Integration Tests

**Files to create:**
- `backend/tests/integration/test_document_search.py`

**Test cases:**
```python
async def test_semantic_search_finds_document():
    """Search query matches document content."""

async def test_search_filters():
    """Document type and quality filters work."""

async def test_similar_documents():
    """Similar documents have related content."""

async def test_search_includes_metadata_only_docs():
    """Documents with no text but metadata are findable."""
```

---

## Phase 5: Unified Search (Future)

**Goal:** Single search across emails and documents
**Effort:** 1 week
**Risk:** Medium (API changes, ranking complexity)
**Depends on:** Phase 4 complete and validated

### Task 5.1: Design Unified Search API

- Define response schema (union of email and document results)
- Define ranking strategy across corpora
- Define pagination approach

### Task 5.2: Implement Unified Search

- Query both email and document embeddings
- Merge and rank results
- Handle pagination across corpora

### Task 5.3: Update API

- New `/api/search` endpoint with `types` parameter
- Backwards compatible with existing email search

---

## Deployment Checklist

### Pre-Deployment

- [ ] All migrations tested on staging DB
- [ ] Backup production database
- [ ] Document hosts config created (`config/document_hosts.yaml`)
- [ ] SSH keys configured for remote hosts (if using SSH retrieval)

### Phase 0 Deployment

- [ ] Deploy code changes
- [ ] Monitor embedding generation for errors
- [ ] Verify search includes attachment content

### Phase 1 Deployment

- [ ] Run migration: `./scripts/db_migrate.sh`
- [ ] Verify tables created
- [ ] Test API endpoints

### Phase 2 Deployment

- [ ] Create `config/document_hosts.yaml`
- [ ] Test SSH access to remote hosts
- [ ] Run initial folder scan (dry-run first)
- [ ] Schedule periodic scans (cron)

### Phase 3 Deployment (if enabled)

- [ ] Install Tesseract: `apt-get install tesseract-ocr tesseract-ocr-deu`
- [ ] Install poppler: `apt-get install poppler-utils`
- [ ] Set `DOCUMENT_OCR_ENABLED=true`
- [ ] Configure budget limits
- [ ] Start OCR worker

---

## Testing Requirements

**Critical:** Every phase MUST include comprehensive unit tests that pass before proceeding to the next phase. Tests are NOT optional and should be written alongside the implementation.

### Test Organization

```
backend/tests/
├── unit/
│   ├── test_embeddings.py              # Phase 0 (DONE)
│   ├── documents/
│   │   ├── test_models.py              # Phase 1 - SQLAlchemy models
│   │   ├── test_repository.py          # Phase 1 - Repository CRUD
│   │   ├── test_processor.py           # Phase 1 - Document processor
│   │   ├── test_embedding_service.py   # Phase 1 - Document embeddings
│   │   ├── test_folder_scanner.py      # Phase 2 - Folder scanning
│   │   ├── test_retrieval.py           # Phase 2 - Document retrieval
│   │   ├── test_ocr_pipeline.py        # Phase 3 - OCR processing
│   │   ├── test_quality_scorer.py      # Phase 3 - Quality scoring
│   │   └── test_search.py              # Phase 4 - Document search
│   └── ...
└── integration/
    ├── test_documents.py               # Phase 1 - Full workflow
    └── test_document_search.py         # Phase 4 - Search integration
```

### Phase-Specific Test Requirements

#### Phase 0 Tests (✓ COMPLETED)
- `test_attachment_text_included_in_embedding`
- `test_multiple_attachments_included`
- `test_attachment_text_truncated`
- `test_max_attachments_limit`
- `test_no_attachments_still_works`
- `test_attachment_without_extracted_text`
- `test_attachment_with_empty_text`
- `test_metadata_still_included_with_attachments`

#### Phase 1 Tests (REQUIRED)

**test_models.py:**
- `test_document_model_creation`
- `test_document_checksum_unique_constraint`
- `test_document_origin_relationship`
- `test_document_embedding_relationship`
- `test_extraction_status_enum_values`
- `test_encrypted_text_field`

**test_repository.py:**
- `test_create_document`
- `test_get_document_by_id`
- `test_get_document_by_checksum`
- `test_add_origin`
- `test_add_duplicate_origin_ignored`
- `test_update_extraction`
- `test_queue_task`
- `test_get_pending_tasks`
- `test_mark_task_completed`
- `test_mark_task_failed_with_retry`

**test_processor.py:**
- `test_register_new_document`
- `test_register_duplicate_document`
- `test_calculate_checksum`
- `test_extract_text_from_pdf`
- `test_extract_text_from_docx`
- `test_handle_unsupported_mime_type`
- `test_handle_extraction_error`

**test_embedding_service.py:**
- `test_prepare_document_for_embedding`
- `test_prepare_document_with_no_text`
- `test_prepare_document_with_metadata_only`
- `test_generate_page_embeddings`
- `test_chunking_long_pages`
- `test_uses_correct_embedding_model`

#### Phase 2 Tests (REQUIRED)

**test_folder_scanner.py:**
- `test_discover_files_recursive`
- `test_discover_files_non_recursive`
- `test_filter_by_extension`
- `test_exclude_patterns`
- `test_skip_hidden_files`
- `test_max_file_size_limit`
- `test_scan_registers_new_documents`
- `test_scan_detects_duplicates`
- `test_incremental_scan_skips_unchanged`

**test_retrieval.py:**
- `test_retrieve_from_local_filesystem`
- `test_retrieve_from_network_mount`
- `test_retrieve_via_ssh`
- `test_retrieve_from_imap`
- `test_check_origin_accessible`
- `test_fallback_to_next_origin`

#### Phase 3 Tests (REQUIRED if OCR enabled)

**test_ocr_pipeline.py:**
- `test_tesseract_extracts_text`
- `test_pdftotext_for_text_pdfs`
- `test_claude_fallback_on_low_quality`
- `test_respects_quality_threshold`
- `test_respects_cost_budget`
- `test_rate_limiting`

**test_quality_scorer.py:**
- `test_good_text_scores_high`
- `test_garbled_text_scores_low`
- `test_empty_text_scores_zero`
- `test_detects_common_ocr_errors`

#### Phase 4 Tests (REQUIRED)

**test_search.py:**
- `test_semantic_search_finds_document`
- `test_search_with_document_type_filter`
- `test_search_with_quality_filter`
- `test_search_with_date_range_filter`
- `test_find_similar_documents`
- `test_search_includes_metadata_only_docs`
- `test_search_performance_under_500ms`

### Test Coverage Requirements

- **Unit tests:** ≥80% line coverage for new code
- **All tests must pass** before merging phase commits
- **Run tests after each phase:** `poetry run pytest backend/tests/unit/`

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 0 | 1-2 days | None |
| Phase 1 | 1-2 weeks | Phase 0 |
| Phase 2 | 1 week | Phase 1 |
| Phase 3 | 1 week | Phase 2 (optional) |
| Phase 4 | 3-5 days | Phase 1 |
| Phase 5 | 1 week | Phase 4 |

**Recommended order:** 0 → 1 → 4 → 2 → 3 → 5

This prioritizes search capability (Phases 0, 1, 4) before folder scanning and OCR, allowing validation of the core system earlier.
