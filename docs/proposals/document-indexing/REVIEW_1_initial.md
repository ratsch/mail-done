# Document indexing proposal — revised critical review (validated against current `mail-done` code)

**Scope**: Review `docs/proposals/DOCUMENT_INDEXING.md` against the actual `mail-done` codebase (as of 2026-01-31) to verify which concerns are valid, which parts are redundant, and what a safer/more incremental implementation path should be.

---

## What `mail-done` already does (relevant reality check)

- **Attachment text extraction already exists (and is security-aware)**  
  - `backend/core/email/processor.py` extracts text from common “document-like” attachments (PDF/DOCX/XLSX/PPTX/RTF/ICS/plain text).  
  - Extraction is **sandboxed by default** (subprocess + CPU/memory limits) via `backend/core/email/sandboxed_extractor.py` and can be disabled with `SANDBOX_ATTACHMENTS=false` (not recommended).

- **Attachment text is *not* persisted in the database today**  
  - The DB stores `emails.attachment_info` with a boolean `extracted` flag, but **does not store the extracted text** (`backend/core/database/repository.py`).
  - Consequence: attachment text can help classification at ingest time, but can’t be searched later.

- **Semantic search exists, but embeddings currently ignore attachments**  
  - Email semantic search uses `email_embeddings` and `VectorSearch` (`backend/core/search/vector_search.py`).  
  - Embedding text preparation (`backend/core/search/embeddings.py`) uses **subject + body + (optional) AI summary**, but **does not include attachment text**.

- **Keyword search is effectively “subject-only” because bodies are encrypted**  
  - `emails.body_*` columns are encrypted-at-rest via SQLAlchemy TypeDecorators (`backend/core/database/encryption.py`, `backend/core/database/models.py`).  
  - `HybridSearch` explicitly defaults keyword search to **`subject_only=True`** because body search over encrypted columns requires table scans (`backend/core/search/hybrid_search.py`).
  - There is already a **pg_trgm trigram index** on `emails.subject` (`alembic/versions/001_initial_schema.py`).

- **Google Drive integration exists, but for export, not indexing**  
  - `backend/core/google/drive_client.py` is oriented around uploading/exporting outputs to Drive/Sheets, not crawling Drive as a document source.

---

## Validated concerns (what’s actually wrong / risky in the proposal)

### 1) The proposal’s document full-text search conflicts with `mail-done`’s encryption model

The proposal defines:
- `documents.extracted_text TEXT`
- `documents.extracted_text_search TSVECTOR GENERATED ALWAYS AS (...) STORED` with a GIN index

**Conflict**:
- In the actual codebase, **encryption is mandatory** (`DB_ENCRYPTION_KEY` is required) and sensitive content is stored using encrypted column types.  
- Storing `documents.extracted_text` in plaintext would be a major design deviation relative to how email bodies and AI summaries are treated.

**Why this matters**:
- If `extracted_text` is encrypted, **Postgres tsvector search won’t work** (tsvector needs plaintext).  
- If `extracted_text` is plaintext, you gain keyword search but you’ve created a **new sensitive plaintext corpus** in the DB.

**Revised guidance**:
You need an explicit architectural decision before implementing doc keyword search:
- **Option A (privacy-aligned)**: keep `extracted_text` encrypted; provide **semantic-only** search over embeddings + keyword search over *non-sensitive metadata* (filename, mime_type, tags, derived doc_type).  
- **Option B (search-first)**: store `extracted_text` plaintext (and accept the risk); then enforce compensating controls (disk encryption, DB access controls, audit, minimal retention, tighter deployment isolation).  
- **Option C (separate index service)**: keep DB encrypted but build a dedicated local full-text index (still requires plaintext somewhere; it’s an ops/security tradeoff, not a free lunch).

### 2) Embedding dimension/model mismatch: proposal vs current schema

The proposal’s schema uses `document_embeddings.embedding vector(1536)` and references `ada-002`.

**Reality**:
- Current `email_embeddings` is defined as `Vector(3072)` (for `text-embedding-3-large`) in `backend/core/database/models.py`.
- `VectorSearch` defaults to `text-embedding-3-large` unless overridden.

**Risk**:
If you add document embeddings with 1536 dims while query embeddings (or email embeddings) are 3072, you’ll either:
- fail at query time (dimension mismatch), or
- end up maintaining **two embedding spaces** (separate query embedding per corpus), which complicates unified search.

**Revised guidance**:
Pick **one embedding model + dimension** for *all searchable corpora* (emails + documents) unless you’re intentionally building separate search indices and accepting the complexity.

### 3) The OCR pipeline in the proposal assumes external tools not present in the repo

The proposal calls for:
- `pdftotext`
- Tesseract OCR
- LibreOffice conversions
- “proven hybrid pipeline from `~/Scans/`”

**Reality**:
- Current extraction is Python-library based and sandboxed; there is **no OCR** and no dependency scaffolding for these external binaries in the repo.
- The “`~/Scans/`” pipeline is not part of this codebase (and is not reproducible for other users).

**Risk**:
Adding these tools increases:
- deployment complexity (OS packages, Docker image size),
- security surface area (LibreOffice on untrusted inputs),
- operational variability across hosts.

**Revised guidance**:
Start by reusing the existing sandboxed extractor for the file types it already supports. Add OCR as a clearly optional, explicitly configured, cost-tracked “Phase 2+” capability.

### 4) “Email attachment indexing” will require new persistence, not just reuse of existing extraction

The proposal suggests “extend `process_inbox.py`” to treat attachments as documents.

**Reality**:
- Attachments are not persisted as binaries; they can be downloaded from IMAP when needed (`backend/core/email/attachment_extractor.py` and `backend/api/routes/attachments.py`).
- Current DB only stores attachment metadata (`emails.attachment_info`) and whether extraction succeeded.

**Implication**:
If you want attachment-based document search (dedup, provenance, “find this PDF again”), you need to add:
- a **binary storage strategy** (copy by checksum into a controlled storage dir, or keep an IMAP-origin pointer and fetch on demand),
- a **document identity** (checksum) and origin linkage (email_id + attachment_index at minimum),
- and likely a background reprocessing path for older emails.

### 5) Unified search across emails + documents is a larger product/API change than the proposal suggests

Current API:
- `/api/search` returns email results (`EmailResponse`) only.

Proposal:
- a unified search that merges emails + documents.

**Reality**:
This touches:
- API response schemas (union types),
- UI expectations (if any),
- ranking strategy across corpora,
- pagination/count correctness (already non-trivial in current email search).

**Revised guidance**:
Treat unified search as a later phase; start with `/api/documents/*` endpoints and optionally add a “merge layer” once document search quality is proven.

---

## What’s good / reusable in the proposal (validated)

- **Checksum-based dedup + origin tracking** is a strong fit.  
  Nothing like that exists today; it would add real value for “found in multiple places” workflows.

- **Extraction versioning + reprocessing** is aligned with how the system already thinks about iterative pipelines (the codebase already tracks model usage/cost in other contexts).

- **Document processing queue** is a solid idea.  
  The current system already has background-ish patterns (processing jobs, embedding generation), so a queue fits.

---

## Revised implementation recommendation (incremental and codebase-aligned)

### Phase 0 (fast value, minimal surface area): make attachment content searchable *without* “documents”

- **Include attachment text in email embeddings** (bounded and sanitized):
  - Append a limited amount of `ProcessedEmail.attachment_texts` to the embedding input text (e.g., first N attachments, first M chars each).
  - This immediately enables semantic search over “what was in the CV / PDF”, with no new schema.
- Optional: persist attachment extracted text encrypted (either:
  - a new `email_attachment_texts` table keyed by `email_id` + `attachment_index`, or
  - an encrypted JSON column on `email_metadata`).

Why this is the right first step:
- It reuses the existing sandboxed extraction work you already pay for at ingest time.
- It addresses a concrete current gap: **semantic search ignores attachment content today**.

### Phase 1 (documents as first-class objects): add `documents` + origins + embeddings, but align with encryption + embeddings reality

- Add SQLAlchemy models + Alembic migration for:
  - `documents`
  - `document_origins`
  - `document_embeddings`
- **Use the same embedding model/dimension as email search** (unless you intentionally split indices).
- **Decide on content-at-rest policy** for `documents.extracted_text` up front:
  - If encrypted: semantic search + metadata keyword search only.
  - If plaintext: full-text search possible, but treat as sensitive.
- Reuse existing extractors where possible:
  - For email attachments: reuse `AttachmentExtractor` to fetch bytes + existing sandboxed extractors to extract text.
  - For local files: use the same sandboxed approach for parsing where feasible.

### Phase 2 (folder scanning): implement `process_folders.py` and host scanning carefully

The proposal mentions `process_folders.py` but it does not exist today.

Implementation notes:
- Make it config-driven (paths, recursion, include/exclude globs).
- Record `origin_host` and `origin_path` (as proposed), but be clear about trust boundaries:
  - do you copy binaries into a central store, or just index references?
- Add a safe incremental mode (mtime + size + checksum caching).

### Phase 3 (OCR + “legacy office”): optional, explicitly configured, sandboxed, cost-tracked

- OCR (Tesseract / LLM vision) and LibreOffice conversion should be:
  - opt-in,
  - rate limited,
  - cost tracked,
  - and isolated (ideally containerized or via hardened subprocess sandboxing).

### Phase 4 (unified search): merge only after documents search quality is validated

- Start with document-only endpoints (`/api/documents/search`, `/api/documents/{id}`, etc.).
- Add unified search later, once you have:
  - consistent embedding space,
  - ranking policy across corpora,
  - and pagination semantics.

---

## Concrete updates suggested to `docs/proposals/DOCUMENT_INDEXING.md`

- **Replace `vector(1536)` / `ada-002` assumptions** with “match the system’s configured embedding model/dims”.
- **Remove or generalize references to `~/Scans/`** and any non-repo pipelines.
- **Clarify encryption vs keyword search tradeoff** (it’s currently underspecified).
- **Add an explicit “Phase 0: attachment text → email embeddings”** as the smallest high-value step.
- **Add a “dependencies & deployment” section** for OCR/LibreOffice/pdftotext (or defer them).

---

## Open questions you should decide before coding

- **Content confidentiality**: Are you willing to store extracted document text plaintext in the DB for keyword search, or must it remain encrypted?  
- **Storage strategy**: For attachments, do you store binaries by checksum, or rely on IMAP as the source-of-truth and fetch on demand?  
- **Unification**: Do you require a single merged search endpoint early, or is document-only search acceptable initially?


