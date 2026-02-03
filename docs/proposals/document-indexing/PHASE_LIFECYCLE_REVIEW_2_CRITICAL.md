## Second-pass Critical Review: Document Lifecycle Edge Cases (Moves/Copies/Edits/OCR) + Orphaned Data

**Status:** 🚨 **NOT OK to ship as-is** (critical inconsistencies found)  
**Date:** 2026-01-31

This is a deliberately harsher review than the earlier “looks good” pass. The goal here is to stress the lifecycle invariants under *real* filesystem behavior (moves, renames, in-place edits), plus long-term hygiene (orphaned docs/origins/embeddings) and make sure the code, schema, and tests all agree.

---

## Executive summary (what’s wrong)

- **DB schema mismatch:** the current code uses `documents.is_orphaned`, `documents.orphaned_at`, and `documents.extraction_structure`, but `alembic/versions/002_documents.py` does **not** create these columns. A clean migration-based deployment will crash.
- **Origin path semantics mismatch:** scanner stores `origin_path` as a *full file path*, while retrieval and tests treat it as a *directory* (and append `origin_filename`). This breaks Phase 2 retrieval + origin verification in real usage.
- **IMAP attachment retrieval is broken:** `DocumentRetrievalService` calls `AttachmentExtractor()` with no constructor args and then calls a non-existent `get_attachment_content()` method; the real extractor requires an `AccountManager` and exposes `get_attachment(...)`.

These are not stylistic issues—they’re correctness/operability blockers.

---

## 1) DB migration vs ORM/repository: hard mismatch (blocker)

### What the migration creates

`alembic/versions/002_documents.py` creates `documents` with fields like `checksum`, `extracted_text`, `extraction_status`, and plaintext metadata, but **does not** include:
- `extraction_structure`
- `is_orphaned`
- `orphaned_at`

There is also no partial index for orphaned docs.

### What the code expects

The repository actively writes and queries these fields:
- `backend/core/documents/repository.py` marks orphaned documents:
  - `document.is_orphaned = True`
  - `document.orphaned_at = datetime.utcnow()`
- Other parts of the documents stack store/use extraction structure (pages/sheets/sections) via `document.extraction_structure`.

### Impact

Any DB created by running migrations will be missing columns required by the current runtime code. You’ll see failures like “column does not exist” during:
- orphaning logic
- extraction structure persistence
- orphan cleanup queries

### Required fix

Create a new Alembic migration (don’t mutate `002_documents.py`) that:
- `ALTER TABLE documents ADD COLUMN extraction_structure JSONB NULL`
- `ALTER TABLE documents ADD COLUMN is_orphaned BOOLEAN NOT NULL DEFAULT false`
- `ALTER TABLE documents ADD COLUMN orphaned_at TIMESTAMPTZ NULL`
- optionally adds `idx_documents_orphaned` partial index (e.g. `WHERE is_orphaned = true AND is_deleted = false`)

---

## 2) `origin_path` semantics mismatch (blocker)

### What the model says

`backend/core/documents/models.py` documents `origin_path` as:
- “Full path to file (for retrieval)”

### What the scanner/processor stores

The document ingestion path stores:
- `origin_path = file_path` (full file path)
- `origin_filename = filename` (basename)

### What retrieval does (wrong under the above semantics)

`backend/core/documents/retrieval.py` builds:

- for retrieval:
  - `full_path = Path(origin.origin_path) / origin.origin_filename`
- for accessibility check:
  - the same pattern

If `origin.origin_path` already includes the filename, this becomes:
`/path/to/file.pdf` + `/file.pdf` → `/path/to/file.pdf/file.pdf` (non-existent).

### Why tests didn’t catch this

Unit tests in `backend/tests/unit/documents/test_retrieval.py` create origins like:
- `origin_path = str(temp_file.parent)` (directory)
- `origin_filename = temp_file.name` (filename)

So tests validate a *different contract* than the production scanner writes.

### Required fix (pick one contract, enforce it everywhere)

You must choose **one** of these and align models, writes, retrieval, and tests:

**Option A (recommended given current DB uniqueness constraints):**
- Treat `origin_path` as **full file path**.
- In retrieval, use `Path(origin.origin_path)` directly (don’t append `origin_filename`).
- Keep `origin_filename` only for display/search convenience.

**Option B:**
- Treat `origin_path` as **directory**, and always append `origin_filename`.
- Then you must change the uniqueness constraint (since multiple docs can share a directory):
  - include `origin_filename` in the unique index, or store a derived `origin_full_path`.

Right now you’re in a broken hybrid: producer writes Option A, consumer/tests assume Option B.

---

## 3) Folder “deleted file” lifecycle is likely broken (high severity)

The “move/rename” lifecycle relies on:
- “old path disappears” → mark old origin deleted
- “new path appears” → add origin and keep doc dedup’d by checksum

However, the deletion detection path currently passes `origin_path` as a **parent directory**, while ingestion stores `origin_path` as **full file path**.

Result: old origins may never be marked deleted → documents won’t become orphaned → orphan cleanup never triggers → stale origins remain forever and retrieval keeps failing for moved/deleted files.

This is a direct consequence of the origin_path contract mismatch above.

---

## 4) IMAP attachment retrieval is broken (blocker for email-attachment origins)

`backend/core/documents/retrieval.py` currently does:
- `from backend.core.email.attachment_extractor import AttachmentExtractor`
- `extractor = AttachmentExtractor()`  ← constructor requires `AccountManager`
- `await extractor.get_attachment_content(...)` ← method does not exist (extractor exposes `get_attachment(...)`)

Additionally, `_check_imap_accessible`:
- opens a DB session via `next(get_db())` inside an async method (lifetime/leak risk)
- calls `await repo.get_by_id(...)` on what appears to be a sync SQLAlchemy repository
- checks `email.attachments`, which likely isn’t a persisted list field in your email model (the email pipeline usually stores attachment metadata separately, e.g., `attachment_info`)

### Required fix

Implement IMAP retrieval by reusing the *existing* attachment mechanism correctly:
- load the Email row (account_id, folder, uid, message_id)
- call `AttachmentExtractor(account_manager).get_attachment(account_id, folder, uid, attachment_index, message_id=...)`
- run the blocking IMAP fetch in an executor (or refactor extractor to async)
- optionally update email folder/uid if extractor finds the email moved (it already supports an `on_location_update` callback)

Until this is fixed, `GET /api/documents/{id}/content` and `/verify-origins` are **misleadingly “implemented” but non-functional** for attachment origins.

---

## 5) Orphaned embeddings / lifecycle hygiene (conceptually good, but currently unsafe)

The repository has strong hygiene primitives:
- mark orphaned when origins drop to 0
- “unorphan” when a new origin arrives
- delete orphaned documents later (cascade deletes embeddings)

But because:
- the DB migration is missing `is_orphaned/orphaned_at`, and
- origins may never be marked deleted due to the path mismatch,

…the orphaning mechanism is at high risk of being “dead code” in a migration-driven deployment, and/or never triggering under normal filesystem churn.

**Net:** you can accumulate dead origins + documents + embeddings with no reliable cleanup path.

---

## 6) Document edits / OCR upgrades: embedding invalidation looks good (but relies on the above)

One thing that *does* look solid in Phase 4: you have explicit support for “regenerate embeddings when text changes” via a `regenerate_embedding` queue task, and `DocumentEmbeddingService.regenerate_pending_embeddings()` processes those tasks (deletes old embeddings then regenerates).

However, this only helps when:
- extraction updates actually persist, and
- tasks reliably run, and
- documents don’t get stuck behind broken origin lifecycles.

---

## Test gaps (why CI can be green while prod is broken)

1) **Retrieval tests validate the wrong origin contract**

`backend/tests/unit/documents/test_retrieval.py` constructs filesystem origins as:
- `origin_path = directory`
- `origin_filename = basename`

But ingestion stores:
- `origin_path = full path`
- `origin_filename = basename`

So retrieval can pass tests and still fail in real scans.

2) **No “move/rename” end-to-end test**

There isn’t a test that:
- scans a folder (ingestion writes origin)
- renames/moves the file
- runs “deleted detection”
- verifies old origin becomes deleted and doc/orphan state evolves correctly.

---

## Concrete recommendations / next actions

### A) Decide + enforce a single `origin_path` contract (highest priority)

Pick either:
- **full file path** (recommended), or
- **directory + filename**

Then:
- update ingestion to always write the chosen shape
- update retrieval to match
- update deletion detection to match
- update/extend tests to match

### B) Add the missing Alembic migration for lifecycle columns

Create a new migration adding:
- `extraction_structure` (JSONB)
- `is_orphaned` (bool, default false)
- `orphaned_at` (timestamptz)
- (optional) partial index for orphan queries

### C) Fix IMAP attachment retrieval in `DocumentRetrievalService`

Refactor `_retrieve_from_imap()` + `_check_imap_accessible()` to:
- use the correct `AttachmentExtractor(account_manager).get_attachment(...)`
- look up Email metadata needed to fetch from IMAP
- avoid creating DB sessions with `next(get_db())` inside async functions

### D) Add 2-3 “contract tests” that match production behavior

- **Filesystem origin contract test:** register a document via processor/scanner path, then retrieve it successfully.
- **Move/rename test:** scan → rename → detect deleted → verify origin deletion + orphan marking.
- **Attachment origin retrieval test (mocked):** verify correct extractor method is called with account_id/folder/uid.

---

## Bottom line

Phase 4 *search* logic may be fine, but the surrounding lifecycle plumbing has **three blockers** that undercut real-world functionality:
- schema mismatch,
- origin path contract mismatch,
- IMAP retrieval mismatch.

I recommend treating this as a “Phase 2 correctness hotfix” before calling Phase 4 complete from an operability perspective.

