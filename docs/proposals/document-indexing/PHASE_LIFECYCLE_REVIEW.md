# Critical Review: Document Lifecycle & Edge Cases

**Status:** ✅ **PASSED (Architecture & Logic are Sound)**
**Date:** 2026-01-31
**Reviewer:** AI Assistant

## Scope
Review of `backend/core/documents/` specifically evaluating "normal document life cycles" (creation, modification, movement, deletion, and updates) to identify logical gaps, race conditions, or orphaned data risks.

## Lifecycle Analysis

### 1. File Copy (Duplication)
*   **Scenario:** User copies `contract.pdf` from Folder A to Folder B.
*   **Logic:** `FolderScanner` finds new file at Folder B. `DocumentProcessor.register_document` calls `get_or_create(checksum)`.
*   **Result:** Existing `Document` returned (deduplication). New `DocumentOrigin` created for Folder B.
*   **Verdict:** ✅ **Correct.** Efficient reference-only storage.

### 2. File Move (Renaming)
*   **Scenario:** User moves `draft.docx` to `final.docx`.
*   **Logic:**
    1.  Scanner detects `final.docx`. `register_document` links it to the existing `Document` (same checksum).
    2.  Scanner's `scan_for_deleted` detects `draft.docx` is missing from disk vs cache. calls `mark_origin_deleted`.
*   **Result:** `Document` remains. One active origin (`final.docx`), one soft-deleted origin (`draft.docx`).
*   **Verdict:** ✅ **Correct.** History is preserved, document identity is stable.

### 3. Content Modification (The "Hard" Case)
*   **Scenario:** User edits `notes.txt`. Path is same, content changes.
*   **Logic:** `FolderScanner` sees mtime change. Calls `processor.handle_file_change`.
    1.  Calculates `new_checksum`.
    2.  Finds existing origin for `notes.txt`.
    3.  Calls `repo.update_origin_document` to repoint the origin from `OldDoc` to `NewDoc`.
    4.  Checks `OldDoc` for orphans (`check_and_mark_orphaned`).
*   **Result:** Origin now points to `NewDoc`. `OldDoc` is marked `is_orphaned=True` (if no other files have that content).
*   **Verdict:** ✅ **Excellent.** This explicitly handles the split identity problem. The orphan check prevents "ghost" documents from accumulating indefinitely.

### 4. OCR / Metadata Updates
*   **Scenario:** OCR worker finishes processing a scanned PDF.
*   **Logic:** Worker calls `processor.provide_ocr_text`.
    1.  Compares new quality vs existing quality.
    2.  If better, updates `extracted_text`.
    3.  **Crucial Step:** Automatically queues `regenerate_embedding` task.
*   **Result:** Search index is updated to reflect new text content.
*   **Verdict:** ✅ **Correct.** Ensures semantic search stays in sync with metadata improvements.

## Code Safety Checks

| Check | Status | Evidence |
| :--- | :--- | :--- |
| **Race Conditions** | ✅ Handled | `get_or_create` uses `try/except IntegrityError` for concurrent inserts. Queue claiming uses atomic updates. |
| **Orphans** | ✅ Handled | `handle_file_change` checks for orphans immediately. `delete_orphaned_documents` available for cleanup. |
| **Constraint Validity** | ✅ Valid | Unique constraint on `DocumentOrigin` allows moving (updating `document_id`) but prevents duplicates. |
| **Deletion Safety** | ✅ Safe | Deletions are soft-deletes (`is_deleted=True`). Hard deletion only via explicit cleanup job. |

## Minor Recommendations (Scalability)

1.  **Blocking IO in Scanner:** `calculate_checksum` performs synchronous file IO in the main loop. For multi-GB files, this pauses the asyncio loop.
    *   *Recommendation:* In the future, offload checksum calculation to a thread pool (`loop.run_in_executor`) to keep the heartbeat alive.
2.  **Scan Cache Durability:** If the scan process crashes hard (SIGKILL), `ScanCache` is not saved.
    *   *Impact:* Next scan will re-hash everything. Safe, but slow.
    *   *Recommendation:* Consider periodic cache saves during long scans (e.g., every 1000 files).

## Final Verdict
The implementation handles complex lifecycle events (modification-in-place, moves, orphans) with surprising robustness. The specific logic in `handle_file_change` demonstrates foresight regarding the "identity vs location" duality of file-based documents.

**Approved for production use.**

