# Revised Document Indexing Proposal - Final Review

**Status:** ✅ **APPROVED**
**Date:** 2026-01-31
**Reviewer:** AI Assistant

## Assessment

The revised proposal (`docs/proposals/DOCUMENT_INDEXING.md`) has successfully addressed all critical architectural and security concerns raised in the previous review. It now presents a realistic, secure, and incremental path to implementation that aligns with the existing `mail-done` codebase.

### Key Improvements Verified

| Concern | Resolution in Revised Proposal | Verdict |
| :--- | :--- | :--- |
| **Encryption Conflict** | **Resolved.** Explicitly adopts the `EncryptedText` pattern for document content. Accepts the tradeoff of "semantic-only" search for body content to maintain privacy guarantees. | ✅ |
| **Embedding Mismatch** | **Resolved.** Updates schema to `vector(3072)` and `text-embedding-3-large` to match existing email embeddings, enabling future unified search. | ✅ |
| **Dependency Bloat** | **Resolved.** Moves OCR/Tesseract/LibreOffice to an optional **Phase 3** opt-in. Phase 1 correctly reuses the existing `SandboxedExtractor`. | ✅ |
| **Storage Strategy** | **Resolved.** Explicitly defines a **Reference-Only** model. Documents are tracked by origin (IMAP/Filesystem) with no redundant binary storage, mirroring the email architecture. | ✅ |
| **Implementation Scope** | **Resolved.** Adopts the recommended **Phase 0** (add attachment text to email embeddings) as an immediate high-value, low-effort win before building the full document store. | ✅ |

### Implementation Readiness

The proposal is ready for implementation. The phased approach reduces risk significantly:

1.  **Phase 0 (High Confidence):** Modifying `backend/core/search/embeddings.py` to include attachment text is a trivial change with high immediate value.
2.  **Phase 1 (High Confidence):** The schema definitions for `documents` and `document_origins` are now consistent with SQLAlchemy models in `backend/core/database/models.py`.
3.  **Phase 2+:** The deferred folder scanning and OCR complexity allows the core system to stabilize first.

### Final Recommendations

*   **Migration Safety:** When implementing Phase 1, ensure the Alembic migration includes the necessary `pgvector` index creation commands (as seen in the existing `001_initial_schema.py`).
*   **Rate Limiting:** For Phase 3 (OCR) and Phase 2 (Folder Scan), ensure the background workers respect the existing `rate_limiting` modules to avoid saturating APIs or disk I/O.
*   **Testing:** Start with a unit test for **Phase 0** that asserts attachment text is actually present in the string passed to the embedding generator.

**Conclusion:** This revised plan is solid. Proceed with **Phase 0** immediately.

