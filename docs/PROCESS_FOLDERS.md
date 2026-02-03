# Document Indexing Guide

> **Prerequisites:** [Deployment Guide](DEPLOYMENT.md) | **Related:** [MCP Server](MCP.md), [API Reference](API.md), [Database](DATABASE.md)

`process_folders.py` scans folders on local or remote hosts to discover and index documents for semantic search. This is the document-equivalent of `process_inbox.py` for emails.

## Quick Start

### Safe Preview (Dry Run)

```bash
# Preview what would be indexed in ~/Documents
python3.11 process_folders.py ~/Documents --dry-run

# Preview with specific extensions
python3.11 process_folders.py /data/invoices --extensions pdf,xlsx --dry-run
```

### Index Documents

```bash
# Index local folder (default: pdf, docx, xlsx, pptx, doc, ppt, xls, csv, txt, md)
python3.11 process_folders.py ~/Documents

# Index with all file types
python3.11 process_folders.py /archive --all-extensions

# Index remote folder (via SSH)
python3.11 process_folders.py /home/user/papers --host remote-server
```

## Indexing Pipeline

Each file goes through:

```
1. File Discovery
   ├── Walk directory tree (recursive by default)
   ├── Filter by extension
   └── Skip hidden files (unless --include-hidden)
        ↓
2. Deduplication
   ├── Calculate SHA-256 checksum
   └── Check if document already exists
        ↓
3. Text Extraction
   ├── PDF: PyMuPDF, pdfplumber
   ├── Office: python-docx, openpyxl
   ├── Text: Direct read (txt, md, csv)
   └── OCR: Optional for scanned docs
        ↓
4. Origin Tracking
   ├── Record host + path
   └── Link to existing document if duplicate
        ↓
5. Embedding Generation (optional)
   └── Vector for semantic search
```

## Command Line Options

```
Usage: process_folders.py PATH [options]

Positional:
  PATH                  Path to folder to scan

Host Configuration:
  --host NAME           Host name from config/document_hosts.yaml (default: localhost)

Scan Options:
  -r, --recursive       Scan subdirectories (default: true)
  --no-recursive        Don't scan subdirectories
  -e, --extensions EXT  Comma-separated extensions (default: pdf,docx,xlsx,...)
  --all-extensions      Include ALL file types (no filter)
  -x, --exclude PAT     Comma-separated patterns to exclude (e.g., *.tmp,*.bak)
  --max-size MB         Maximum file size in MB (default: 100)
  --include-hidden      Include hidden files and directories

Processing Options:
  -l, --limit N         Maximum files to process
  -n, --dry-run         Preview without making changes
  --no-extract          Skip text extraction (register only)
  --skip-embeddings     Skip embedding generation
  --per-page-embeddings Generate embeddings per page (vs. single per document)
  --reindex             Re-extract and re-embed existing documents
  --commit-interval N   Commit to database every N files (default: 50)
  --detect-deleted      Find and mark deleted files

Cache Options:
  --cache-dir DIR       Directory for scan cache (enables incremental scanning)
  --no-cache            Disable scan cache

Output:
  -v, --verbose         Show detailed progress
  -q, --quiet           Suppress non-error output
```

## Common Workflows

### Index a Project Archive

```bash
# Scan papers directory, limit to PDFs
python3.11 process_folders.py ~/Papers --extensions pdf --verbose

# Output:
# Scanning /Users/me/Papers on mbp-GR-2
# Recursive: True, Extensions: ['pdf']
# Embedding generation: ENABLED (single embedding per document)
# Progress: 50 processed, 48 new, 2 duplicates
# ...
# ============================================================
# Scan completed in 125.3 seconds
# Files processed: 150
# New documents: 145
# Duplicates (existing): 5
# Errors: 0
# ============================================================
```

### Re-index After Adding OCR

```bash
# Re-extract text and regenerate embeddings for scanned documents
python3.11 process_folders.py /archive/scanned --reindex --extensions pdf
```

### Index Remote NAS

First, configure the host in `config/document_hosts.yaml`:

```yaml
hosts:
  nas:
    name: "nas"
    type: "ssh"
    ssh_host: "nas.local"
    ssh_user: "admin"
    mount_point: "/volume1"
```

Then scan:

```bash
python3.11 process_folders.py /volume1/Documents --host nas
```

### Detect Deleted Files

After files are moved or deleted:

```bash
# Mark deleted files in database
python3.11 process_folders.py /archive --detect-deleted
```

### Incremental Scanning

Use cache to skip unchanged files:

```bash
# First scan (full)
python3.11 process_folders.py /data --cache-dir ~/.cache/mail-done

# Subsequent scans (only new/changed files)
python3.11 process_folders.py /data --cache-dir ~/.cache/mail-done
```

## Supported File Types

### Default Extensions

When no `--extensions` specified, these are indexed:

| Type | Extensions |
|------|------------|
| PDF | pdf |
| Microsoft Office | docx, xlsx, pptx, doc, ppt, xls |
| Text | txt, md, csv |

### Text Extraction by Type

| Format | Extraction Method | Notes |
|--------|-------------------|-------|
| **PDF** | PyMuPDF + pdfplumber | Falls back to OCR if no text layer |
| **DOCX** | python-docx | Extracts all text including headers/footers |
| **XLSX** | openpyxl | Extracts all sheets as text |
| **PPTX** | python-pptx | Extracts slide text and notes |
| **TXT/MD** | Direct read | UTF-8 with fallback encodings |
| **CSV** | pandas | Tab-separated output |
| **Images** | OCR (optional) | Requires Tesseract |

### Enable All File Types

```bash
# Index everything (code, images, etc.)
python3.11 process_folders.py /project --all-extensions
```

## Document Deduplication

Documents are deduplicated by SHA-256 checksum:

```
Same file in multiple locations = Single document + Multiple origins
```

Example:
```
/archive/2023/report.pdf  →  Document #1
/backup/report.pdf        →  Document #1 (same checksum, new origin)
/projects/report_v2.pdf   →  Document #2 (different content)
```

Benefits:
- Single embedding per unique document
- All locations tracked via `document_origins` table
- Search returns document once, shows all locations

## Integration with Email Attachments

Documents can have multiple origin types:

| Origin Type | Source | Created By |
|-------------|--------|------------|
| `folder` | File system scan | `process_folders.py` |
| `email_attachment` | Email attachment | `process_inbox.py --backfill-attachments` |

When the same file exists as both an email attachment and a folder file:
- Single `documents` record (same checksum)
- Two `document_origins` records
- Unified search returns it regardless of source

## Host Configuration

Create `config/document_hosts.yaml` for non-local hosts:

```yaml
hosts:
  # Local machine (uses actual hostname)
  localhost:
    type: local
    mount_point: /

  # Remote server via SSH
  remote-server:
    name: "remote-server"
    type: ssh
    ssh_host: "server.example.com"
    ssh_user: "user"
    ssh_key: "~/.ssh/id_rsa"  # Optional
    mount_point: /home/user

  # NAS with custom port
  nas:
    name: "nas"
    type: ssh
    ssh_host: "nas.local"
    ssh_port: 2222
    ssh_user: "admin"
    mount_point: /volume1
```

## Performance Tuning

### Batch Commits

Control database commit frequency:

```bash
# Commit every 100 files (faster for large scans)
python3.11 process_folders.py /archive --commit-interval 100

# Commit only at end (fastest, but risky if interrupted)
python3.11 process_folders.py /archive --commit-interval 0
```

### Skip Embeddings

For fast initial indexing, skip embeddings:

```bash
# Index without embeddings (fast)
python3.11 process_folders.py /archive --skip-embeddings

# Generate embeddings later
python3.11 -m backend.core.documents.embeddings --generate-missing
```

### Per-Page vs Single Embeddings

```bash
# Single embedding per document (default, good for search)
python3.11 process_folders.py /docs

# Per-page embeddings (better for long documents)
python3.11 process_folders.py /books --per-page-embeddings
```

## Searching Indexed Documents

After indexing, documents are searchable via:

### MCP Tools

```
semantic_document_search - Search by meaning
search_document_by_name - Search by filename
find_similar_documents - Find related documents
```

### API Endpoints

```bash
# Semantic search
curl -H "X-API-Key: xxx" \
  "http://localhost:8000/api/documents/search/semantic?query=invoice+2024"

# Search by filename
curl -H "X-API-Key: xxx" \
  "http://localhost:8000/api/documents/search/by-name?name=report"

# Get document details
curl -H "X-API-Key: xxx" \
  "http://localhost:8000/api/documents/{document_id}"
```

### Web UI

The web UI supports multi-source search with checkboxes for:
- Email text
- Emailed documents (attachments)
- Document files (folders)

## Troubleshooting

### "Unknown host" Error

```
Error: Unknown host: nas
Configure hosts in config/document_hosts.yaml
```

**Solution:** Add host configuration to `config/document_hosts.yaml`.

### SSH Connection Failed

```
Failed to scan folder: SSH connection failed
```

**Solutions:**
1. Verify SSH access: `ssh user@host`
2. Check SSH key permissions: `chmod 600 ~/.ssh/id_rsa`
3. Add host to known_hosts: `ssh-keyscan host >> ~/.ssh/known_hosts`

### Text Extraction Failed

```
Extraction failed for document.pdf: No text found
```

**Solutions:**
1. Check if PDF has text layer (not scanned image)
2. Enable OCR for scanned documents
3. Check file isn't corrupted: `pdfinfo document.pdf`

### Out of Memory

For very large documents or many files:

```bash
# Process in smaller batches
python3.11 process_folders.py /archive --limit 100

# Skip embedding generation (memory-intensive)
python3.11 process_folders.py /archive --skip-embeddings
```

## Related Documentation

- [Email Processing](PROCESS_INBOX.md) - Process emails with `process_inbox.py`
- [Attachment Indexing](PROCESS_INBOX.md#attachment-indexing) - Index email attachments
- [API Reference](API.md#document-endpoints) - Document API endpoints
- [Database Schema](DATABASE.md#document-tables) - Document table structure
- [MCP Server](MCP.md) - Search documents via AI assistants
