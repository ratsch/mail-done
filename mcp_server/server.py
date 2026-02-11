"""
Email Search MCP Server

A Model Context Protocol server that exposes email search capabilities
to AI assistants like Claude/Cursor.

Architecture:
    Cursor → MCP Server → Backend API → Database

Usage:
    python -m mcp_server.server

Or add to Cursor's MCP configuration.

Security:
- Transport: stdio (local only, no network exposure)
- Authentication: Required MCP_API_KEY environment variable
- Backend: Calls FastAPI backend via HTTP (requires API_KEY)
- Audit: All tool calls are logged
"""
import asyncio
import logging
import json
import os
from typing import Any, Optional
from contextlib import asynccontextmanager
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Tool,
    TextContent,
)

# Setup logging - log to file for audit trail
log_dir = os.path.expanduser("~/.email-mcp-logs")
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"mcp_server_{datetime.now().strftime('%Y%m%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()  # Also log to stderr for debugging
    ]
)
logger = logging.getLogger(__name__)

# Security: REQUIRED API key for MCP access
MCP_API_KEY = os.getenv("MCP_API_KEY")
if not MCP_API_KEY:
    error_msg = (
        "MCP_API_KEY environment variable is required. "
        "Generate a secure key and set it in your .env file: "
        "MCP_API_KEY=$(openssl rand -base64 32)"
    )
    logger.critical(error_msg)
    raise ValueError(error_msg)
else:
    logger.info("MCP API key authentication enabled")

# Attachment feature flag
MCP_ENABLE_FILE_DOWNLOADS = os.getenv("MCP_ENABLE_FILE_DOWNLOADS", "false").lower() == "true"
MCP_FILE_CACHE_DIR = os.getenv("MCP_FILE_CACHE_DIR")

# Validate attachment configuration if enabled
if MCP_ENABLE_FILE_DOWNLOADS:
    if not MCP_FILE_CACHE_DIR:
        error_msg = (
            "MCP_ENABLE_FILE_DOWNLOADS=true but MCP_FILE_CACHE_DIR not configured. "
            "Set MCP_FILE_CACHE_DIR to a directory path (e.g., ~/Downloads/.mcp_attachments/work)"
        )
        logger.critical(error_msg)
        raise ValueError(error_msg)
    
    # Try to create cache directory to validate it's accessible
    from pathlib import Path
    cache_dir = Path(MCP_FILE_CACHE_DIR).expanduser()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"File download cache directory validated: {cache_dir}")
    except Exception as e:
        error_msg = f"Cannot create cache directory {cache_dir}: {e}"
        logger.critical(error_msg)
        raise ValueError(error_msg)
else:
    logger.info("File downloads disabled (set MCP_ENABLE_FILE_DOWNLOADS=true to enable)")

# Import API client (calls backend, not direct DB)
from mcp_server.api_client import EmailAPIClient


def audit_log(tool_name: str, arguments: dict, user_info: str = "local"):
    """Log tool usage for audit trail."""
    # Sanitize arguments - don't log full query text, just metadata
    safe_args = {
        k: v if k not in ('query', 'topic', 'sender') else f"<{len(str(v))} chars>"
        for k, v in arguments.items()
    }
    logger.info(f"AUDIT: tool={tool_name} user={user_info} args={safe_args}")


def create_server() -> Server:
    """Create and configure the MCP server."""
    server = Server("email-and-document-search")
    
    # Tool definitions
    TOOLS = [
        Tool(
            name="semantic_search",
            description="""[EMAIL] Search emails semantically by meaning/concept.

Searches email body and subject text using vector embeddings.
Does NOT search document files - use semantic_document_search for files.

**Date Range:**
- DEFAULT: Last 3 months (90 days) - fast, responsive searches
- CUSTOM: Specify date_from/date_to for longer ranges
- HARD LIMIT: Maximum 2 years back (730 days) - cannot be exceeded

**Examples:**
- "machine learning papers from collaborators"
- "grant deadlines and funding opportunities"
- "PhD applications about genomics"

Returns email summaries with similarity scores.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query describing what you're looking for"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score 0-1 (default: 0.6). Lower = more results but less relevant",
                        "default": 0.6
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by email category (e.g., 'application-phd', 'invitation-speaking')"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD). Default: 90 days ago. Max: 730 days ago (HARD LIMIT)"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD). Default: today"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["semantic", "keyword", "hybrid"],
                        "description": "Search mode: 'semantic' (meaning, recommended), 'keyword' (subject only), 'hybrid' (both). Note: keyword search on encrypted bodies is slow.",
                        "default": "semantic"
                    },
                    "sender": {
                        "type": "string",
                        "description": "Filter by sender email address or name (partial match supported, e.g., 'john.doe' or 'stanford.edu')"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_by_sender",
            description="""[EMAIL] Find emails from a specific sender.

Searches the email database by sender address or name.

**Date Range:**
- DEFAULT: Last 3 months (90 days) - fast, responsive
- CUSTOM: Specify date_from/date_to for longer ranges
- HARD LIMIT: Maximum 2 years back (730 days)

**Examples:**
- "john.doe@stanford.edu" - exact email match
- "Stanford" - partial match on domain or name
- "Yoshua Bengio" - search by name

Returns emails sorted by date (newest first).""",
            inputSchema={
                "type": "object",
                "properties": {
                    "sender": {
                        "type": "string",
                        "description": "Sender email address, name, or partial match"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 20)",
                        "default": 20
                    },
                    "category": {
                        "type": "string",
                        "description": "Filter by email category"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date filter (YYYY-MM-DD format)"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date filter (YYYY-MM-DD format)"
                    }
                },
                "required": ["sender"]
            }
        ),
        Tool(
            name="search_by_topic",
            description="""[EMAIL] Search emails by research topic or scientific concept.

Searches email content using semantic embeddings, optimized for academic/research topics.

**Examples:**
- "CRISPR gene editing"
- "reinforcement learning robotics"
- "single-cell RNA sequencing"

Can filter by email categories like PhD applications or speaking invitations.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "Research topic or scientific concept"
                    },
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by categories (e.g., ['application-phd', 'invitation-speaking'])"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 20)",
                        "default": 20
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score 0-1 (default: 0.6)",
                        "default": 0.6
                    }
                },
                "required": ["topic"]
            }
        ),
        Tool(
            name="find_similar_emails",
            description="""[EMAIL] Find emails similar to a reference email.

Given an email ID, find other emails with similar content/topic.
Uses email embeddings - for similar documents use find_similar_documents.

**Useful for:**
- "Find other applications like this one"
- "Show me similar collaboration requests"
- "Group related correspondence"

Requires an email ID (UUID) from previous email search results.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "UUID of the reference email"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar emails to find (default: 10)",
                        "default": 10
                    },
                    "same_category_only": {
                        "type": "boolean",
                        "description": "Only return emails in same category (default: false)",
                        "default": False
                    },
                    "exclude_same_sender": {
                        "type": "boolean",
                        "description": "Exclude emails from same sender (default: false)",
                        "default": False
                    }
                },
                "required": ["email_id"]
            }
        ),
        Tool(
            name="get_email_details",
            description="""[EMAIL] Get full details of a specific email.

Given an email ID, returns the complete email including:
- Full body text
- All recipients (to, cc)
- Attachment metadata (names, sizes) - use list_attachments to download
- AI analysis (category, urgency, action items)
- For applications: applicant info and scores

Use after email search to get the full content.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "UUID of the email to retrieve"
                    }
                },
                "required": ["email_id"]
            }
        ),
        Tool(
            name="list_categories",
            description="""[EMAIL] List all available email categories with counts.

Returns AI-assigned email categories like:
- application-phd (PhD applications)
- application-postdoc (Postdoc applications)
- invitation-speaking (Speaking invitations)
- collaboration-research (Research collaborations)

Use to understand what email categories are available for filtering email searches.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_top_senders",
            description="""[EMAIL] List top email senders by email count.

Shows the most frequent email senders with:
- Email address and name
- Total email count
- VIP status
- Typical email category

Useful for understanding who sends the most emails.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top senders to return (default: 20)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="semantic_document_search",
            description="""[DOCUMENT] Search indexed document files semantically.

Searches files (PDFs, Office docs, text files) that have been indexed from:
- Local folders (scanned directories)
- Email attachments (if attachment indexing is enabled)

Does NOT search email body text - use semantic_search for emails.

**Returns for each document:**
- Filename, file type, size
- Dates: document_date, first_seen_at, last_seen_at
- Locations: All origins with host name and full path
- Similarity score

**Examples:**
- "invoices from 2024"
- "machine learning papers"
- "contract renewal terms" """,
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score 0-1 (default: 0.5)",
                        "default": 0.5
                    },
                    "document_type": {
                        "type": "string",
                        "description": "Filter by document type (e.g., 'invoice', 'contract')"
                    },
                    "mime_type": {
                        "type": "string",
                        "description": "Filter by MIME type (e.g., 'application/pdf')"
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD)"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="semantic_search_unified",
            description="""[EMAIL+DOCUMENT] Search across BOTH emails AND documents in a single query.

Searches both email body text AND indexed document files, returning merged results ranked by similarity.

**Useful when:**
- Looking for information that might be in an email OR a file
- "Find everything about project X"
- "What do I have about machine learning?"

**Returns:**
- Mixed results with result_type ("email" or "document")
- For emails: subject, sender, date, summary
- For documents: filename, locations (origins), dates

Use 'types' param to filter: "all", "email", or "document".""",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query"
                    },
                    "types": {
                        "type": "string",
                        "enum": ["all", "email", "document"],
                        "description": "What to search: all (default), email, or document",
                        "default": "all"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 10)",
                        "default": 10
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score 0-1 (default: 0.5)",
                        "default": 0.5
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start date (YYYY-MM-DD)"
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End date (YYYY-MM-DD)"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="get_document_details",
            description="""[DOCUMENT] Get full details of a specific indexed document file.

Given a document ID, returns complete file information including:
- Filename, file type, size, page count
- Dates: document_date (file date), first_seen_at, last_seen_at
- All locations (origins) with host name and full path
- Extracted text preview
- AI category and tags

Use after document search to get full file information.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "UUID of the document"
                    }
                },
                "required": ["document_id"]
            }
        ),
        Tool(
            name="find_similar_documents",
            description="""[DOCUMENT] Find document files similar to a reference document.

Given a document ID, find other indexed files with similar content.
For similar emails, use find_similar_emails instead.

**Useful for:**
- "Find other invoices like this one"
- "Show me similar contracts"
- "Group related documents"

Requires a document ID (UUID) from previous document search results.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "UUID of the reference document"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of similar documents to find (default: 10)",
                        "default": 10
                    },
                    "same_type_only": {
                        "type": "boolean",
                        "description": "Only return documents of same type (default: false)",
                        "default": False
                    },
                    "similarity_threshold": {
                        "type": "number",
                        "description": "Minimum similarity score 0-1 (default: 0.5)",
                        "default": 0.5
                    }
                },
                "required": ["document_id"]
            }
        ),
        Tool(
            name="search_document_by_name",
            description="""[DOCUMENT] Search indexed document files by filename.

Find document files by partial or exact filename match.
Searches only indexed files, not email subjects.

**Examples:**
- "invoice" - finds all files with "invoice" in name
- "2024-01" - finds files with "2024-01" in name
- ".pdf" - finds all PDF files

Returns documents sorted by relevance and date.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Filename or partial filename to search for"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of results (default: 20)",
                        "default": 20
                    },
                    "mime_type": {
                        "type": "string",
                        "description": "Filter by MIME type (e.g., 'application/pdf')"
                    },
                    "host": {
                        "type": "string",
                        "description": "Filter by origin host (e.g., 'mbp-GR-2', 'nvme-pi')"
                    }
                },
                "required": ["name"]
            }
        ),
        Tool(
            name="get_document_index_stats",
            description="""[DOCUMENT] Get document indexing statistics and overview.

Returns aggregate statistics about indexed document files:
- Total documents indexed
- Documents by extraction status (completed, pending, failed)
- Documents by host (which machines have indexed files)
- Documents by origin type (folder vs email_attachment)
- OCR statistics (documents needing OCR)
- Recent indexing activity

**Use this to:**
- Monitor overall indexing progress
- Check how many documents need OCR
- See breakdown by host or origin type""",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "Filter stats to a specific host (e.g., 'mbp-GR-2')"
                    },
                    "path_prefix": {
                        "type": "string",
                        "description": "Filter stats to a path prefix (e.g., '/Users/raetsch/Documents')"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="list_indexed_folder",
            description="""[DOCUMENT] List contents of a folder on a remote host.

Browses the actual filesystem of an indexed folder and returns its contents.
**Security:** Only folders that were previously indexed can be browsed.

**Returns for each file:**
- Filename, size, modification date
- Whether the file is already indexed
- Document ID if indexed

**Use this to:**
- Browse files in a previously indexed folder
- See which files in a folder are indexed vs not yet indexed
- Discover new files added since last indexing""",
            inputSchema={
                "type": "object",
                "properties": {
                    "host": {
                        "type": "string",
                        "description": "Host where the folder is located (e.g., 'mbp-GR-2')"
                    },
                    "folder_path": {
                        "type": "string",
                        "description": "Full path to the folder (e.g., '/Users/raetsch/Documents/Papers')"
                    },
                    "include_subfolders": {
                        "type": "boolean",
                        "description": "Include files in subfolders (default: false)",
                        "default": False
                    }
                },
                "required": ["host", "folder_path"]
            }
        ),
        Tool(
            name="download_document",
            description="""[DOCUMENT] Download an indexed document file to local cache.

Retrieves the actual file from its origin (filesystem via SSH, or email attachment).
If primary origin is unavailable, falls back to other origins with same checksum.

**Returns:**
- cached_path: Local path to downloaded file
- original_filename: Original filename
- size_bytes: File size
- content_type: MIME type
- from_cache: Whether file was already cached
- origin_used: Which origin the file was retrieved from

**Cache:**
Files are cached using document ID + checksum prefix.
Re-downloading same document returns cached version.

REQUIRES: MCP_ENABLE_FILE_DOWNLOADS=true (uses same cache directory).""",
            inputSchema={
                "type": "object",
                "properties": {
                    "document_id": {
                        "type": "string",
                        "description": "UUID of the document to download"
                    },
                    "origin_index": {
                        "type": "integer",
                        "description": "Which origin to try first (0=primary, default)",
                        "default": 0
                    },
                    "fallback": {
                        "type": "boolean",
                        "description": "Try other origins if first fails (default: true)",
                        "default": True
                    }
                },
                "required": ["document_id"]
            }
        ),
        Tool(
            name="list_imap_folders",
            description="""[EMAIL/IMAP] List all email folders on the IMAP server.

Returns the actual folder structure from the mail server, including:
- INBOX
- Sent, Drafts, Trash, Archive
- Custom folders and subfolders

Use this to discover what email folders exist before listing emails.

**Account parameter:** Specify which email account to query (e.g., 'personal', 'work', 'eth').""",
            inputSchema={
                "type": "object",
                "properties": {
                    "account": {
                        "type": "string",
                        "description": "Email account to query (e.g., 'personal', 'work', 'eth')"
                    }
                },
                "required": ["account"]
            }
        ),
        Tool(
            name="get_imap_folder_status",
            description="""[EMAIL/IMAP] Get email folder status (message counts) efficiently.

Uses IMAP STATUS command - very fast, no message data is fetched.

**Returns:**
- total: Total number of emails in folder
- unseen: Number of unread emails
- recent: Number of recent emails (new since last check)
- uidnext: Next UID that will be assigned
- uidvalidity: UID validity value

**Use this to:**
- Check how many emails are in a folder
- Verify email counts after export/backup
- Monitor folder sizes

**Example folders:**
- "INBOX"
- "Sent"
- "Old Sent Messages/2016"
- "Archive/2024"
""",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Folder name/path (e.g., 'INBOX', 'Sent', 'Old Sent Messages/2016')"
                    },
                    "account": {
                        "type": "string",
                        "description": "Email account to query (e.g., 'personal', 'work', 'eth')"
                    }
                },
                "required": ["folder", "account"]
            }
        ),
        Tool(
            name="list_imap_folder_emails",
            description="""[EMAIL/IMAP] List emails currently in a specific IMAP folder.

Fetches directly from the IMAP server (not from local database).
Returns email summaries with:
- UID (unique identifier on server)
- Message-ID
- Subject
- From address and name
- Date
- Flags (read/unread, flagged)
- Whether it has attachments

**Parameters:**
- folder: Folder path (e.g., "INBOX", "Sent", "Archive/2024")
- limit: Max messages to return (default: 50, max: 500)
- since_date: Only messages since date (YYYY-MM-DD)

Emails are returned newest first.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "folder": {
                        "type": "string",
                        "description": "Folder name/path (e.g., 'INBOX', 'Sent', 'Archive/2024')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of messages to return (default: 50, max: 500)",
                        "default": 50
                    },
                    "since_date": {
                        "type": "string",
                        "description": "Only return messages since this date (YYYY-MM-DD format)"
                    },
                    "account": {
                        "type": "string",
                        "description": "Email account to query (e.g., 'personal', 'work', 'eth')"
                    }
                },
                "required": ["folder", "account"]
            }
        ),
        # ==========================================================================
        # APPLICATION REVIEW TOOLS
        # ==========================================================================
        Tool(
            name="list_applications",
            description="""[APPLICATION] List and search PhD/postdoc/intern applications.

Returns applications with filters matching the Application Review Portal functionality.

**Filters available:**
- category: 'application-phd', 'application-postdoc', 'application-intern'
- min_recommendation_score: Overall recommendation (1-10)
- min_excellence_score: Scientific excellence (1-10)
- min_research_fit_score: Research fit (1-10)
- search_name: Partial name match
- profile_tags: Filter by tags (e.g., 'ml-experience', 'genomics')
- highest_degree: Filter by degree (e.g., 'PhD', 'Masters')
- application_status: 'pending', 'reviewed', 'decided'
- has_decision: true/false
- received_after/received_before: Date range (YYYY-MM-DD)

**Results include:**
- Applicant name, institution, date
- All AI scores (scientific excellence, research fit, recommendation)
- Technical experience scores
- Review ratings and decision status
- Google Drive folder links""",
            inputSchema={
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["application-phd", "application-postdoc", "application-intern", "application-visiting"],
                        "description": "Filter by application type"
                    },
                    "min_recommendation_score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Minimum overall recommendation score (1-10)"
                    },
                    "min_excellence_score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Minimum scientific excellence score (1-10)"
                    },
                    "min_research_fit_score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "description": "Minimum research fit score (1-10)"
                    },
                    "search_name": {
                        "type": "string",
                        "description": "Search by applicant name (partial match)"
                    },
                    "received_after": {
                        "type": "string",
                        "description": "Only applications received after this date (YYYY-MM-DD)"
                    },
                    "received_before": {
                        "type": "string",
                        "description": "Only applications received before this date (YYYY-MM-DD)"
                    },
                    "application_status": {
                        "type": "string",
                        "enum": ["pending", "reviewed", "decided"],
                        "description": "Filter by application status"
                    },
                    "has_decision": {
                        "type": "boolean",
                        "description": "Filter by whether application has a decision"
                    },
                    "application_source": {
                        "type": "string",
                        "description": "Filter by source (e.g., 'ai_center', 'direct')"
                    },
                    "collection_id": {
                        "type": "string",
                        "description": "Filter by collection UUID"
                    },
                    "profile_tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by profile tags (AND logic - all must be present)"
                    },
                    "highest_degree": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Filter by degree level (OR logic - any match)"
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["date", "overall_recommendation_score", "scientific_excellence_score", "research_fit_score", "applicant_name"],
                        "description": "Sort field (default: date)"
                    },
                    "sort_order": {
                        "type": "string",
                        "enum": ["asc", "desc"],
                        "description": "Sort order (default: desc)"
                    },
                    "page": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Page number (default: 1)",
                        "default": 1
                    },
                    "page_size": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": "Results per page (default: 20, max: 100)",
                        "default": 20
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="get_application_details",
            description="""[APPLICATION] Get full details of a specific application.

Given an application email ID, returns complete information including:

**Applicant Info:**
- Name, email, institution, nationality
- Current situation, highest degree
- Online profiles (GitHub, LinkedIn, Google Scholar)

**AI Scores with Reasons:**
- Scientific excellence score and reasoning
- Research fit score and reasoning
- Overall recommendation score and reasoning

**Technical Experience Scores:**
- Coding experience (with evidence)
- Omics/genomics experience
- Medical data experience
- Sequence analysis experience
- Image analysis experience

**AI Evaluation:**
- Summary of application
- Key strengths and concerns
- Suggested next steps

**Google Drive Links:**
- Folder path with all application materials
- Email text link
- Attachment links (CV, publications, etc.)
- Consolidated and reference letter attachments

**Reviews:**
- Average rating from reviewers
- Individual reviews with comments
- Final decision (if made)

Use after list_applications to get full details of interesting candidates.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "email_id": {
                        "type": "string",
                        "description": "UUID of the application email"
                    }
                },
                "required": ["email_id"]
            }
        ),
        Tool(
            name="get_application_tags",
            description="""[APPLICATION] Get available profile tags for filtering applications.

Returns all profile tags that have been assigned to applications, with counts.

**Example tags:**
- ml-experience: Machine learning experience
- genomics: Genomics background
- clinical: Clinical research experience
- industry: Industry experience

Use these tags with list_applications profile_tags filter.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_application_collections",
            description="""[APPLICATION] Get available application collections.

Collections are named groups of applications for organizational purposes.

Returns list of collections with:
- Collection ID (UUID)
- Collection name
- Application count

Use collection_id with list_applications to filter by collection.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),

        # ── Time / Date utility tools ──────────────────────────────────
        Tool(
            name="get_current_datetime",
            description="""[UTILITY] Get the current date, time, day of week, and timezone.

Returns precise current time information including:
- Date and day of week
- Current time with seconds
- Timezone name and UTC offset
- ISO 8601 formatted datetime
- Unix timestamp

Use this whenever you need to know the current date/time or day of week.
Default timezone is Europe/Zurich.""",
            inputSchema={
                "type": "object",
                "properties": {
                    "timezone": {
                        "type": "string",
                        "description": "IANA timezone name (e.g., 'Europe/Zurich', 'America/New_York', 'Asia/Tokyo'). Default: Europe/Zurich",
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="convert_timezone",
            description="""[UTILITY] Convert a time from one timezone to another.

Converts a given time (and optional date) between two timezones.
Useful for scheduling across time zones.

Examples:
- "What time is 9am Zurich in New York?"
- "Convert 14:30 Tokyo time to London"
- "When is 8:30 EST in CET?" """,
            inputSchema={
                "type": "object",
                "properties": {
                    "time": {
                        "type": "string",
                        "description": "Time in HH:MM or HH:MM:SS format (e.g., '09:00', '14:30:00')",
                    },
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format (default: today)",
                    },
                    "from_timezone": {
                        "type": "string",
                        "description": "Source IANA timezone (e.g., 'Europe/Zurich')",
                    },
                    "to_timezone": {
                        "type": "string",
                        "description": "Target IANA timezone (e.g., 'America/New_York')",
                    },
                },
                "required": ["time", "from_timezone", "to_timezone"]
            }
        ),
        Tool(
            name="get_date_info",
            description="""[UTILITY] Get day of week, week number, and relative info for any date.

Given a date, returns:
- Day of week (Monday, Tuesday, ...)
- ISO week number and day of year
- Whether it's today, tomorrow, or yesterday
- How many days from today and a human-readable relative description

Examples:
- "What day of the week is February 14?"
- "Is March 1 a weekday?"
- "How many days until December 25?" """,
            inputSchema={
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Date in YYYY-MM-DD format",
                    },
                    "timezone": {
                        "type": "string",
                        "description": "Timezone for 'today' reference (default: Europe/Zurich)",
                    },
                },
                "required": ["date"]
            }
        ),
    ]

    # Add email attachment tools if enabled
    if MCP_ENABLE_FILE_DOWNLOADS:
        TOOLS.extend([
            Tool(
                name="list_attachments",
                description="""[EMAIL ATTACHMENT] List attachments for a specific email.

Returns metadata for each attachment: filename, content_type, size.
These are raw email attachments. If attachment indexing is enabled,
some attachments may also be searchable via semantic_document_search.

Use download_attachment to retrieve the actual file.""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "UUID of the email"
                        }
                    },
                    "required": ["email_id"]
                }
            ),
            Tool(
                name="download_attachment",
                description="""[EMAIL ATTACHMENT] Download an email attachment to local cache.

Downloads a specific attachment from an email by index.
For indexed document files, use download_document instead.

Files are cached - if already downloaded, returns cached path.
NOTE: Downloaded attachments persist in cache - delete manually if sensitive.

REQUIRES: MCP_ENABLE_FILE_DOWNLOADS=true""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "email_id": {
                            "type": "string",
                            "description": "UUID of the email"
                        },
                        "attachment_index": {
                            "type": "integer",
                            "description": "0-based index of attachment (from list_attachments)"
                        }
                    },
                    "required": ["email_id", "attachment_index"]
                }
            ),
            Tool(
                name="clear_attachment_cache",
                description="""[EMAIL ATTACHMENT] Clear the email attachment cache directory.

Clears downloaded email attachments from local cache.
Optionally clear only files older than specified days.

REQUIRES: MCP_ENABLE_FILE_DOWNLOADS=true""",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "older_than_days": {
                            "type": "integer",
                            "description": "Only delete files older than this many days (default: delete all)"
                        }
                    },
                    "required": []
                }
            )
        ])
    
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        """Return list of available tools."""
        return TOOLS
    
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        """Execute a tool and return results."""
        try:
            # Audit log the request
            audit_log(name, arguments)
            
            # Create API client (calls backend service)
            client = EmailAPIClient()
            
            # Route to appropriate tool
            if name == "semantic_search":
                result = await client.semantic_search(
                    query=arguments["query"],
                    top_k=arguments.get("top_k", 10),
                    similarity_threshold=arguments.get("similarity_threshold", 0.6),
                    category=arguments.get("category"),
                    date_from=arguments.get("date_from"),
                    date_to=arguments.get("date_to"),
                    mode=arguments.get("mode", "semantic"),
                    sender=arguments.get("sender")
                )
            
            elif name == "search_by_sender":
                result = await client.search_by_sender(
                    sender=arguments["sender"],
                    top_k=arguments.get("top_k", 20),
                    category=arguments.get("category"),
                    date_from=arguments.get("date_from"),
                    date_to=arguments.get("date_to")
                )
            
            elif name == "search_by_topic":
                result = await client.search_by_topic(
                    topic=arguments["topic"],
                    categories=arguments.get("categories"),
                    top_k=arguments.get("top_k", 20),
                    similarity_threshold=arguments.get("similarity_threshold", 0.6)
                )
            
            elif name == "find_similar_emails":
                result = await client.find_similar_emails(
                    email_id=arguments["email_id"],
                    top_k=arguments.get("top_k", 10),
                    same_category_only=arguments.get("same_category_only", False),
                    exclude_same_sender=arguments.get("exclude_same_sender", False)
                )
            
            elif name == "get_email_details":
                result = await client.get_email_details(
                    email_id=arguments["email_id"]
                )
            
            elif name == "list_categories":
                result = await client.list_categories()
            
            elif name == "list_top_senders":
                result = await client.list_top_senders(
                    top_k=arguments.get("top_k", 20)
                )
            
            elif name == "semantic_document_search":
                result = await client.search_documents(
                    query=arguments["query"],
                    top_k=arguments.get("top_k", 10),
                    similarity_threshold=arguments.get("similarity_threshold", 0.5),
                    document_type=arguments.get("document_type"),
                    mime_type=arguments.get("mime_type"),
                    date_from=arguments.get("date_from"),
                    date_to=arguments.get("date_to")
                )

            elif name == "semantic_search_unified":
                result = await client.search_unified(
                    query=arguments["query"],
                    types=arguments.get("types", "all"),
                    top_k=arguments.get("top_k", 10),
                    similarity_threshold=arguments.get("similarity_threshold", 0.5),
                    date_from=arguments.get("date_from"),
                    date_to=arguments.get("date_to")
                )

            elif name == "get_document_details":
                result = await client.get_document_details(
                    document_id=arguments["document_id"]
                )

            elif name == "find_similar_documents":
                result = await client.find_similar_documents(
                    document_id=arguments["document_id"],
                    top_k=arguments.get("top_k", 10),
                    same_type_only=arguments.get("same_type_only", False),
                    similarity_threshold=arguments.get("similarity_threshold", 0.5)
                )

            elif name == "search_document_by_name":
                result = await client.search_document_by_name(
                    name=arguments["name"],
                    top_k=arguments.get("top_k", 20),
                    mime_type=arguments.get("mime_type"),
                    host=arguments.get("host")
                )

            elif name == "get_document_index_stats":
                result = await client.get_document_index_stats(
                    host=arguments.get("host"),
                    path_prefix=arguments.get("path_prefix")
                )

            elif name == "list_indexed_folder":
                result = await client.list_indexed_folder(
                    host=arguments["host"],
                    folder_path=arguments["folder_path"],
                    include_subfolders=arguments.get("include_subfolders", False)
                )

            elif name == "download_document":
                if not MCP_ENABLE_FILE_DOWNLOADS:
                    result = {"error": "Document downloads not enabled. Set MCP_ENABLE_FILE_DOWNLOADS=true"}
                else:
                    # Audit log document download attempt
                    audit_log(
                        "download_document",
                        {
                            "document_id": arguments["document_id"],
                            "origin_index": arguments.get("origin_index", 0)
                        }
                    )
                    result = await client.download_document(
                        document_id=arguments["document_id"],
                        origin_index=arguments.get("origin_index", 0),
                        fallback=arguments.get("fallback", True)
                    )
                    # Log result
                    if result.get("success"):
                        logger.info(
                            f"AUDIT: Document downloaded - document_id={arguments['document_id']}, "
                            f"filename={result.get('original_filename', 'unknown')}, "
                            f"size={result.get('size_bytes', 0)}, "
                            f"from_cache={result.get('from_cache', False)}, "
                            f"origin={result.get('origin_used', 'unknown')}"
                        )
                    else:
                        logger.warning(
                            f"AUDIT: Document download failed - document_id={arguments['document_id']}, "
                            f"error={result.get('error', 'unknown')}"
                        )

            elif name == "list_imap_folders":
                result = await client.list_imap_folders(
                    account=arguments["account"]
                )

            elif name == "get_imap_folder_status":
                result = await client.get_folder_status(
                    folder=arguments["folder"],
                    account=arguments["account"]
                )

            elif name == "list_imap_folder_emails":
                result = await client.list_folder_emails(
                    folder=arguments["folder"],
                    account=arguments["account"],
                    limit=arguments.get("limit", 50),
                    since_date=arguments.get("since_date")
                )
            
            elif name == "list_attachments":
                if not MCP_ENABLE_FILE_DOWNLOADS:
                    result = {"error": "File downloads not enabled. Set MCP_ENABLE_FILE_DOWNLOADS=true"}
                else:
                    result = await client.list_attachments(
                        email_id=arguments["email_id"]
                    )
            
            elif name == "download_attachment":
                if not MCP_ENABLE_FILE_DOWNLOADS:
                    result = {"error": "File downloads not enabled. Set MCP_ENABLE_FILE_DOWNLOADS=true"}
                else:
                    # Audit log attachment download attempt
                    audit_log(
                        "download_attachment",
                        {
                            "email_id": arguments["email_id"],
                            "attachment_index": arguments["attachment_index"]
                        }
                    )
                    result = await client.download_attachment(
                        email_id=arguments["email_id"],
                        attachment_index=arguments["attachment_index"]
                    )
                    # Log result (success or failure)
                    if result.get("success"):
                        logger.info(
                            f"AUDIT: Attachment downloaded - email_id={arguments['email_id']}, "
                            f"index={arguments['attachment_index']}, "
                            f"filename={result.get('original_filename', 'unknown')}, "
                            f"size={result.get('size_bytes', 0)}, "
                            f"from_cache={result.get('from_cache', False)}"
                        )
                    else:
                        logger.warning(
                            f"AUDIT: Attachment download failed - email_id={arguments['email_id']}, "
                            f"index={arguments['attachment_index']}, "
                            f"error={result.get('error', 'unknown')}"
                        )
            
            elif name == "clear_attachment_cache":
                if not MCP_ENABLE_FILE_DOWNLOADS:
                    result = {"error": "File downloads not enabled. Set MCP_ENABLE_FILE_DOWNLOADS=true"}
                else:
                    result = await client.clear_attachment_cache(
                        older_than_days=arguments.get("older_than_days")
                    )

            # ==========================================================================
            # APPLICATION REVIEW HANDLERS
            # ==========================================================================
            elif name == "list_applications":
                result = await client.list_applications(
                    category=arguments.get("category"),
                    min_recommendation_score=arguments.get("min_recommendation_score"),
                    min_excellence_score=arguments.get("min_excellence_score"),
                    min_research_fit_score=arguments.get("min_research_fit_score"),
                    search_name=arguments.get("search_name"),
                    received_after=arguments.get("received_after"),
                    received_before=arguments.get("received_before"),
                    application_status=arguments.get("application_status"),
                    has_decision=arguments.get("has_decision"),
                    application_source=arguments.get("application_source"),
                    collection_id=arguments.get("collection_id"),
                    profile_tags=arguments.get("profile_tags"),
                    highest_degree=arguments.get("highest_degree"),
                    sort_by=arguments.get("sort_by"),
                    sort_order=arguments.get("sort_order"),
                    page=arguments.get("page", 1),
                    page_size=arguments.get("page_size", 20)
                )

            elif name == "get_application_details":
                result = await client.get_application_details(
                    email_id=arguments["email_id"]
                )

            elif name == "get_application_tags":
                result = await client.get_application_available_tags()

            elif name == "get_application_collections":
                result = await client.get_application_collections()

            # ── Time / Date utility handlers (no backend API needed) ───
            elif name == "get_current_datetime":
                from zoneinfo import ZoneInfo
                tz_name = arguments.get("timezone", "Europe/Zurich")
                try:
                    tz = ZoneInfo(tz_name)
                except KeyError:
                    result = {"error": f"Unknown timezone: {tz_name}"}
                else:
                    now = datetime.now(tz)
                    result = {
                        "date": now.strftime("%Y-%m-%d"),
                        "day_of_week": now.strftime("%A"),
                        "time": now.strftime("%H:%M:%S"),
                        "timezone": tz_name,
                        "utc_offset": now.strftime("%z")[:3] + ":" + now.strftime("%z")[3:],
                        "iso8601": now.isoformat(),
                        "unix_timestamp": int(now.timestamp()),
                    }

            elif name == "convert_timezone":
                from zoneinfo import ZoneInfo
                time_str = arguments["time"]
                from_tz_name = arguments["from_timezone"]
                to_tz_name = arguments["to_timezone"]
                date_str = arguments.get("date")
                try:
                    from_tz = ZoneInfo(from_tz_name)
                    to_tz = ZoneInfo(to_tz_name)
                except KeyError as e:
                    result = {"error": f"Unknown timezone: {e}"}
                else:
                    try:
                        # Parse time
                        parts = time_str.split(":")
                        hour, minute = int(parts[0]), int(parts[1])
                        second = int(parts[2]) if len(parts) > 2 else 0
                        # Parse date (default: today in source timezone)
                        if date_str:
                            d = datetime.strptime(date_str, "%Y-%m-%d").date()
                        else:
                            d = datetime.now(from_tz).date()
                        # Build aware datetime and convert
                        dt_from = datetime(d.year, d.month, d.day, hour, minute, second, tzinfo=from_tz)
                        dt_to = dt_from.astimezone(to_tz)
                        # Offset difference (use abs to avoid floor-division sign bug)
                        diff_seconds = dt_to.utcoffset().total_seconds() - dt_from.utcoffset().total_seconds()
                        sign = "+" if diff_seconds >= 0 else "-"
                        abs_diff = abs(diff_seconds)
                        diff_hours = int(abs_diff // 3600)
                        diff_mins = int(abs_diff % 3600 // 60)
                        result = {
                            "input": {
                                "date": dt_from.strftime("%Y-%m-%d"),
                                "time": dt_from.strftime("%H:%M:%S"),
                                "timezone": from_tz_name,
                                "day_of_week": dt_from.strftime("%A"),
                            },
                            "output": {
                                "date": dt_to.strftime("%Y-%m-%d"),
                                "time": dt_to.strftime("%H:%M:%S"),
                                "timezone": to_tz_name,
                                "day_of_week": dt_to.strftime("%A"),
                            },
                            "utc_offset_difference": f"{sign}{abs(diff_hours)}:{diff_mins:02d}",
                        }
                    except (ValueError, IndexError) as e:
                        result = {"error": f"Invalid time/date format: {e}"}

            elif name == "get_date_info":
                from zoneinfo import ZoneInfo
                date_str = arguments["date"]
                tz_name = arguments.get("timezone", "Europe/Zurich")
                try:
                    tz = ZoneInfo(tz_name)
                except KeyError:
                    result = {"error": f"Unknown timezone: {tz_name}"}
                else:
                    try:
                        target = datetime.strptime(date_str, "%Y-%m-%d").date()
                        today = datetime.now(tz).date()
                        delta = (target - today).days
                        # Relative description
                        if delta == 0:
                            relative = "today"
                        elif delta == 1:
                            relative = "tomorrow"
                        elif delta == -1:
                            relative = "yesterday"
                        elif delta > 0:
                            relative = f"in {delta} days"
                        else:
                            relative = f"{abs(delta)} days ago"
                        result = {
                            "date": date_str,
                            "day_of_week": target.strftime("%A"),
                            "week_number": target.isocalendar()[1],
                            "day_of_year": target.timetuple().tm_yday,
                            "is_today": delta == 0,
                            "is_tomorrow": delta == 1,
                            "is_yesterday": delta == -1,
                            "days_from_today": delta,
                            "relative": relative,
                        }
                    except ValueError as e:
                        result = {"error": f"Invalid date format: {e}"}

            else:
                result = {"error": f"Unknown tool: {name}"}
            
            # Format response
            return [TextContent(
                type="text",
                text=json.dumps(result, indent=2, default=str)
            )]
            
        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return [TextContent(
                type="text",
                text=json.dumps({"error": str(e)})
            )]
    
    return server


async def run_server():
    """Run the MCP server using stdio transport."""
    server = create_server()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def main():
    """Entry point for the MCP server."""
    asyncio.run(run_server())


if __name__ == "__main__":
    main()
