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
MCP_ENABLE_ATTACHMENTS = os.getenv("MCP_ENABLE_ATTACHMENTS", "false").lower() == "true"
MCP_ATTACHMENT_CACHE_DIR = os.getenv("MCP_ATTACHMENT_CACHE_DIR")

# Validate attachment configuration if enabled
if MCP_ENABLE_ATTACHMENTS:
    if not MCP_ATTACHMENT_CACHE_DIR:
        error_msg = (
            "MCP_ENABLE_ATTACHMENTS=true but MCP_ATTACHMENT_CACHE_DIR not configured. "
            "Set MCP_ATTACHMENT_CACHE_DIR to a directory path (e.g., ~/Downloads/.mcp_attachments/work)"
        )
        logger.critical(error_msg)
        raise ValueError(error_msg)
    
    # Try to create cache directory to validate it's accessible
    from pathlib import Path
    cache_dir = Path(MCP_ATTACHMENT_CACHE_DIR).expanduser()
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Attachment cache directory validated: {cache_dir}")
    except Exception as e:
        error_msg = f"Cannot create cache directory {cache_dir}: {e}"
        logger.critical(error_msg)
        raise ValueError(error_msg)
else:
    logger.info("Attachment downloads disabled (set MCP_ENABLE_ATTACHMENTS=true to enable)")

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
    server = Server("email-search")
    
    # Tool definitions
    TOOLS = [
        Tool(
            name="semantic_search",
            description="""Search emails semantically by meaning/concept.

**Date Range:**
- DEFAULT: Last 3 months (90 days) - fast, responsive searches
- CUSTOM: Specify date_from/date_to for longer ranges
- HARD LIMIT: Maximum 2 years back (730 days) - cannot be exceeded

Use this to find emails related to a topic or concept, even if they don't contain exact keywords.
Examples:
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
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="search_by_sender",
            description="""Find emails from a specific sender.

**Date Range:**
- DEFAULT: Last 3 months (90 days) - fast, responsive
- CUSTOM: Specify date_from/date_to for longer ranges
- HARD LIMIT: Maximum 2 years back (730 days)
            
Search by email address or sender name.
Examples:
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
            description="""Search emails by research topic or scientific concept.
            
Optimized for finding academic/research-related emails.
Examples:
- "CRISPR gene editing"
- "reinforcement learning robotics"
- "single-cell RNA sequencing"

Can filter by categories like PhD applications or speaking invitations.""",
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
            description="""Find emails similar to a reference email.
            
Given an email ID, find other emails with similar content/topic.
Useful for:
- "Find other applications like this one"
- "Show me similar collaboration requests"
- "Group related correspondence"

Requires an email ID (UUID) from previous search results.""",
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
            description="""Get full details of a specific email.
            
Given an email ID, returns the complete email including:
- Full body text
- All recipients (to, cc)
- Attachments info
- AI analysis (category, urgency, action items)
- For applications: applicant info and scores

Use after searching to get the full content.""",
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
            description="""List all available email categories with counts.
            
Returns categories like:
- application-phd (PhD applications)
- application-postdoc (Postdoc applications)
- invitation-speaking (Speaking invitations)
- collaboration-research (Research collaborations)
- etc.

Use to understand what categories are available for filtering.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="list_top_senders",
            description="""List top email senders by email count.
            
Shows the most frequent senders with:
- Email address and name
- Total email count
- VIP status
- Typical category

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
            name="list_imap_folders",
            description="""List all folders on the IMAP server.

Returns the actual folder structure from the mail server, including:
- INBOX
- Sent, Drafts, Trash, Archive
- Custom folders and subfolders

Use this to discover what folders exist before listing emails in a specific folder.""",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="get_folder_status",
            description="""Get folder status (total and unread message counts) efficiently.

Uses IMAP STATUS command - very fast, no message data is fetched.

**Returns:**
- total: Total number of messages in folder
- unseen: Number of unread messages  
- recent: Number of recent messages (new since last check)
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
                    }
                },
                "required": ["folder"]
            }
        ),
        Tool(
            name="list_folder_emails",
            description="""List emails currently in a specific IMAP folder.

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

Messages are returned newest first.""",
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
                    }
                },
                "required": ["folder"]
            }
        )
    ]
    
    # Add attachment tools if enabled
    if MCP_ENABLE_ATTACHMENTS:
        TOOLS.extend([
            Tool(
                name="list_attachments",
                description="List attachments for an email. Returns filename, content_type, size for each attachment.",
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
                description="Download an email attachment to local cache directory. Returns the cached file path and original filename. REQUIRES: MCP_ENABLE_ATTACHMENTS=true. Files are cached - if already downloaded, returns existing path without re-downloading. NOTE: Downloaded attachments persist in cache - delete manually if sensitive.",
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
                description="Clear the attachment cache directory. Optionally clear only files older than specified days. REQUIRES: MCP_ENABLE_ATTACHMENTS=true.",
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
                    mode=arguments.get("mode", "semantic")
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
            
            elif name == "list_imap_folders":
                result = await client.list_imap_folders()
            
            elif name == "get_folder_status":
                result = await client.get_folder_status(
                    folder=arguments["folder"]
                )
            
            elif name == "list_folder_emails":
                result = await client.list_folder_emails(
                    folder=arguments["folder"],
                    limit=arguments.get("limit", 50),
                    since_date=arguments.get("since_date")
                )
            
            elif name == "list_attachments":
                if not MCP_ENABLE_ATTACHMENTS:
                    result = {"error": "Attachment downloads not enabled. Set MCP_ENABLE_ATTACHMENTS=true"}
                else:
                    result = await client.list_attachments(
                        email_id=arguments["email_id"]
                    )
            
            elif name == "download_attachment":
                if not MCP_ENABLE_ATTACHMENTS:
                    result = {"error": "Attachment downloads not enabled. Set MCP_ENABLE_ATTACHMENTS=true"}
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
                if not MCP_ENABLE_ATTACHMENTS:
                    result = {"error": "Attachment downloads not enabled. Set MCP_ENABLE_ATTACHMENTS=true"}
                else:
                    result = await client.clear_attachment_cache(
                        older_than_days=arguments.get("older_than_days")
                    )
            
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
