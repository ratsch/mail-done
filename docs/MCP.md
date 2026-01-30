# MCP Server Documentation

> **Prerequisites:** [Deployment Guide](DEPLOYMENT.md) (backend must be running) | **Related:** [API Reference](API.md)

The mail-done MCP (Model Context Protocol) server enables AI assistants like Claude and Cursor to search and access your email database.

## Overview

The MCP server provides tools for:
- **Semantic Search**: Natural language email search
- **Sender Search**: Find emails from specific senders
- **Topic Search**: Search by research topics
- **Email Details**: Get full email content
- **Folder Listing**: Browse IMAP folders
- **Attachment Access**: List and download attachments (optional)

## Architecture

```
AI Assistant → MCP Server → Backend API → Database
              (thin HTTP     (auth, rate
               client)        limiting)
```

The MCP server is a lightweight HTTP client that:
- Connects to the FastAPI backend
- Translates MCP tool calls to API requests
- Never accesses the database directly

## Prerequisites

1. **Backend API running**: The FastAPI backend must be accessible
2. **API key**: Generate a secure API key for MCP authentication
3. **Environment configured**: MCP needs to know where to find the backend

## Configuration

### Environment Variables

Create or update `.env`:

```bash
# Required for MCP server
MCP_API_KEY=<generate-secure-key>
BACKEND_API_KEY=<must-match-backend-API_KEY>
EMAIL_API_URL=http://localhost:8000

# Optional: Enable attachment downloads
MCP_ENABLE_ATTACHMENTS=false
```

Generate a secure key:
```bash
openssl rand -base64 32
```

### Start the MCP Server

```bash
cd mail-done
./run_mcp_server.sh
```

Or directly:
```bash
poetry run python -m mcp_server.server
```

## Configuring AI Assistants

### Claude Code (CLI)

Edit `~/.claude/claude_code_config.json`:

```json
{
  "mcpServers": {
    "email-search": {
      "command": "/path/to/mail-done/run_mcp_server.sh",
      "env": {
        "MCP_API_KEY": "your-mcp-key",
        "BACKEND_API_KEY": "your-backend-key",
        "EMAIL_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

Restart Claude Code after configuration changes.

### Claude Desktop

Edit the Claude Desktop configuration file:

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
**Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "email-search": {
      "command": "/path/to/mail-done/run_mcp_server.sh",
      "env": {
        "MCP_API_KEY": "your-mcp-key",
        "BACKEND_API_KEY": "your-backend-key",
        "EMAIL_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

Restart Claude Desktop after changes.

### Cursor

Edit `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "email-search": {
      "command": "/path/to/mail-done/run_mcp_server.sh",
      "env": {
        "MCP_API_KEY": "your-mcp-key",
        "BACKEND_API_KEY": "your-backend-key",
        "EMAIL_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

Restart Cursor to load the new MCP server.

### Remote Backend (Self-Hosted Server)

If your backend runs on a remote server (e.g., Raspberry Pi via Tailscale):

```json
{
  "mcpServers": {
    "email-search": {
      "command": "/path/to/mail-done/run_mcp_server.sh",
      "env": {
        "MCP_API_KEY": "your-mcp-key",
        "BACKEND_API_KEY": "your-backend-key",
        "EMAIL_API_URL": "http://your-server:8000"
      }
    }
  }
}
```

With Tailscale:
```json
{
  "EMAIL_API_URL": "http://mail-done.your-tailnet.ts.net:8000"
}
```

## Available MCP Tools

### `semantic_search`

Search emails by meaning/concept:

```
Query: "machine learning papers from collaborators"
```

Parameters:
- `query` (required): Natural language search query
- `mode`: "semantic", "keyword", or "hybrid" (default)
- `top_k`: Max results (default: 10)
- `similarity_threshold`: 0-1 (default: 0.6)
- `date_from`, `date_to`: Date range filter (YYYY-MM-DD)
- `category`: Filter by email category

### `search_by_sender`

Find emails from a specific sender:

```
Sender: "john.doe@university.edu"
```

Parameters:
- `sender` (required): Email address, name, or partial match
- `top_k`: Max results (default: 20)
- `date_from`, `date_to`: Date range filter

### `search_by_topic`

Search by research topic:

```
Topic: "CRISPR gene editing"
```

Parameters:
- `topic` (required): Research topic or scientific concept
- `categories`: Filter by categories
- `similarity_threshold`: 0-1 (default: 0.6)

### `get_email_details`

Get full email content:

Parameters:
- `email_id` (required): UUID from search results

Returns full body, headers, attachments, and AI analysis.

### `find_similar_emails`

Find emails similar to a reference:

Parameters:
- `email_id` (required): UUID of reference email
- `top_k`: Number of similar emails (default: 10)
- `exclude_same_sender`: Exclude same sender (default: false)

### `list_categories`

List all email categories with counts.

### `list_top_senders`

List most frequent senders.

### `list_imap_folders`

List available IMAP folders.

### `list_folder_emails`

List emails in a specific folder:

Parameters:
- `folder` (required): Folder path
- `limit`: Max messages (default: 50)
- `since_date`: Only messages since date

### `list_attachments`

List attachments for an email:

Parameters:
- `email_id` (required): UUID of the email

### `download_attachment`

Download an attachment (requires `MCP_ENABLE_ATTACHMENTS=true`):

Parameters:
- `email_id` (required): UUID of the email
- `attachment_index`: 0-based attachment index

## Example Usage

After configuration, you can ask your AI assistant:

**Claude/Cursor prompts:**
- "Search my emails for machine learning papers"
- "Find emails from Stanford collaborators"
- "Show me PhD applications about genomics"
- "What emails need a reply this week?"
- "Find similar emails to this conference invitation"

## Security Features

- **Rate Limiting**: 60 requests/minute per IP
- **Audit Logging**: All requests logged
- **Attachment Restrictions**: Content access requires explicit opt-in
- **Mandatory Authentication**: Both MCP and backend keys required
- **No Direct DB Access**: All queries go through API

## Testing

Test the MCP server:

```bash
poetry run python test_mcp_server.py
```

Verify connection:
```bash
curl http://localhost:8000/health
```

## Troubleshooting

### "Backend not reachable"

1. Ensure backend is running: `curl http://localhost:8000/health`
2. Check `EMAIL_API_URL` is correct
3. Verify network connectivity

### "Authentication failed"

1. Ensure `BACKEND_API_KEY` matches backend's `API_KEY`
2. Check for typos in key values
3. Regenerate keys if needed

### "Import error" when starting MCP

1. Run `poetry install` in the mail-done directory
2. Ensure Python 3.11 is being used
3. Check Poetry environment: `poetry env info`

### Claude/Cursor not showing email tools

1. Verify configuration file syntax (valid JSON)
2. Check the path to `run_mcp_server.sh` is absolute
3. Restart the AI application completely
4. Check MCP server logs for errors

## Date Range Policy

Default search ranges:
- **Default**: Last 3 months (90 days) for fast searches
- **Maximum**: 2 years (730 days) hard limit
- **Custom**: Specify `date_from`/`date_to` for specific ranges

This prevents accidentally scanning the entire email history.
