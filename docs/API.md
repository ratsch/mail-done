# FastAPI Backend Documentation

> **Prerequisites:** [Deployment Guide](DEPLOYMENT.md) | **Related:** [MCP Server](MCP.md), [Database](DATABASE.md)

The mail-done backend provides a RESTful API for email management, search, and processing.

## Quick Start

### Start the Server

```bash
# Development (with auto-reload)
poetry run uvicorn backend.api.main:app --reload --port 8000

# Production (via Docker)
docker compose -f deploy/docker-compose.yml up -d api
```

### API Documentation

Interactive documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Authentication

All `/api/*` endpoints require authentication via API key:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/emails
```

The API key is set via the `API_KEY` environment variable.

### Alternative Authentication Methods

The API supports multiple authentication methods:
- **API Key**: `X-API-Key` header
- **Signed Requests**: HMAC-SHA256 signatures for MCP integration
- **OAuth 2.0**: For web UI authentication
- **JWT Tokens**: Bearer token authentication

## Core Endpoints

### Health & Info

#### `GET /` - API Info
```json
{
  "name": "mail-done API",
  "version": "2.0.0",
  "status": "running"
}
```

#### `GET /health` - Health Check
```json
{
  "status": "healthy",
  "database": "connected",
  "version": "2.0.0"
}
```

### Email Endpoints

#### `GET /api/emails` - List Emails

Query parameters:
- `page` (int): Page number (default: 1)
- `page_size` (int): Items per page (default: 50, max: 100)
- `folder` (string): Filter by folder
- `category` (string): Filter by AI category
- `vip_level` (string): Filter by VIP level ("urgent", "high", "medium")
- `needs_reply` (boolean): Filter emails needing reply
- `is_seen` (boolean): Filter read/unread
- `is_flagged` (boolean): Filter flagged emails
- `search` (string): Full-text search

Example:
```bash
curl -H "X-API-Key: xxx" \
  "http://localhost:8000/api/emails?page=1&page_size=20&is_seen=false"
```

Response:
```json
{
  "emails": [
    {
      "id": "uuid",
      "message_id": "...",
      "from_address": "sender@example.com",
      "subject": "Email subject",
      "date": "2025-01-15T10:00:00",
      "folder": "INBOX",
      "is_seen": false,
      "email_metadata": {
        "vip_level": "high",
        "ai_category": "work",
        "ai_confidence": 0.95,
        "needs_reply": true
      }
    }
  ],
  "total": 150,
  "page": 1,
  "pages": 8
}
```

#### `GET /api/emails/{email_id}` - Get Email Details

Returns full email content including body and sender history.

#### `PUT /api/emails/{email_id}/metadata` - Update Metadata

```json
{
  "user_notes": "Follow up next week",
  "project_tags": ["project-alpha"],
  "needs_reply": true
}
```

#### `DELETE /api/emails/{email_id}` - Delete Email

Removes email from database (not from IMAP server).

### Search Endpoints

#### `GET /api/search/semantic` - Semantic Search

Search emails using natural language:

```bash
curl -H "X-API-Key: xxx" \
  "http://localhost:8000/api/search/semantic?q=machine+learning+papers&mode=hybrid&limit=20"
```

Parameters:
- `q` (string): Natural language query
- `mode` (string): "semantic", "keyword", or "hybrid" (default)
- `limit` (int): Max results (default: 20)
- `similarity_threshold` (float): Minimum similarity (0-1, default: 0.6)

#### `GET /api/search/by-sender` - Search by Sender

```bash
curl -H "X-API-Key: xxx" \
  "http://localhost:8000/api/search/by-sender?sender=john@example.com"
```

#### `GET /api/search/by-topic` - Search by Topic

Optimized for research/academic topics:

```bash
curl -H "X-API-Key: xxx" \
  "http://localhost:8000/api/search/by-topic?topic=CRISPR+gene+editing"
```

### Statistics Endpoints

#### `GET /api/stats` - System Statistics

```json
{
  "total_emails": 5234,
  "emails_today": 45,
  "ai_classifications": 4200,
  "needs_reply_count": 12,
  "unread_count": 156,
  "categories_breakdown": {
    "work": 2100,
    "personal": 450,
    "newsletter": 1200
  }
}
```

#### `GET /api/stats/senders` - Top Senders

List senders with email counts.

#### `GET /api/stats/categories/breakdown` - Category Breakdown

```bash
curl -H "X-API-Key: xxx" \
  "http://localhost:8000/api/stats/categories/breakdown?time_range=week"
```

### Folder Endpoints

#### `GET /api/emails/folders/list` - List Folders

```json
{
  "folders": [
    {"name": "INBOX", "total": 350, "unread": 42},
    {"name": "Sent", "total": 1250, "unread": 0}
  ]
}
```

## Processing Endpoints

#### `POST /api/process/trigger` - Trigger Processing

Start email processing job:

```json
{
  "new_only": true,
  "limit": 100,
  "dry_run": false,
  "use_ai": true,
  "generate_embeddings": true
}
```

#### `GET /api/process/status` - Processing Status

Get current processing job status.

## Cost Tracking Endpoints

#### `GET /api/costs/summary` - Cost Summary

```json
{
  "today": 0.45,
  "this_month": 12.34,
  "total": 156.78
}
```

#### `GET /api/costs/overview` - Detailed Cost Breakdown

```bash
curl -H "X-API-Key: xxx" \
  "http://localhost:8000/api/costs/overview?days=30"
```

## Error Handling

All errors return JSON with consistent format:

```json
{
  "detail": "Error description",
  "error_code": "VALIDATION_ERROR"
}
```

Common HTTP status codes:
- `400`: Bad request (validation error)
- `401`: Unauthorized (missing/invalid API key)
- `404`: Resource not found
- `429`: Rate limited
- `500`: Internal server error

## Rate Limiting

Default limits:
- 60 requests/minute per IP
- 600 requests/minute per API key

Rate limit headers:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## Architecture

```
backend/
├── api/
│   ├── main.py           # FastAPI app, middleware
│   ├── routes/           # API route handlers
│   ├── signed_auth.py    # Signed request authentication
│   └── rate_limiting.py  # Rate limiter
├── core/
│   ├── ai/               # LLM classification
│   ├── email/            # Email processing
│   ├── database/         # SQLAlchemy models
│   ├── search/           # Vector search
│   └── auth/             # Authentication
└── tests/                # Test suite
```
