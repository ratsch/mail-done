# mail-done Web UI

A modern, local web interface for the mail-done email processing system. This lightweight web UI connects to your backend API to provide:

- **Semantic Search** - Search emails using natural language
- **Cost Overview** - Track LLM API usage and costs
- **Inbox Processing** - Trigger email processing jobs
- **Statistics** - View email database statistics

## Features

### Semantic Search
Search your emails using natural language queries. The interface supports three search modes:
- **Hybrid**: Combines keyword matching and semantic similarity (recommended)
- **Semantic**: Pure vector similarity search for finding conceptually related emails
- **Keyword**: Traditional text search

Example queries:
- "machine learning papers"
- "PhD application deadlines"
- "meeting invites from last week"

### Cost Tracking
Monitor your LLM API usage:
- Daily, monthly, and total costs
- Breakdown by model (GPT-4o, GPT-4o-mini, embeddings)
- Breakdown by task (classification, embedding, search)
- Cost projections

### Inbox Processing
Trigger email processing from the web interface:
- Process new (unseen) emails only
- Or process a specific number of emails
- Toggle AI classification and embedding generation
- Real-time status updates

### Statistics Dashboard
View your email database stats:
- Total emails and AI classifications
- Emails with embeddings
- Emails needing replies
- Category breakdown

## Setup

### Prerequisites

- Docker and Docker Compose (or Podman)
- Backend API running (see main project deployment)

### Quick Start

1. **Navigate to the web-ui directory:**

```bash
cd mail-done/web-ui
```

2. **Create `.env` file:**

```bash
cp env-template .env
```

3. **Edit `.env` with your API details:**

```bash
# Backend API URL (local or remote)
BACKEND_API_URL=http://localhost:8000

# Web UI port
WEB_UI_PORT=8080

# Optional: API key if backend requires authentication
# API_KEY=your-api-key
```

4. **Build and run with Docker Compose:**

```bash
# Using the start script (recommended)
./start.sh

# Or manually
docker compose up -d
```

5. **Access the web interface:**

Open your browser to: http://localhost:8080

### Development Mode (without Docker)

If you prefer to run locally without Docker:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The server will start on http://localhost:8080

## Configuration

### Environment Variables

| Variable | Description | Required | Default |
|----------|-------------|----------|---------|
| `BACKEND_API_URL` | URL of your backend API | Yes | - |
| `WEB_UI_PORT` | Port for web server | No | 8080 |
| `API_KEY` | API key for backend auth | No | - |

### Connecting to Remote Backend

For a remote backend (e.g., on a Raspberry Pi or server):

```bash
# Direct connection
BACKEND_API_URL=http://your-server:8000

# Via Tailscale
BACKEND_API_URL=http://mail-done.your-tailnet.ts.net:8000
```

## Usage

### Semantic Search

1. Navigate to the **Semantic Search** tab
2. Enter your query in natural language
3. Select search mode (Hybrid recommended)
4. Choose number of results
5. Click **Search**

Results show:
- Email subject and sender
- Date received
- Content preview
- Similarity score
- Category and reply status

### Cost Overview

1. Navigate to the **Cost Overview** tab
2. Select analysis period (7, 30, or 90 days)
3. Click **Refresh** to load latest data

View:
- Costs for today, this month, and total
- Breakdown by model and task
- Daily cost trends
- Monthly and yearly projections

### Process Inbox

1. Navigate to the **Process Inbox** tab
2. Configure processing options:
   - **Process new emails only**: Only process unseen emails
   - **Limit**: Max emails to process (if not processing new only)
   - **Dry run**: Preview without making changes
   - **Use AI classification**: Enable AI classification
   - **Generate embeddings**: Generate vectors for semantic search
3. Click **Start Processing**
4. Monitor real-time status updates

### Statistics

1. Navigate to the **Statistics** tab
2. Click **Refresh Stats** to load latest data

View:
- Total emails in database
- Emails with embeddings
- AI classifications count
- Emails needing replies
- Top categories breakdown

## Architecture

The web UI is a lightweight proxy to your backend API:

```
┌─────────────┐      HTTP      ┌─────────────┐      HTTP      ┌──────────────┐
│   Browser   │ ◄────────────► │   Web UI    │ ◄────────────► │  Backend API │
│  (You)      │                │  (Docker)   │                │              │
└─────────────┘                └─────────────┘                └──────────────┘
                                      │
                                      │ Triggers
                                      ▼
                              ┌─────────────┐
                              │ process_    │
                              │ inbox.py    │
                              └─────────────┘
```

**Benefits:**
- No duplicate database connections
- Lightweight and fast
- Can run on same machine as backend or separately

## Docker Commands

### Start the service

```bash
# Recommended - auto-detects docker compose version
./start.sh

# Or manually
docker compose up -d
```

### View logs

```bash
docker compose logs -f
```

### Stop the service

```bash
docker compose down
```

### Rebuild after changes

```bash
docker compose up -d --build
```

### Check health status

```bash
curl http://localhost:8080/health
```

## Troubleshooting

### Cannot connect to backend API

**Symptoms**: Red connection status, "Cannot connect to API"

**Solutions**:
1. Verify `BACKEND_API_URL` in `.env` is correct
2. Ensure backend API is running and accessible
3. Test backend directly:
   ```bash
   curl http://localhost:8000/health
   ```
4. Check network connectivity (firewall, VPN, etc.)

### Search returns no results

**Possible causes**:
1. No embeddings generated yet
   - Go to **Process Inbox** tab
   - Enable "Generate embeddings"
   - Process your emails
2. Search query too specific
   - Try broader terms
   - Use hybrid mode instead of semantic-only

### Processing fails to start

**Solutions**:
1. Ensure `process_inbox.py` is accessible
2. Check that Poetry environment is set up correctly
3. Verify IMAP credentials in `.env`
4. Review logs:
   ```bash
   docker compose logs web-ui
   ```

### Cost tracking shows "not available"

**Explanation**: Cost tracking requires the cost endpoints in the backend.

**Solution**: Ensure the backend has cost tracking enabled.

## API Endpoints

The web UI exposes these endpoints:

### Public Endpoints

- `GET /` - Web interface
- `GET /health` - Health check
- `GET /api/search/simple?q={query}&mode={mode}` - Search emails
- `GET /api/stats` - Email statistics
- `GET /api/costs/summary` - Cost summary
- `GET /api/costs/overview?days={N}` - Detailed cost breakdown
- `POST /api/process/trigger` - Trigger inbox processing
- `GET /api/process/status` - Get processing status

All endpoints proxy to the backend API except for process triggering.

## Security Notes

This web UI is designed for **local use only**:

- No authentication (assumes local trusted network)
- Runs on localhost by default
- CORS allows all origins (since it's local)

**Do not expose this to the internet without:**
1. Adding authentication
2. Configuring proper CORS
3. Using HTTPS
4. Adding rate limiting

## Development

### Project Structure

```
web-ui/
├── app.py              # FastAPI backend
├── static/
│   ├── index.html      # Main UI
│   ├── style.css       # Styling
│   └── app.js          # Frontend logic
├── shared_auth/        # Authentication utilities
├── Dockerfile          # Container definition
├── docker-compose.yml  # Docker Compose config
├── requirements.txt    # Python dependencies
├── env-template        # Environment template
└── README.md           # This file
```

### Adding New Features

1. **New API endpoint**: Add to `app.py`
2. **New UI tab**:
   - Add tab button in `index.html`
   - Add tab pane in `index.html`
   - Add JavaScript handlers in `app.js`
   - Add styles in `style.css`

### Tech Stack

**Backend:**
- FastAPI (Python web framework)
- httpx (HTTP client for backend API)
- uvicorn (ASGI server)

**Frontend:**
- Vanilla JavaScript (no framework overhead)
- Modern CSS with CSS Grid and Flexbox
- Responsive design

## Related Documentation

- [mail-done Main Project](../README.md)
- [Deployment Guide](../docs/DEPLOYMENT.md)
- [API Documentation](../docs/API.md)
- [Database Schema](../docs/DATABASE.md)
