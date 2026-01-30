# mail-done Web UI

A modern, local web interface for the mail-done email processing system. This lightweight web UI connects to your Railway API to provide:

- 🔍 **Semantic Search** - Search emails using natural language
- 💰 **Cost Overview** - Track OpenAI API usage and costs
- ⚙️ **Inbox Processing** - Trigger email processing jobs
- 📊 **Statistics** - View email database statistics

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
Monitor your OpenAI API usage:
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

- Docker and Docker Compose
- Access to your Railway API (must be deployed and running)
- Railway API key

### Quick Start

1. **Clone or navigate to the web-ui directory:**

```bash
cd mail-done/web-ui
```

2. **Create `.env` file:**

```bash
cp .env.example .env
```

3. **Edit `.env` with your Railway API details:**

```bash
RAILWAY_API_URL=https://your-app.railway.app
WEB_UI_PORT=8080
```

**Note:** No API keys needed - authentication uses signed requests via OAuth.

4. **Build and run with Docker Compose:**

```bash
# Using the start script (recommended - auto-detects docker compose version)
./start.sh

# Or manually:
# For newer Docker Desktop (Docker Compose V2)
docker compose up -d

# For older docker-compose (V1)
docker-compose up -d
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
| `RAILWAY_API_URL` | URL of your Railway API | Yes | - |
| `WEB_UI_PORT` | Port for web server | No | 8080 |

### Getting Your Railway API Details

1. **Railway API URL**:
   - Go to your Railway project dashboard
   - Find your deployed service
   - Copy the public URL (e.g., `https://your-app.railway.app`)

**Note:** No API keys needed - authentication uses signed requests via OAuth.

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

**Note**: Cost tracking requires the cost endpoints to be deployed on Railway. If not available, the UI will show basic stats only.

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

The processing runs in the background on your local machine and connects to your IMAP server.

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

This web UI is a lightweight proxy to your Railway API:

```
┌─────────────┐      HTTP      ┌─────────────┐      Database      ┌──────────────┐
│   Browser   │ ◄────────────► │   Web UI    │ ◄────────────────► │   Railway    │
│  (You)      │                │  (Docker)   │                    │     API      │
└─────────────┘                └─────────────┘                    └──────────────┘
                                      │
                                      │ Triggers local
                                      ▼
                              ┌─────────────┐
                              │ process_    │
                              │ inbox.py    │
                              └─────────────┘
```

**Benefits:**
- No duplicate database connections
- Leverages existing Railway API
- Lightweight and fast
- Can run on same machine as email processor

## Docker Commands

### Start the service

```bash
# Recommended - auto-detects docker compose version
./start.sh

# Or manually with Docker Compose V2 (newer)
docker compose up -d

# Or with docker-compose V1 (older)
docker-compose up -d
```

### View logs

```bash
# Docker Compose V2
docker compose logs -f

# docker-compose V1
docker-compose logs -f
```

### Stop the service

```bash
# Docker Compose V2
docker compose down

# docker-compose V1
docker-compose down
```

### Rebuild after changes

```bash
# Docker Compose V2
docker compose up -d --build

# docker-compose V1
docker-compose up -d --build
```

### Check health status

```bash
curl http://localhost:8080/health
```

## Troubleshooting

### Cannot connect to Railway API

**Symptoms**: Red connection status, "Cannot connect to Railway API"

**Solutions**:
1. Verify `RAILWAY_API_URL` in `.env` is correct
2. Ensure Railway API is running and accessible
3. Check Railway API key is set correctly
4. Test Railway API directly:
   ```bash
   curl https://your-app.railway.app/health
   ```

### Search returns no results

**Possible causes**:
1. No embeddings generated yet
   - Go to **Process Inbox** tab
   - Enable "Generate embeddings"
   - Process your emails
2. Search query too specific
   - Try broader terms
   - Use hybrid mode instead of semantic-only
3. API key not configured
   - Check Railway API key in `.env`

### Processing fails to start

**Solutions**:
1. Ensure `process_inbox.py` is accessible
2. Check that Poetry environment is set up correctly
3. Verify IMAP credentials in email-processor `.env`
4. Review logs:
   ```bash
   docker-compose logs web-ui
   ```

### Cost tracking shows "not available"

**Explanation**: Cost tracking endpoints need to be added to Railway API.

**Solution**: The web UI is ready, but you need to deploy the cost tracking endpoints from `email-processor/backend/core/database/cost_tracking.py` to Railway.

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

All endpoints proxy to Railway API except for process triggering.

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
├── Dockerfile          # Container definition
├── docker-compose.yml  # Docker Compose config
├── requirements.txt    # Python dependencies
├── .env.example        # Environment template
└── README.md          # This file
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
- httpx (HTTP client for Railway API)
- uvicorn (ASGI server)

**Frontend:**
- Vanilla JavaScript (no framework overhead)
- Modern CSS with CSS Grid and Flexbox
- Responsive design

## Roadmap

Future enhancements:

- [ ] Add dedicated cost endpoints to Railway API
- [ ] Real-time processing progress with WebSockets
- [ ] Email preview modal with full content
- [ ] Export search results to CSV
- [ ] Saved searches and filters
- [ ] Dark mode toggle
- [ ] Email analytics dashboard

## License

Part of the mail-done project. See main project for license details.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review Railway API logs
3. Check Docker logs: `docker-compose logs`
4. Ensure all environment variables are set correctly

## Related Documentation

- [mail-done Main Project](../README.md)
- [Email Processor Documentation](../email-processor/README.md)
- [Railway Deployment Guide](../email-processor/RAILWAY_DEPLOYMENT.md)
- [Search Documentation](../email-processor/ADVANCED_SEARCH_COMPLETE.md)
- [Cost Tracking](../email-processor/COST_TRACKING.md)

