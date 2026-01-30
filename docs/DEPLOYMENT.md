# Deployment Guide

This guide covers deploying mail-done to a self-hosted server using Docker or Podman.

## Prerequisites

- Docker or Podman with Compose support
- PostgreSQL 16+ with pgvector extension (included in deployment)
- IMAP email account credentials
- OpenAI API key (or Azure OpenAI credentials)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repo-url> mail-done
cd mail-done
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your credentials
vim .env
```

Required environment variables:

```bash
# Database (auto-generated if using deploy.sh)
POSTGRES_PASSWORD=<secure-password>
DB_ENCRYPTION_KEY=<fernet-key>

# IMAP Credentials
IMAP_HOST=imap.example.com
IMAP_PORT=993
IMAP_USERNAME=your-email@example.com
IMAP_PASSWORD=your-password

# LLM Provider (choose one)
OPENAI_API_KEY=sk-...

# Or Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### 3. Deploy

**Using Docker:**
```bash
./deploy/deploy.sh
```

**Using Podman (e.g., Raspberry Pi):**
```bash
./deploy/deploy.sh --pi
```

The deployment script will:
1. Generate secure passwords if `.env` doesn't exist
2. Build and start all containers
3. Run database migrations
4. Test the health endpoint

## Deployment Options

### Standard Docker Deployment

Uses `deploy/docker-compose.yml`:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Services:
- **db**: PostgreSQL 16 with pgvector
- **api**: FastAPI backend on port 8000

### Raspberry Pi / ARM64 Deployment

Uses `deploy/docker-compose.pi.yml` with host networking:

```bash
podman-compose -f deploy/docker-compose.pi.yml up -d
```

Optimizations:
- Host networking (simpler with Tailscale)
- Memory limits for constrained devices
- Optional Tailscale sidecar for remote access

## Services

### Backend API (port 8000)

The FastAPI backend provides:
- REST API for email access
- Semantic search endpoints
- Statistics and cost tracking
- Health monitoring

Test the API:
```bash
curl http://localhost:8000/health
```

### Web UI (port 8080)

Optional lightweight web interface for:
- Semantic email search
- Inbox processing triggers
- System statistics

Deploy separately:
```bash
cd web-ui
docker compose up -d
```

### PostgreSQL Database

PostgreSQL 16 with extensions:
- **pgvector**: Vector similarity search
- **uuid-ossp**: UUID generation
- **pg_trgm**: Text search

Connection string:
```
postgresql://postgres:$POSTGRES_PASSWORD@localhost:5432/email_processor
```

## Environment Variables Reference

### Required

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | Database password |
| `DB_ENCRYPTION_KEY` | Fernet key for field-level encryption |
| `IMAP_HOST` | IMAP server hostname |
| `IMAP_PORT` | IMAP server port (usually 993) |
| `IMAP_USERNAME` | Email address |
| `IMAP_PASSWORD` | Email password or app password |
| `OPENAI_API_KEY` | OpenAI API key (or use Azure) |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `API_PORT` | Backend API port | 8000 |
| `POSTGRES_PORT` | Database port | 5432 |
| `POSTGRES_USER` | Database user | postgres |
| `POSTGRES_DB` | Database name | email_processor |
| `DRY_RUN` | Disable IMAP modifications | false |
| `CONFIG_DIR` | Custom config directory overlay | ./config |
| `PROMPTS_DIR` | Custom prompts directory overlay | ./backend/core/ai/prompts |

### Azure OpenAI (alternative to OpenAI)

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Deployment name (e.g., gpt-4o) |

## Custom Configuration Overlay

For private configurations (VIP lists, custom prompts), use the overlay system:

```bash
# Set in .env
CONFIG_DIR=/path/to/private-config
PROMPTS_DIR=/path/to/private-config/prompts
```

This allows keeping sensitive configurations in a separate private repository while using the main mail-done codebase.

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "database": "connected",
  "version": "2.0.0"
}
```

### View Logs

```bash
# Docker
docker compose -f deploy/docker-compose.yml logs -f api

# Podman
podman-compose -f deploy/docker-compose.pi.yml logs -f api
```

### Service Status

```bash
docker compose -f deploy/docker-compose.yml ps
```

## Maintenance

### Restart Services

```bash
docker compose -f deploy/docker-compose.yml restart api
```

### Update Deployment

```bash
git pull
docker compose -f deploy/docker-compose.yml up -d --build
```

### Database Backup

```bash
docker exec mail-done-db pg_dump -U postgres email_processor > backup.sql
```

### Database Restore

```bash
docker exec -i mail-done-db psql -U postgres email_processor < backup.sql
```

## Troubleshooting

### API not responding

1. Check container status: `docker compose ps`
2. Check logs: `docker compose logs api`
3. Verify database is healthy: `docker compose logs db`

### Database connection failed

1. Ensure PostgreSQL is running
2. Check `DATABASE_URL` format
3. Verify network connectivity between containers

### IMAP connection failed

1. Verify credentials in `.env`
2. Check if app password is needed (Gmail, etc.)
3. Test manually: `openssl s_client -connect imap.example.com:993`

### Migrations failed

Run migrations manually:
```bash
docker exec mail-done-api poetry run alembic upgrade head
```

## Security Notes

- Change default passwords in production
- Use HTTPS reverse proxy (nginx, Caddy) for external access
- Consider Tailscale for secure remote access
- Enable rate limiting in production
- Keep `DB_ENCRYPTION_KEY` secure and backed up

## Remote Access with Tailscale

For secure remote access without exposing ports:

1. Install Tailscale on the server
2. Uncomment the Tailscale sidecar in `docker-compose.pi.yml`
3. Set `TS_AUTHKEY` in environment
4. Access via Tailscale hostname: `http://mail-done:8000`
