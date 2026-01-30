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

See the dedicated [Raspberry Pi Deployment Guide](#raspberry-pi-deployment-guide) below for complete instructions.

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

---

## Raspberry Pi Deployment Guide

This section provides complete, step-by-step instructions for deploying mail-done on a Raspberry Pi (tested on Pi 4 with 4GB RAM running Raspberry Pi OS Bookworm).

### System Requirements

**Hardware:**
- Raspberry Pi 4 (4GB+ RAM recommended)
- 16GB+ SD card or USB SSD (SSD recommended for faster builds)
- Network connection

**Software:**
- Raspberry Pi OS Bookworm (64-bit) or Debian 12+
- Python 3.11+

### Step 1: Install System Packages

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install required packages
sudo apt-get install -y \
    podman \
    podman-compose \
    git \
    openssl \
    python3 \
    python3-cryptography \
    curl

# Verify installations
podman --version      # Should be 4.x+
podman-compose --version
python3 --version     # Should be 3.11+
```

### Step 2: Clone the Repository

```bash
cd ~

# Option A: Clone via HTTPS (no SSH key required)
git clone https://github.com/ratsch/mail-done.git mail-done

# Option B: Clone via SSH (requires SSH key configured)
git clone git@github.com:ratsch/mail-done.git mail-done

cd mail-done
```

**Note:** If you get `Host key verification failed` with SSH, use the HTTPS URL instead.

### Step 3: Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Generate secure credentials
POSTGRES_PASSWORD=$(openssl rand -hex 16)
DB_ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
API_KEY=$(openssl rand -hex 24)

# Update .env with generated values
sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$POSTGRES_PASSWORD|" .env
sed -i "s|^DB_ENCRYPTION_KEY=.*|DB_ENCRYPTION_KEY=$DB_ENCRYPTION_KEY|" .env
sed -i "s|^API_KEY=.*|API_KEY=$API_KEY|" .env

# Save credentials for reference
echo "Generated credentials:"
echo "  POSTGRES_PASSWORD: $POSTGRES_PASSWORD"
echo "  API_KEY: $API_KEY"
echo "  DB_ENCRYPTION_KEY: (saved in .env)"
```

**Edit `.env` to add your credentials:**

```bash
vim .env
```

Required settings:
```bash
# LLM Provider (choose one)
OPENAI_API_KEY=sk-...

# Or Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://your-instance.openai.azure.com
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# IMAP credentials (optional - for email processing)
IMAP_HOST_WORK=imap.example.com
IMAP_PORT_WORK=993
IMAP_USERNAME_WORK=your-email@example.com
IMAP_PASSWORD_WORK=your-password
```

### Step 4: Setup Config Files

```bash
# Copy all config examples to active configs
for example in config/*.example.yaml; do
    target="${example%.example.yaml}.yaml"
    [ -f "$target" ] || cp "$example" "$target"
done

# Optionally customize configs
ls config/*.yaml
```

### Step 5: Deploy Using the Script

The easiest method uses the provided deployment script:

```bash
./deploy/deploy-pi.sh
```

The script will:
1. Check prerequisites (podman, podman-compose, openssl, python3)
2. Generate credentials if `.env` doesn't exist
3. Copy config examples to active configs
4. Build the API container image
5. Start PostgreSQL with pgvector
6. Start the API container
7. Wait for services to be healthy

**Script options:**
```bash
./deploy/deploy-pi.sh           # Full deployment
./deploy/deploy-pi.sh --status  # Show service status
./deploy/deploy-pi.sh --logs    # View container logs
./deploy/deploy-pi.sh --stop    # Stop all services
./deploy/deploy-pi.sh --clean   # Remove containers, volumes, and configs
```

### Step 6: Initialize Database Tables

After the first deployment, initialize the database schema:

```bash
podman exec mail-done-api python3 -c "
from backend.core.database.connection import init_db, create_tables
init_db()
create_tables()
print('Database tables created successfully')
"
```

### Step 7: Verify Deployment

```bash
# Check container status
podman ps --filter "name=mail-done"

# Test health endpoint
curl http://localhost:8000/health

# Test API info
curl http://localhost:8000/

# Test with API key
API_KEY=$(grep "^API_KEY=" .env | cut -d= -f2)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/emails?limit=5
```

Expected health response:
```json
{
  "status": "healthy",
  "version": "2.0.0",
  "database": "connected",
  "checks": {"database": "ok"}
}
```

**Comprehensive Functional Test:**

Use the provided test script to verify all components:

```bash
# Test local deployment
python3 deploy/test-deployment.py

# Test remote deployment
python3 deploy/test-deployment.py http://hostname:8000
```

Expected output:
```
============================================================
  Mail-Done Deployment Tests
  Target: http://localhost:8000
============================================================

Core Health Checks:
✓ Health endpoint accessible
✓ Status is healthy
✓ Database connected
✓ Database health check passed

API Information:
✓ Root endpoint accessible
✓ Version info present
✓ Features listed
  Version: 2.0.0

API Documentation:
✓ OpenAPI spec accessible
✓ API has endpoints defined (91 found)

Authentication Enforcement:
✓ Stats requires auth
✓ Emails requires auth
✓ Admin endpoints require auth
✓ Applications requires auth

============================================================
  All 12 tests passed!
  Deployment is working correctly.
============================================================
```

### Manual Deployment (Alternative)

If you prefer manual control or the script doesn't work:

```bash
cd ~/mail-done

# Source environment
set -a && source .env && set +a

# Build the API image
podman-compose -f deploy/docker-compose.pi.yml build

# Create database volume
podman volume create mail-done-db-data

# Start PostgreSQL
podman run -d \
    --name mail-done-db \
    --network host \
    --restart unless-stopped \
    -e POSTGRES_USER="${POSTGRES_USER:-postgres}" \
    -e POSTGRES_PASSWORD="$POSTGRES_PASSWORD" \
    -e POSTGRES_DB="${POSTGRES_DB:-email_processor}" \
    -e PGPORT="${POSTGRES_PORT:-5432}" \
    -v mail-done-db-data:/var/lib/postgresql/data \
    -v "$PWD/deploy/init-db.sql:/docker-entrypoint-initdb.d/init.sql:ro" \
    docker.io/pgvector/pgvector:pg16

# Wait for database
sleep 10
podman exec mail-done-db pg_isready -U postgres

# Start API
podman run -d \
    --name mail-done-api \
    --network host \
    --restart unless-stopped \
    --env-file .env \
    -e PORT="${API_PORT:-8000}" \
    -e DATABASE_URL="postgresql://${POSTGRES_USER:-postgres}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT:-5432}/${POSTGRES_DB:-email_processor}" \
    -v "$PWD/config:/app/config:ro" \
    localhost/deploy_api:latest

# Initialize tables
podman exec mail-done-api python3 -c "
from backend.core.database.connection import init_db, create_tables
init_db()
create_tables()
"
```

### Raspberry Pi Troubleshooting

#### Slow Container Builds

**Problem:** STEP 9/11 COPY takes 20+ minutes

**Cause:** Large files (logs, data directories) being copied into the container.

**Solution:** Ensure `.dockerignore` exists and excludes large directories:
```bash
cat .dockerignore | grep -E "email-processor|venv|LOG"
```

If missing, the key exclusions are:
```
email-processor/
venv/
.venv/
LOG_imapsync/
*.log
```

#### Network Mode Error

**Problem:** `Error: cannot set multiple networks without bridge network mode, selected mode host`

**Cause:** Bug in podman-compose with host networking.

**Solution:** The deploy script uses direct `podman run` commands to bypass this. If using podman-compose directly, this error can be ignored if containers start anyway.

#### Registry Resolution Error

**Problem:** `short-name "pgvector/pgvector:pg16" did not resolve to an alias and no unqualified-search registries are defined`

**Cause:** Podman on some systems doesn't have Docker Hub configured as a default registry.

**Solution:** The deploy script now uses full image paths (`docker.io/pgvector/pgvector:pg16`). If you encounter this with other images, use the full path format: `docker.io/image:tag`

Alternatively, configure registries in `/etc/containers/registries.conf`:
```ini
[registries.search]
registries = ['docker.io']
```

#### Memory Cgroups Error

**Problem:** `crun: opening file 'memory.max' for writing: No such file or directory`

**Cause:** cgroups v2 memory controller not fully enabled.

**Solution:** Either:
1. Don't use memory limits (the deploy script already handles this)
2. Or enable cgroups v2 memory controller:
   ```bash
   # Add to /boot/cmdline.txt (Pi OS) or /boot/firmware/cmdline.txt
   cgroup_enable=memory cgroup_memory=1
   # Reboot
   sudo reboot
   ```

#### Database Tables Don't Exist

**Problem:** `relation "emails" does not exist`

**Cause:** Database schema not initialized.

**Solution:** Run table creation:
```bash
podman exec mail-done-api python3 -c "
from backend.core.database.connection import init_db, create_tables
init_db()
create_tables()
"
```

#### Container Keeps Restarting

**Problem:** API container restarts in a loop

**Solution:** Check logs for the actual error:
```bash
podman logs mail-done-api --tail 50
```

Common causes:
- Missing environment variables (check `.env`)
- Database not ready (wait longer, or restart API)
- Invalid config files (check YAML syntax)

### Pi-Specific Performance Notes

1. **First build on fresh system:** Takes 10-15 minutes as it downloads base images and installs all dependencies. Poetry installs ~110 packages.

2. **Subsequent builds:** With Docker cache, builds complete in ~30 seconds (only COPY step runs).

3. **SD card vs SSD:** USB SSD dramatically improves build times and container I/O.

4. **Memory:** With 4GB RAM, both containers run fine. PostgreSQL will use available memory for caching.

5. **Image pulls:** First deployment downloads ~500MB of container images (python:3.11-slim, pgvector/pgvector:pg16).

4. **CPU:** Container builds are CPU-intensive. The Pi 4 handles this but it takes time.

### Systemd Service (Optional)

To auto-start on boot, create a systemd service:

```bash
# Copy the service file
sudo cp deploy/mail-done.service /etc/systemd/system/

# Edit if needed (check paths)
sudo vim /etc/systemd/system/mail-done.service

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable mail-done
sudo systemctl start mail-done

# Check status
sudo systemctl status mail-done
```

### Updating the Deployment

```bash
cd ~/mail-done

# Stop services
./deploy/deploy-pi.sh --stop

# Pull updates
git pull

# Rebuild and restart
./deploy/deploy-pi.sh
```

### Complete Removal

To completely remove the deployment:

```bash
# Stop and remove everything
./deploy/deploy-pi.sh --clean

# Remove the .env file (optional - contains your credentials)
rm .env

# Remove images (optional)
podman rmi localhost/deploy_api pgvector/pgvector:pg16
```

### Authentication Configuration

mail-done uses multiple authentication methods depending on the endpoint:

#### 1. API Key Authentication (Basic API Access)

Most `/api/*` endpoints use simple API key authentication.

**Setup:**
1. The API key is auto-generated during deployment (saved in `.env`)
2. Use the `X-API-Key` header for requests

**Example:**
```bash
# Get your API key
API_KEY=$(grep "^API_KEY=" .env | cut -d= -f2)

# Make authenticated request
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/emails
```

#### 2. Signed Request Authentication (Secure Client Access)

For more secure access (scripts, automation), use Ed25519 signed requests.

**Setup:**

1. Generate a keypair:
```bash
python3 -c "
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import base64

pk = Ed25519PrivateKey.generate()
private_key = base64.b64encode(pk.private_bytes_raw()).decode()
public_key = base64.b64encode(pk.public_key().public_bytes_raw()).decode()

print(f'Private key (keep secret): {private_key}')
print(f'Public key (add to config): {public_key}')
"
```

2. Add the public key to `config/clients.yaml`:
```yaml
clients:
  my-script:
    description: "My automation script"
    public_keys:
      - "YOUR_BASE64_PUBLIC_KEY_HERE"
    scopes:
      - "*"  # Full access, or specific scopes like "emails:read"
    enabled: true
```

3. Use the private key in your scripts to sign requests (see `docs/SECURITY_DESIGN_REQUEST_SIGNING.md`).

#### 3. Admin Panel Authentication (Google OAuth)

The `/admin/*` endpoints require Google OAuth authentication for lab member access.

**Setup:**

1. Create a Google Cloud OAuth 2.0 Client:
   - Go to [Google Cloud Console](https://console.cloud.google.com/)
   - Create or select a project
   - Navigate to APIs & Services > Credentials
   - Create OAuth 2.0 Client ID (Web application)
   - Add authorized redirect URI: `http://localhost:8000/auth/google/callback`

2. Add credentials to `.env`:
```bash
# Google OAuth for Admin Panel
GOOGLE_CLIENT_ID_V0_PORTAL=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET_V0_PORTAL=your-client-secret
GOOGLE_REDIRECT_URI_V0_PORTAL=http://localhost:8000/auth/google/callback

# JWT Configuration
JWT_SECRET=$(openssl rand -hex 32)
JWT_EXPIRATION_HOURS=24
```

3. Create the first admin user directly in the database:
```bash
podman exec mail-done-api python3 -c "
from backend.core.database.connection import init_db, get_session
from backend.core.database.models import LabMember
import uuid

init_db()
session = next(get_session())

admin = LabMember(
    id=uuid.uuid4(),
    email='your-email@example.com',
    name='Admin User',
    role='admin',
    can_review=True,
    is_active=True
)
session.add(admin)
session.commit()
print(f'Created admin user: {admin.email}')
"
```

4. Access the admin panel:
   - Navigate to `http://localhost:8000/admin/`
   - Click "Login with Google"
   - Sign in with the email you registered

**Admin Roles:**
- `admin`: Full access to all admin endpoints
- `reviewer`: Can review applications but not manage users
- `member`: Basic lab member (read-only access)

#### Environment Variables for Authentication

| Variable | Description | Required |
|----------|-------------|----------|
| `API_KEY` | Simple API key for `/api/*` endpoints | Yes |
| `JWT_SECRET` | Secret for signing JWT tokens | For admin panel |
| `JWT_EXPIRATION_HOURS` | JWT token lifetime (default: 24) | No |
| `GOOGLE_CLIENT_ID_V0_PORTAL` | Google OAuth client ID | For admin panel |
| `GOOGLE_CLIENT_SECRET_V0_PORTAL` | Google OAuth client secret | For admin panel |
| `GOOGLE_REDIRECT_URI_V0_PORTAL` | OAuth callback URL | For admin panel |

### API Endpoints Quick Reference

After deployment, these endpoints are available:

| Endpoint | Auth | Description |
|----------|------|-------------|
| `GET /health` | No | Health check |
| `GET /` | No | API info |
| `GET /docs` | No | OpenAPI documentation |
| `GET /api/emails` | API Key | List emails |
| `GET /api/costs/overview` | API Key | Cost analytics |
| `GET /api/debug/config` | API Key | Debug configuration |
| `POST /api/applemail/color` | API Key | Apple Mail color lookup |

Use the API key header: `X-API-Key: <your-api-key>`

Example:
```bash
API_KEY=$(grep "^API_KEY=" .env | cut -d= -f2)
curl -H "X-API-Key: $API_KEY" http://localhost:8000/api/emails?limit=10
```
