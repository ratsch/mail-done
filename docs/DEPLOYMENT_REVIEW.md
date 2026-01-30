# Deployment Documentation Review

**Date:** 2026-01-30
**Reviewer:** Claude
**Scope:** Complete review of deployment documentation for self-sufficiency

---

## Executive Summary

The current deployment documentation is **not self-sufficient**. A new user following these instructions would encounter multiple blockers. This document identifies all issues and provides specific fixes.

### Overall Assessment

| Category | Status | Notes |
|----------|--------|-------|
| Repository Setup | 🔴 Incomplete | Missing repo URL |
| Environment Config | 🟠 Confusing | .env vs accounts.yaml unclear |
| Minimal Config | 🔴 Missing | No "minimum viable" config documented |
| Config Overlay | 🟠 Incomplete | Mentioned but not actionable |
| Database Setup | 🟠 Confusing | Alembic vs create_tables() conflict |
| Unit Tests | 🔴 Missing | Not in deployment flow |
| Email Processing | 🟠 Incomplete | Not integrated into deployment |
| MCP Server | 🟠 Incomplete | Not part of deployment flow |
| Automation | 🟡 Partial | Cron in PROCESS_INBOX.md but not DEPLOYMENT.md |
| Cross-References | 🟠 Incomplete | Docs exist but not linked in deployment flow |

### Documentation Structure

Current documentation files:

| Document | Purpose | Referenced from DEPLOYMENT.md? |
|----------|---------|-------------------------------|
| `README.md` | Overview, quick start | No (should be entry point) |
| `docs/DEPLOYMENT.md` | Production deployment | N/A (this is the main doc) |
| `docs/API.md` | FastAPI endpoints | ❌ No |
| `docs/DATABASE.md` | Schema, pgvector | ❌ No |
| `docs/MCP.md` | Claude/Cursor integration | ❌ No (mentioned briefly) |
| `docs/PROCESS_INBOX.md` | Email processing | ❌ No |
| `web-ui/README.md` | Web interface | ❌ No |
| `.env.example` | Environment template | ✅ Yes |
| `config/*.example.yaml` | Config templates | ✅ Yes (partially) |

**Issue:** DEPLOYMENT.md should link to all related docs at appropriate points in the deployment flow.

---

## Critical Issues (Deployment Will Fail)

### 1. Missing Repository URL

**Location:** `docs/DEPLOYMENT.md`, line 17

**Current:**
```bash
git clone <repo-url> mail-done
```

**Fix:**
```bash
git clone https://github.com/your-org/mail-done.git
cd mail-done
```

**Action Required:** Add actual repository URL or document that user must substitute their own.

---

### 2. IMAP Configuration Mismatch

**Problem:** `.env.example` and `accounts.yaml` both contain IMAP configuration, but their relationship is unclear.

**Current `.env.example` (lines 37-40):**
```bash
IMAP_USERNAME_WORK=your.email@company.com
IMAP_PASSWORD_WORK=your-app-password
```

**Missing:**
```bash
IMAP_HOST_WORK=mail.example.com
IMAP_PORT_WORK=993
```

**Current `accounts.example.yaml` (lines 13-20):**
```yaml
work:
  imap:
    host: mail.example.com
    port: 993
    # Credentials from: IMAP_USERNAME_WORK, IMAP_PASSWORD_WORK
```

**Fix Required:**
1. Add `IMAP_HOST_*` and `IMAP_PORT_*` to `.env.example`
2. Document clearly: "Hosts go in `accounts.yaml`, credentials go in `.env`"
3. Or: consolidate to single location

**Recommended Fix for `.env.example`:**
```bash
# =============================================================================
# IMAP CONFIGURATION (Required)
# =============================================================================
# NOTE: Server hosts/ports are in config/accounts.yaml
# Only credentials are stored in .env for security

# --- Work Account ---
IMAP_USERNAME_WORK=your.email@company.com
IMAP_PASSWORD_WORK=your-app-password
# Host/port configured in: config/accounts.yaml → accounts.work.imap
```

---

### 3. Dockerfile Port Mismatch

**Location:** `Dockerfile` line 33, 37

**Current:**
```dockerfile
EXPOSE 8080
CMD sh -c "... --port ${PORT:-8080}"
```

**But `docker-compose.yml` sets:**
```yaml
environment:
  PORT: 8000
```

**Fix:** Change Dockerfile to match expected behavior:
```dockerfile
EXPOSE 8000
CMD sh -c "... --port ${PORT:-8000}"
```

---

### 4. DATABASE_URL Host Confusion

**Location:** `.env.example` line 24

**Current:**
```bash
DATABASE_URL=postgresql://postgres:change-this-secure-password@db:5432/email_processor
```

**Problem:** Uses `@db:5432` which is Docker network hostname, but:
- Pi deployment uses `@localhost` (host networking)
- Local development uses `@localhost`

**Fix:** Add comments explaining both scenarios:
```bash
# For Docker deployment (default):
DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/email_processor

# For Raspberry Pi / host networking (uncomment and use this instead):
# DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@localhost:5432/email_processor

# For local development without Docker:
# DATABASE_URL=postgresql://postgres:password@localhost:5432/email_processor
```

---

### 5. OAuth2 Script Reference to Non-Existent File

**Location:** `.env.example` line 56

**Current:**
```bash
# Run: python scripts/oauth2_setup.py to get the refresh token
```

**Problem:** No `scripts/` directory exists in the repository.

**Fix Options:**
1. Create `scripts/oauth2_setup.py` with OAuth2 setup flow
2. Or remove the reference and document manual OAuth2 setup
3. Or link to external documentation

---

## High Priority Issues (Causes Confusion)

### 6. Alembic vs create_tables() Conflict

**Location:** `docs/DEPLOYMENT.md` lines 416-420 vs `Dockerfile` line 37

**Dockerfile runs:**
```bash
[ -f alembic.ini ] && poetry run alembic upgrade head
```

**But docs say to manually run:**
```python
from backend.core.database.connection import init_db, create_tables
init_db()
create_tables()
```

**Fix:** Clarify in DEPLOYMENT.md:
```markdown
### Step 6: Initialize Database

The API container automatically runs Alembic migrations at startup if `alembic.ini` exists.

**If migrations don't exist yet** (fresh deployment), manually create tables:

```bash
podman exec mail-done-api python3 -c "
from backend.core.database.connection import init_db, create_tables
init_db()
create_tables()
print('Database tables created')
"
```

**Note:** After initial setup, Alembic handles all schema changes automatically.
```

---

### 7. No Minimal Configuration Section

**Problem:** No clear documentation of what's the absolute minimum to get started.

**Fix:** Add new section to DEPLOYMENT.md:

```markdown
## Minimal Configuration (Quick Start)

To get mail-done running with basic functionality, you need **only these settings**:

### Required Environment Variables (.env)

```bash
# Database (auto-generated by deploy script)
POSTGRES_PASSWORD=<generated>
DB_ENCRYPTION_KEY=<generated>

# Single email account
IMAP_USERNAME_WORK=your.email@example.com
IMAP_PASSWORD_WORK=your-app-password

# LLM for classification (pick one)
OPENAI_API_KEY=sk-...
```

### Required Config Files

Only ONE config file is strictly required:

1. **`config/accounts.yaml`** - Email server settings
   ```yaml
   accounts:
     work:
       display_name: "Work Email"
       imap:
         host: imap.example.com
         port: 993
         use_ssl: true
   settings:
     default_account: work
   ```

### Optional Config Files

These enhance functionality but aren't required to start:

| File | Purpose | Default Behavior |
|------|---------|------------------|
| `vip_senders.yaml` | Priority senders | No VIP detection |
| `classification_rules.yaml` | Rule-based sorting | AI classification only |
| `ai_category_actions.yaml` | Actions per category | No automatic actions |

### Minimal Deployment Command

```bash
# 1. Clone and configure
git clone <repo-url> mail-done && cd mail-done
cp .env.example .env
# Edit .env with your IMAP credentials and OpenAI key

# 2. Create minimal accounts.yaml
cp config/accounts.example.yaml config/accounts.yaml
# Edit config/accounts.yaml with your IMAP host

# 3. Deploy
./deploy/deploy-pi.sh

# 4. Test
curl http://localhost:8000/health
```
```

---

### 8. Config Overlay Not Actionable

**Location:** `docs/DEPLOYMENT.md` lines 165-173

**Current:**
```markdown
For private configurations (VIP lists, custom prompts), use the overlay system:

```bash
CONFIG_DIR=/path/to/private-config
PROMPTS_DIR=/path/to/private-config/prompts
```
```

**Problem:** No step-by-step instructions for setting up an overlay.

**Fix:** Expand to full section:

```markdown
## Private Configuration Overlay

The overlay system allows you to keep sensitive configurations (VIP lists, institutional rules) in a **separate private repository** while using the main mail-done codebase.

### Why Use Overlays?

- Keep VIP sender lists private (not in public repo)
- Customize prompts for your organization
- Share base config while personalizing rules

### Setting Up a Config Overlay

#### Step 1: Create Private Config Repository

```bash
# Create separate private repo
mkdir ~/mail-done-config
cd ~/mail-done-config
git init

# Create directory structure
mkdir -p prompts
```

#### Step 2: Copy Configs to Overlay

```bash
# Copy only the files you want to customize
cp ~/mail-done/config/vip_senders.example.yaml ~/mail-done-config/vip_senders.yaml
cp ~/mail-done/config/classification_rules.example.yaml ~/mail-done-config/classification_rules.yaml

# Edit with your private settings
vim ~/mail-done-config/vip_senders.yaml
```

#### Step 3: Configure Overlay Paths

Add to `.env`:
```bash
# Point to your private config directory
CONFIG_DIR=/home/user/mail-done-config

# Optional: Custom prompts directory
PROMPTS_DIR=/home/user/mail-done-config/prompts
```

#### Step 4: Mount in Docker/Podman

For Docker deployment, update `docker-compose.yml`:
```yaml
volumes:
  - ${CONFIG_DIR:-../config}:/app/config:ro
```

For Podman (already handled by deploy-pi.sh):
```bash
-v "$CONFIG_DIR:/app/config:ro"
```

### Overlay Directory Structure

```
mail-done-config/           # Your private repo
├── vip_senders.yaml        # VIP definitions (private)
├── classification_rules.yaml
├── ai_category_actions.yaml
└── prompts/                # Custom prompts
    ├── classifier.txt
    └── response_generator.txt
```

### Overlay Precedence

Files in `CONFIG_DIR` completely replace the defaults - there is no merging.
If a file exists in your overlay, it's used; otherwise, the default is used.

### Keeping Overlays in Sync

When mail-done adds new config options:

```bash
# Check for new example files
cd ~/mail-done
git pull
diff config/vip_senders.example.yaml ~/mail-done-config/vip_senders.yaml
```
```

---

### 9. Unit Tests Not in Deployment Flow

**Problem:** DEPLOYMENT.md never mentions running tests to verify deployment.

**Fix:** Add testing section after "Verify Deployment":

```markdown
### Step 8: Run Unit Tests (Recommended)

Verify the deployment is working correctly:

```bash
# Run all tests inside container
podman exec mail-done-api poetry run pytest -v

# Run with coverage
podman exec mail-done-api poetry run pytest --cov=backend

# Quick smoke test
podman exec mail-done-api poetry run pytest backend/tests/unit/test_health.py -v
```

Expected output:
```
========================= test session starts ==========================
collected 45 items

backend/tests/unit/test_health.py::test_health_endpoint PASSED
backend/tests/unit/test_database.py::test_connection PASSED
...
========================= 45 passed in 12.34s ==========================
```

**If tests fail:**
1. Check database connection: `podman logs mail-done-db`
2. Check environment variables: `podman exec mail-done-api env | grep -E "DATABASE|API"`
3. Check config files mounted: `podman exec mail-done-api ls -la /app/config/`
```

---

### 10. Email Processing Not in Deployment Flow

**Problem:** After deployment, user has no guidance on actually processing emails.

**Fix:** Add section after testing:

```markdown
### Step 9: Process Your First Emails

With deployment complete, process emails:

```bash
# Preview mode first (safe - no changes)
podman exec mail-done-api python3 process_inbox.py --dry-run --limit 10

# Process new emails
podman exec mail-done-api python3 process_inbox.py --new-only --limit 50
```

See [docs/PROCESS_INBOX.md](PROCESS_INBOX.md) for complete processing guide.

### Step 10: Set Up Automation (Optional)

For automatic email processing, add a cron job:

```bash
# Edit crontab
crontab -e

# Add: Process new emails every 15 minutes
*/15 * * * * podman exec mail-done-api python3 process_inbox.py --new-only >> /var/log/mail-done.log 2>&1
```

Or use the systemd timer (see [docs/PROCESS_INBOX.md#automation](PROCESS_INBOX.md#automation)).
```

---

### 11. MCP Server Not Part of Deployment

**Problem:** MCP server setup is in separate doc but not integrated into deployment flow.

**Fix:** Add section after email processing:

```markdown
### Step 11: Configure MCP Server (Optional)

To enable email search in Claude or Cursor:

```bash
# Test MCP server locally
./run_mcp_server.sh
```

Then configure your AI assistant (Claude Code, Claude Desktop, or Cursor):

```json
{
  "mcpServers": {
    "email-search": {
      "command": "/path/to/mail-done/run_mcp_server.sh",
      "env": {
        "BACKEND_API_KEY": "<your-API_KEY-from-.env>",
        "EMAIL_API_URL": "http://localhost:8000"
      }
    }
  }
}
```

See [docs/MCP.md](MCP.md) for complete MCP configuration.
```

---

## Medium Priority Issues

### 12. No Encryption Key Backup Warning

**Location:** Throughout deployment docs

**Fix:** Add prominent warning:

```markdown
## ⚠️ CRITICAL: Backup Your Encryption Key

The `DB_ENCRYPTION_KEY` in `.env` encrypts sensitive email data.

**If you lose this key, all encrypted data is UNRECOVERABLE.**

After deployment:
```bash
# Backup the key securely
grep DB_ENCRYPTION_KEY .env > ~/secure-location/mail-done-encryption-key.txt
chmod 600 ~/secure-location/mail-done-encryption-key.txt
```

Store this backup:
- In a password manager
- On encrypted external storage
- NOT in the same location as your deployment
```

---

### 13. Web UI Deployment Unclear

**Location:** `deploy/docker-compose.yml` lines 85-106

**Problem:** Web UI is commented out with no guidance on when/how to enable.

**Fix:** Add section:

```markdown
### Optional: Enable Web UI

The Web UI provides a browser interface for email search and management.

To enable, edit `deploy/docker-compose.yml` and uncomment the `web-ui` service:

```yaml
web-ui:
  build:
    context: ../web-ui
    dockerfile: Dockerfile
  # ... rest of config
```

Then restart:
```bash
docker compose -f deploy/docker-compose.yml up -d --build
```

Access at: http://localhost:8080
```

---

### 14. No Log Rotation

**Problem:** Container logs will grow indefinitely.

**Fix:** Add logging section:

```markdown
### Configure Log Rotation

For production deployments, limit log growth:

**Docker:**
Add to `docker-compose.yml`:
```yaml
services:
  api:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

**Podman:**
```bash
podman run ... --log-opt max-size=10m --log-opt max-file=3
```
```

---

### 15. No SSL/TLS Setup

**Problem:** Docs mention "use HTTPS reverse proxy" but give no examples.

**Fix:** Add section:

```markdown
### Secure Access with Caddy (Recommended)

For HTTPS access, use Caddy as reverse proxy:

```bash
# Install Caddy
sudo apt install caddy

# Configure /etc/caddy/Caddyfile
mail-done.yourdomain.com {
    reverse_proxy localhost:8000
}

# Start Caddy
sudo systemctl enable --now caddy
```

Caddy automatically provisions Let's Encrypt certificates.

**Alternative: Use Tailscale**

For private access without public exposure:
```bash
tailscale up
# Access via: http://your-pi-hostname:8000
```
```

---

## Medium Priority Issues (Continued)

### 16. Missing Cross-References to Other Documents

**Problem:** DEPLOYMENT.md exists in isolation without proper links to companion documentation.

**Required Cross-References:**

| At This Point | Link To | Purpose |
|---------------|---------|---------|
| After "Test API health" | `docs/API.md` | Full API endpoint reference |
| After database setup | `docs/DATABASE.md` | Schema details, troubleshooting |
| After "Configure Rules" | `docs/PROCESS_INBOX.md` | Detailed rule syntax, examples |
| After deployment complete | `docs/MCP.md` | AI assistant integration |
| In troubleshooting section | All docs | Cross-reference for specific issues |

**Fix:** Add a "Related Documentation" section at the end of DEPLOYMENT.md:

```markdown
## Related Documentation

After deployment, consult these guides:

| Guide | Use Case |
|-------|----------|
| [Email Processing](PROCESS_INBOX.md) | Configure rules, process emails, set up automation |
| [MCP Server](MCP.md) | Integrate with Claude, Claude Desktop, or Cursor |
| [API Reference](API.md) | Build integrations, understand endpoints |
| [Database Schema](DATABASE.md) | Query data directly, understand structure |
| [Web UI](../web-ui/README.md) | Browser-based email search |

### Quick Links

- **"How do I process emails?"** → [PROCESS_INBOX.md](PROCESS_INBOX.md)
- **"How do I search emails from Claude?"** → [MCP.md](MCP.md)
- **"What API endpoints exist?"** → [API.md](API.md)
- **"How is data stored?"** → [DATABASE.md](DATABASE.md)
```

**Also add inline references** at key points:

After health check:
```markdown
Test the API:
```bash
curl http://localhost:8000/health
```

For full API documentation, see [docs/API.md](API.md).
```

After database initialization:
```markdown
For database schema details and direct queries, see [docs/DATABASE.md](DATABASE.md).
```

After config files section:
```markdown
For detailed rule syntax and examples, see [docs/PROCESS_INBOX.md](PROCESS_INBOX.md#configuration-files).
```

---

## Low Priority Issues

### 17. Documents Don't Reference Each Other

**Problem:** The individual docs (API.md, DATABASE.md, MCP.md, PROCESS_INBOX.md) don't link to each other.

**Example:** MCP.md mentions "Backend API must be running" but doesn't link to DEPLOYMENT.md or API.md.

**Fix:** Add header to each doc:

```markdown
---
**Prerequisites:** [Deployment Guide](DEPLOYMENT.md) | **Related:** [API Reference](API.md)
---
```

**Document Dependency Map:**

```
README.md (entry point)
    ↓
DEPLOYMENT.md (deploy first)
    ↓
    ├── DATABASE.md (schema reference)
    ├── API.md (endpoint reference)
    ├── PROCESS_INBOX.md (email processing)
    │       ↓
    │       └── config/*.yaml files
    └── MCP.md (AI integration)
            ↓
            └── Requires: API running
```

---

### 18. No Local Development Instructions

**Problem:** Only production deployment covered, not local dev setup.

**Fix:** Add to README.md or create `docs/DEVELOPMENT.md`:

```markdown
## Local Development

### Without Docker

```bash
# Install PostgreSQL with pgvector locally
# ... platform-specific instructions

# Install dependencies
poetry install

# Set up environment
cp .env.example .env
# Edit .env: DATABASE_URL=postgresql://postgres:password@localhost:5432/email_processor

# Run migrations
poetry run alembic upgrade head

# Start API
poetry run uvicorn backend.api.main:app --reload --port 8000
```
```

---

### 19. Config Files Relationship Unclear

**Problem:** Many config files with unclear dependencies.

**Fix:** Add config reference table:

```markdown
## Configuration Files Reference

| File | Required | Purpose | Depends On |
|------|----------|---------|------------|
| `accounts.yaml` | **Yes** | Email server settings | `.env` credentials |
| `categories.yaml` | No | Email category definitions | Built-in defaults |
| `vip_senders.yaml` | No | Priority sender list | None |
| `classification_rules.yaml` | No | Rule-based sorting | `categories.yaml` |
| `ai_category_actions.yaml` | No | Actions per AI category | `categories.yaml` |
| `preprocessing_rules.yaml` | No | Forwarding detection | None |
| `clients.yaml` | No | API client auth | For signed requests only |
```

---

## Summary of Required Changes

### Files to Modify

1. **`docs/DEPLOYMENT.md`** - Major updates:
   - [ ] Add repository URL
   - [ ] Add minimal configuration section
   - [ ] Expand config overlay instructions
   - [ ] Add unit testing step
   - [ ] Add email processing step
   - [ ] Add MCP configuration step
   - [ ] Add encryption key backup warning
   - [ ] Clarify Alembic vs create_tables()
   - [ ] Add Web UI enable instructions
   - [ ] Add log rotation
   - [ ] Add SSL/TLS setup
   - [ ] Add "Related Documentation" section with links to all docs
   - [ ] Add inline cross-references at appropriate points

2. **`.env.example`** - Updates:
   - [ ] Add missing `IMAP_HOST_*` variables or clarify relationship
   - [ ] Fix `DATABASE_URL` comments for different scenarios
   - [ ] Remove or fix OAuth2 script reference

3. **`Dockerfile`** - Fix:
   - [ ] Change default port from 8080 to 8000

4. **`README.md`** - Updates:
   - [ ] Add local development section or link to DEVELOPMENT.md

5. **`docs/API.md`** - Add cross-references:
   - [ ] Add prerequisites header linking to DEPLOYMENT.md
   - [ ] Link to MCP.md for AI integration

6. **`docs/DATABASE.md`** - Add cross-references:
   - [ ] Add prerequisites header
   - [ ] Link to DEPLOYMENT.md for setup

7. **`docs/MCP.md`** - Add cross-references:
   - [ ] Add prerequisites header linking to DEPLOYMENT.md
   - [ ] Link to API.md for endpoint details

8. **`docs/PROCESS_INBOX.md`** - Add cross-references:
   - [ ] Add prerequisites header
   - [ ] Link to config file examples

### New Files to Create

1. **`docs/DEVELOPMENT.md`** - Local development guide
2. **`scripts/oauth2_setup.py`** - OAuth2 token setup (or remove references)

### Estimated Effort

| Task | Complexity | Time |
|------|------------|------|
| Update DEPLOYMENT.md | High | 2-3 hours |
| Fix .env.example | Low | 30 min |
| Fix Dockerfile | Low | 5 min |
| Create overlay instructions | Medium | 1 hour |
| Add testing section | Low | 30 min |
| Add cross-references to all docs | Medium | 1 hour |
| Create DEVELOPMENT.md | Medium | 1 hour |
| Total | | ~7 hours |

---

## Verification Checklist

After fixes, a new user should be able to:

- [ ] Clone repository using documented URL
- [ ] Understand minimum required configuration
- [ ] Deploy without errors using `deploy-pi.sh`
- [ ] Run unit tests and see them pass
- [ ] Process first batch of emails
- [ ] Set up config overlay for private settings
- [ ] Configure MCP for Claude/Cursor
- [ ] Set up automation (cron)
- [ ] Enable HTTPS access

---

*Review completed. Recommend implementing fixes before next deployment.*
