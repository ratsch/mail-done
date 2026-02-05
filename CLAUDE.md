# CLAUDE.md - mail-done

**See also:** `~/git/services/mail-done-config/CLAUDE.md` for deployment details and private config.

## Deployment

**IMPORTANT: To deploy changes to the Pi backend, use:**

```bash
~/git/services/mail-done-config/deploy/sync-and-deploy.sh deploy
```

This script:
- Syncs code to nvme-pi (excluding .env to preserve Pi-specific paths)
- Rebuilds and restarts the container
- Verifies health

**DO NOT** manually rsync or copy files to the Pi - use the script.

### Deployment Commands

```bash
# Sync + deploy backend
~/git/services/mail-done-config/deploy/sync-and-deploy.sh deploy

# Sync + deploy web UI
~/git/services/mail-done-config/deploy/sync-and-deploy.sh webui

# Sync + deploy both
~/git/services/mail-done-config/deploy/sync-and-deploy.sh all

# Check status
~/git/services/mail-done-config/deploy/sync-and-deploy.sh status

# Sync only (no deploy)
~/git/services/mail-done-config/deploy/sync-and-deploy.sh sync
```

## Project Structure

- **backend/** - FastAPI backend (Python)
- **mcp_server/** - MCP server for Claude Code integration
- **deploy/** - Local deployment scripts (use mail-done-config/deploy for Pi)
- **config/** - Default config (Pi uses mail-done-config overlay)

## Related Repositories

- **mail-done-config** (`~/git/services/mail-done-config`) - Private config overlay and Pi deployment scripts
- **lab-application-review** (`~/git/services/lab-application-review`) - Application review portal (Next.js)
