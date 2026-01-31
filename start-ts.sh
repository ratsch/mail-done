#!/bin/bash
set -e

# Start Tailscale daemon in userspace mode
tailscaled --tun=userspace-networking --state=/var/lib/tailscale/tailscaled.state &
sleep 3

# Authenticate with Tailscale (--reset to override existing settings)
tailscale up --reset --authkey=${TS_AUTHKEY} --hostname=${TS_HOSTNAME:-md}

# Set up Tailscale serve to proxy the API port
tailscale serve --bg ${PORT:-8000}

# Run database migrations
poetry run alembic upgrade head

# Start the API
exec poetry run uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
