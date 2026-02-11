#!/bin/bash
set -e

# Start Tailscale daemon in userspace mode
tailscaled --tun=userspace-networking --state=/var/lib/tailscale/tailscaled.state &
sleep 3

# Authenticate with Tailscale
# Try with authkey first; if it fails (expired/single-use key),
# fall back to checking if already authenticated from persisted state
if ! tailscale up --reset --authkey=${TS_AUTHKEY} --hostname=${TS_HOSTNAME:-md} 2>/dev/null; then
    echo "Authkey failed, checking existing Tailscale state..."
    if tailscale status >/dev/null 2>&1; then
        echo "Tailscale already authenticated from persisted state"
    else
        echo "ERROR: Tailscale not authenticated and authkey failed"
        exit 1
    fi
fi

# Set up Tailscale serve to proxy the API port
tailscale serve --bg ${PORT:-8000}

# Run database migrations
poetry run alembic upgrade head

# Start the API
exec poetry run uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8000}
