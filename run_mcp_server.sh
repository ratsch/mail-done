#!/bin/bash
# MCP Server launcher script
# This ensures Poetry runs in the correct directory

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$SCRIPT_DIR"
exec poetry run python -m mcp_server.server
