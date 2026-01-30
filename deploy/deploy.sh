#!/bin/bash
#
# mail-done Deployment Script
#
# Deploys mail-done to a server using Docker/Podman
#
# Usage:
#   ./deploy.sh              # Deploy with docker compose
#   ./deploy.sh --podman     # Deploy with podman-compose (for Pi)
#   ./deploy.sh --pi         # Alias for --podman with Pi compose file
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Parse arguments
USE_PODMAN=false
USE_PI_CONFIG=false
COMPOSE_FILE="deploy/docker-compose.yml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --podman)
            USE_PODMAN=true
            shift
            ;;
        --pi)
            USE_PODMAN=true
            USE_PI_CONFIG=true
            COMPOSE_FILE="deploy/docker-compose.pi.yml"
            shift
            ;;
        --compose-file|-f)
            COMPOSE_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--podman] [--pi] [--compose-file FILE]"
            exit 1
            ;;
    esac
done

# Select compose command
if $USE_PODMAN; then
    COMPOSE_CMD="podman-compose"
else
    COMPOSE_CMD="docker compose"
fi

echo "=========================================="
echo "  mail-done Deployment"
echo "=========================================="
echo ""
echo "Project directory: $PROJECT_DIR"
echo "Compose file: $COMPOSE_FILE"
echo "Using: $COMPOSE_CMD"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "Creating .env from .env.example..."
    cp .env.example .env

    # Generate secure passwords
    POSTGRES_PASSWORD=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)
    DB_ENCRYPTION_KEY=$(openssl rand -base64 32)
    API_KEY=$(openssl rand -base64 32)

    # Update .env with generated values
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS sed
        sed -i '' "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$POSTGRES_PASSWORD/" .env
        sed -i '' "s/^DB_ENCRYPTION_KEY=.*/DB_ENCRYPTION_KEY=$DB_ENCRYPTION_KEY/" .env
        sed -i '' "s/^API_KEY=.*/API_KEY=$API_KEY/" .env
    else
        # Linux sed
        sed -i "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$POSTGRES_PASSWORD/" .env
        sed -i "s/^DB_ENCRYPTION_KEY=.*/DB_ENCRYPTION_KEY=$DB_ENCRYPTION_KEY/" .env
        sed -i "s/^API_KEY=.*/API_KEY=$API_KEY/" .env
    fi

    echo ""
    echo "Generated secure credentials in .env"
    echo ""
    echo "IMPORTANT: Configure the following in .env:"
    echo "  - IMAP credentials (IMAP_USERNAME_WORK, IMAP_PASSWORD_WORK)"
    echo "  - LLM API keys (OPENAI_API_KEY or AZURE_OPENAI_*)"
    echo ""
    read -p "Press Enter to continue after configuring .env..."
fi

# Copy config examples if not already configured
echo "Checking config files..."
CONFIG_CREATED=false
for example in config/*.example.yaml; do
    [ -f "$example" ] || continue
    target="${example%.example.yaml}.yaml"
    if [ ! -f "$target" ]; then
        cp "$example" "$target"
        echo "  Created: $(basename "$target")"
        CONFIG_CREATED=true
    fi
done

if $CONFIG_CREATED; then
    echo ""
    echo "Config files created from examples."
    echo "You may want to customize them in config/*.yaml"
    echo ""
else
    echo "  All config files already exist."
fi

# Load environment
set -a
source .env
set +a

# Check config overlay
if [ -n "$CONFIG_DIR" ]; then
    if [ -d "$CONFIG_DIR" ]; then
        echo "Using private config overlay: $CONFIG_DIR"
    else
        echo ""
        echo "ERROR: CONFIG_DIR is set but directory does not exist: $CONFIG_DIR"
        echo ""
        echo "Either:"
        echo "  1. Create/clone the config directory at $CONFIG_DIR"
        echo "  2. Or remove CONFIG_DIR from .env to use default configs"
        exit 1
    fi
else
    echo "Using default config: $PROJECT_DIR/config"
fi

echo ""
echo "Building containers..."
$COMPOSE_CMD -f "$COMPOSE_FILE" build

echo ""
echo "Starting services..."
$COMPOSE_CMD -f "$COMPOSE_FILE" up -d

echo ""
echo "Waiting for services to be ready..."
sleep 10

# Check status
echo ""
echo "Service status:"
$COMPOSE_CMD -f "$COMPOSE_FILE" ps

# Test health endpoint
API_PORT=${API_PORT:-8000}
echo ""
echo "Testing API health..."
if curl -s "http://localhost:$API_PORT/health" > /dev/null 2>&1; then
    echo "API is healthy at http://localhost:$API_PORT"
    curl -s "http://localhost:$API_PORT/health" | python3 -m json.tool 2>/dev/null || curl -s "http://localhost:$API_PORT/health"
else
    echo "API not responding yet. Check logs with:"
    echo "  $COMPOSE_CMD -f $COMPOSE_FILE logs -f api"
fi

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "API URL: http://localhost:$API_PORT"
echo ""
echo "Useful commands:"
echo "  View logs:     $COMPOSE_CMD -f $COMPOSE_FILE logs -f"
echo "  Stop:          $COMPOSE_CMD -f $COMPOSE_FILE down"
echo "  Restart API:   $COMPOSE_CMD -f $COMPOSE_FILE restart api"
echo ""
