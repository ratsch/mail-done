#!/bin/bash
#
# mail-done Raspberry Pi Deployment Script
#
# This script handles the complete deployment of mail-done on a Raspberry Pi
# running Debian/Raspberry Pi OS with Podman.
#
# Usage:
#   ./deploy/deploy-pi.sh              # Full deployment
#   ./deploy/deploy-pi.sh --clean      # Remove existing deployment first
#   ./deploy/deploy-pi.sh --status     # Show deployment status
#   ./deploy/deploy-pi.sh --stop       # Stop services
#   ./deploy/deploy-pi.sh --logs       # Show logs
#
# Prerequisites:
#   - Raspberry Pi with Debian/Raspberry Pi OS (bookworm or later)
#   - Git installed
#   - Internet access for pulling images
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COMPOSE_FILE="deploy/docker-compose.pi.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Prerequisites check
# =============================================================================
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check for podman
    if ! command -v podman &> /dev/null; then
        log_error "podman not found. Installing..."
        sudo apt-get update
        sudo apt-get install -y podman
    fi

    # Check for podman-compose
    if ! command -v podman-compose &> /dev/null; then
        log_info "Installing podman-compose..."
        sudo apt-get update
        sudo apt-get install -y podman-compose
    fi

    # Check for openssl (for generating secrets)
    if ! command -v openssl &> /dev/null; then
        log_error "openssl not found. Please install it."
        exit 1
    fi

    # Check for python3 (for generating Fernet key)
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found. Please install it."
        exit 1
    fi

    # Check for cryptography module
    if ! python3 -c "from cryptography.fernet import Fernet" 2>/dev/null; then
        log_warn "cryptography module not found. Installing..."
        pip3 install cryptography --break-system-packages 2>/dev/null || \
            sudo apt-get install -y python3-cryptography
    fi

    log_info "All prerequisites satisfied."
}

# =============================================================================
# Environment setup
# =============================================================================
setup_environment() {
    log_info "Setting up environment..."
    cd "$PROJECT_DIR"

    if [ ! -f ".env" ]; then
        log_info "Creating .env from .env.example..."
        cp .env.example .env

        # Generate secure credentials
        POSTGRES_PASSWORD=$(openssl rand -hex 16)
        DB_ENCRYPTION_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
        API_KEY=$(openssl rand -hex 24)

        # Update .env with generated values
        sed -i "s|^POSTGRES_PASSWORD=.*|POSTGRES_PASSWORD=$POSTGRES_PASSWORD|" .env
        sed -i "s|^DB_ENCRYPTION_KEY=.*|DB_ENCRYPTION_KEY=$DB_ENCRYPTION_KEY|" .env
        sed -i "s|^API_KEY=.*|API_KEY=$API_KEY|" .env

        log_info "Generated secure credentials:"
        echo "  POSTGRES_PASSWORD: $POSTGRES_PASSWORD"
        echo "  API_KEY: $API_KEY"
        echo "  DB_ENCRYPTION_KEY: (generated)"
        echo ""
        log_warn "IMPORTANT: Configure the following in .env before starting:"
        echo "  - IMAP credentials (IMAP_USERNAME_WORK, IMAP_PASSWORD_WORK)"
        echo "  - LLM API keys (OPENAI_API_KEY or AZURE_OPENAI_*)"
    else
        log_info ".env already exists, skipping credential generation."
    fi
}

# =============================================================================
# Config files setup
# =============================================================================
setup_config_files() {
    log_info "Setting up config files..."
    cd "$PROJECT_DIR"

    CONFIG_CREATED=false
    for example in config/*.example.yaml; do
        [ -f "$example" ] || continue
        target="${example%.example.yaml}.yaml"
        if [ ! -f "$target" ]; then
            cp "$example" "$target"
            log_info "  Created: $(basename "$target")"
            CONFIG_CREATED=true
        fi
    done

    if $CONFIG_CREATED; then
        log_info "Config files created from examples."
        echo "  You may want to customize them in config/*.yaml"
    else
        log_info "All config files already exist."
    fi
}

# =============================================================================
# Build and start services
# =============================================================================
build_and_start() {
    log_info "Building containers..."
    cd "$PROJECT_DIR"

    # Source environment for podman-compose
    set -a
    source .env
    set +a

    podman-compose -f "$COMPOSE_FILE" build

    log_info "Starting services..."
    podman-compose -f "$COMPOSE_FILE" up -d

    log_info "Waiting for services to be ready..."
    sleep 15

    # Check database health
    log_info "Checking database health..."
    for i in {1..30}; do
        if podman exec mail-done-db pg_isready -U postgres -d email_processor &>/dev/null; then
            log_info "Database is ready."
            break
        fi
        if [ $i -eq 30 ]; then
            log_error "Database failed to start. Check logs with: podman logs mail-done-db"
            exit 1
        fi
        sleep 2
    done

    # Check API health
    log_info "Checking API health..."
    API_PORT=${API_PORT:-8000}
    for i in {1..30}; do
        if curl -s "http://localhost:$API_PORT/health" &>/dev/null; then
            log_info "API is healthy at http://localhost:$API_PORT"
            curl -s "http://localhost:$API_PORT/health" | python3 -m json.tool 2>/dev/null || \
                curl -s "http://localhost:$API_PORT/health"
            break
        fi
        if [ $i -eq 30 ]; then
            log_warn "API not responding yet. Check logs with: podman logs mail-done-api"
        fi
        sleep 2
    done
}

# =============================================================================
# Clean deployment
# =============================================================================
clean_deployment() {
    log_info "Cleaning existing deployment..."
    cd "$PROJECT_DIR"

    # Source environment if exists
    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
    fi

    # Stop and remove containers
    podman-compose -f "$COMPOSE_FILE" down -v 2>/dev/null || true

    # Remove images
    podman rmi mail-done-api 2>/dev/null || true
    podman rmi localhost/mail-done_api 2>/dev/null || true

    # Remove config files (but keep .env)
    log_info "Removing generated config files..."
    for example in config/*.example.yaml; do
        [ -f "$example" ] || continue
        target="${example%.example.yaml}.yaml"
        if [ -f "$target" ]; then
            rm -f "$target"
            log_info "  Removed: $(basename "$target")"
        fi
    done

    log_info "Clean complete."
}

# =============================================================================
# Show status
# =============================================================================
show_status() {
    log_info "Deployment status:"
    cd "$PROJECT_DIR"

    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
    fi

    echo ""
    podman-compose -f "$COMPOSE_FILE" ps 2>/dev/null || echo "No services running."

    echo ""
    log_info "Container status:"
    podman ps -a --filter "name=mail-done" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

    echo ""
    API_PORT=${API_PORT:-8000}
    if curl -s "http://localhost:$API_PORT/health" &>/dev/null; then
        log_info "API health: OK"
    else
        log_warn "API health: NOT RESPONDING"
    fi
}

# =============================================================================
# Stop services
# =============================================================================
stop_services() {
    log_info "Stopping services..."
    cd "$PROJECT_DIR"

    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
    fi

    podman-compose -f "$COMPOSE_FILE" down
    log_info "Services stopped."
}

# =============================================================================
# Show logs
# =============================================================================
show_logs() {
    cd "$PROJECT_DIR"

    if [ -f ".env" ]; then
        set -a
        source .env
        set +a
    fi

    podman-compose -f "$COMPOSE_FILE" logs -f
}

# =============================================================================
# Main
# =============================================================================
main() {
    echo "=========================================="
    echo "  mail-done Raspberry Pi Deployment"
    echo "=========================================="
    echo ""
    echo "Project directory: $PROJECT_DIR"
    echo ""

    case "${1:-deploy}" in
        --clean)
            clean_deployment
            ;;
        --status)
            show_status
            ;;
        --stop)
            stop_services
            ;;
        --logs)
            show_logs
            ;;
        deploy|"")
            check_prerequisites
            setup_environment
            setup_config_files
            build_and_start

            echo ""
            echo "=========================================="
            echo "  Deployment Complete!"
            echo "=========================================="
            echo ""
            echo "API URL: http://localhost:${API_PORT:-8000}"
            echo ""
            echo "Useful commands:"
            echo "  Status:    ./deploy/deploy-pi.sh --status"
            echo "  Logs:      ./deploy/deploy-pi.sh --logs"
            echo "  Stop:      ./deploy/deploy-pi.sh --stop"
            echo "  Clean:     ./deploy/deploy-pi.sh --clean"
            echo ""
            echo "Run tests:"
            echo "  podman exec mail-done-api poetry run pytest -v"
            echo ""
            ;;
        *)
            echo "Usage: $0 [--clean|--status|--stop|--logs]"
            exit 1
            ;;
    esac
}

main "$@"
