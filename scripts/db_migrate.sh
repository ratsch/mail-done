#!/bin/bash
#
# Database Migration Helper Script
#
# Usage:
#   ./scripts/db_migrate.sh              # Run pending migrations
#   ./scripts/db_migrate.sh --status     # Show current migration status
#   ./scripts/db_migrate.sh --stamp      # Mark existing DB as current (no changes)
#   ./scripts/db_migrate.sh --help       # Show help
#
# For existing databases that already have all tables:
#   ./scripts/db_migrate.sh --stamp
#
# For new databases:
#   ./scripts/db_migrate.sh
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Load environment
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

show_help() {
    echo "Database Migration Helper"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  (none)      Run pending migrations (upgrade to head)"
    echo "  --status    Show current migration status"
    echo "  --stamp     Mark existing database as up-to-date without running migrations"
    echo "  --history   Show migration history"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  # For new databases - run all migrations:"
    echo "  $0"
    echo ""
    echo "  # For existing databases with all tables already created:"
    echo "  $0 --stamp"
    echo ""
    echo "  # Check current status:"
    echo "  $0 --status"
}

check_database() {
    if [ -z "$DATABASE_URL" ]; then
        log_error "DATABASE_URL not set. Check your .env file."
        exit 1
    fi
    log_info "Database URL configured"
}

run_migrations() {
    log_info "Running database migrations..."
    poetry run alembic upgrade head
    log_info "Migrations complete!"
}

show_status() {
    log_info "Current migration status:"
    poetry run alembic current
    echo ""
    log_info "Pending migrations:"
    poetry run alembic history --verbose | head -20
}

stamp_database() {
    log_warn "This will mark the database as being at the latest migration"
    log_warn "without actually running any migrations."
    echo ""
    log_warn "Only use this for EXISTING databases that already have all tables."
    echo ""
    read -p "Continue? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi

    log_info "Stamping database as current..."
    poetry run alembic stamp head
    log_info "Database stamped as up-to-date!"
}

show_history() {
    log_info "Migration history:"
    poetry run alembic history --verbose
}

# Main
case "${1:-}" in
    --help|-h)
        show_help
        ;;
    --status)
        check_database
        show_status
        ;;
    --stamp)
        check_database
        stamp_database
        ;;
    --history)
        check_database
        show_history
        ;;
    "")
        check_database
        run_migrations
        ;;
    *)
        log_error "Unknown option: $1"
        show_help
        exit 1
        ;;
esac
