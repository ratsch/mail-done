#!/bin/bash
#
# Initialize a private configuration overlay for mail-done
#
# This script creates a minimal private config directory that you can
# customize and keep in a separate (private) git repository.
#
# Usage:
#   ./scripts/init-config-overlay.sh ~/mail-done-config
#   ./scripts/init-config-overlay.sh /path/to/config --minimal
#   ./scripts/init-config-overlay.sh /path/to/config --full
#
# Options:
#   --minimal   Only create accounts.yaml (required) and vip_senders.yaml
#   --full      Copy all config files for complete customization
#   (default)   Create recommended set: accounts, vip_senders, classification_rules
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_SOURCE="$PROJECT_DIR/config"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_step() { echo -e "${BLUE}[→]${NC} $1"; }

# Parse arguments
TARGET_DIR=""
MODE="recommended"

while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            MODE="minimal"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --help|-h)
            echo "Usage: $0 <target-directory> [--minimal|--full]"
            echo ""
            echo "Creates a private configuration overlay for mail-done."
            echo ""
            echo "Options:"
            echo "  --minimal   Only accounts.yaml and vip_senders.yaml"
            echo "  --full      All config files"
            echo "  (default)   Recommended set for most users"
            echo ""
            echo "Example:"
            echo "  $0 ~/mail-done-config"
            exit 0
            ;;
        *)
            if [ -z "$TARGET_DIR" ]; then
                TARGET_DIR="$1"
            else
                echo "Error: Unknown argument: $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$TARGET_DIR" ]; then
    echo "Usage: $0 <target-directory> [--minimal|--full]"
    echo ""
    echo "Example: $0 ~/mail-done-config"
    exit 1
fi

# Expand ~ if present
TARGET_DIR="${TARGET_DIR/#\~/$HOME}"

echo ""
echo "=========================================="
echo "  mail-done Config Overlay Initialization"
echo "=========================================="
echo ""
echo "Target directory: $TARGET_DIR"
echo "Mode: $MODE"
echo ""

# Check if target exists
if [ -d "$TARGET_DIR" ]; then
    if [ "$(ls -A "$TARGET_DIR" 2>/dev/null)" ]; then
        log_warn "Directory exists and is not empty: $TARGET_DIR"
        read -p "Continue and add missing files? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    log_step "Creating directory: $TARGET_DIR"
    mkdir -p "$TARGET_DIR"
fi

# Define which files to copy based on mode
case $MODE in
    minimal)
        FILES=(
            "accounts.example.yaml:accounts.yaml"
            "vip_senders.example.yaml:vip_senders.yaml"
        )
        ;;
    full)
        FILES=(
            "accounts.example.yaml:accounts.yaml"
            "vip_senders.example.yaml:vip_senders.yaml"
            "classification_rules.example.yaml:classification_rules.yaml"
            "ai_category_actions.example.yaml:ai_category_actions.yaml"
            "categories.example.yaml:categories.yaml"
            "preprocessing_rules.example.yaml:preprocessing_rules.yaml"
            "clients.example.yaml:clients.yaml"
            "inquiry_templates.example.yaml:inquiry_templates.yaml"
        )
        ;;
    recommended|*)
        FILES=(
            "accounts.example.yaml:accounts.yaml"
            "vip_senders.example.yaml:vip_senders.yaml"
            "classification_rules.example.yaml:classification_rules.yaml"
            "ai_category_actions.example.yaml:ai_category_actions.yaml"
        )
        ;;
esac

# Copy files
log_step "Copying config files..."
for file_pair in "${FILES[@]}"; do
    src="${file_pair%%:*}"
    dst="${file_pair##*:}"

    if [ -f "$TARGET_DIR/$dst" ]; then
        log_warn "Skipping (exists): $dst"
    elif [ -f "$CONFIG_SOURCE/$src" ]; then
        cp "$CONFIG_SOURCE/$src" "$TARGET_DIR/$dst"
        log_info "Created: $dst"
    else
        log_warn "Source not found: $src"
    fi
done

# Create prompts directory
if [ ! -d "$TARGET_DIR/prompts" ]; then
    mkdir -p "$TARGET_DIR/prompts"
    log_info "Created: prompts/"
fi

# Create .gitignore
if [ ! -f "$TARGET_DIR/.gitignore" ]; then
    cat > "$TARGET_DIR/.gitignore" << 'EOF'
# Backup files
*.bak
*.orig
*~

# Editor files
.DS_Store
.idea/
.vscode/
EOF
    log_info "Created: .gitignore"
fi

# Create README
if [ ! -f "$TARGET_DIR/README.md" ]; then
    cat > "$TARGET_DIR/README.md" << 'EOF'
# mail-done Private Configuration

This directory contains private configuration for mail-done.

## Files

| File | Purpose |
|------|---------|
| `accounts.yaml` | Email server settings (IMAP hosts, ports) |
| `vip_senders.yaml` | Priority sender list |
| `classification_rules.yaml` | Rule-based email sorting |
| `ai_category_actions.yaml` | Folder moves per AI category |
| `prompts/` | Custom AI prompts |

## Usage

### Local Development

```bash
export CONFIG_DIR=/path/to/this/directory
python3 process_inbox.py --dry-run --limit 10
```

### Deployment

Add to `.env` on your server:
```bash
CONFIG_DIR=/path/to/this/directory
```

Then deploy or restart the API container.

## Syncing to Server

```bash
# Push changes
git add -A && git commit -m "Update config" && git push

# On server
cd /path/to/this/directory && git pull
# Restart API to pick up changes
```

## Security

This repo should be **private** as it may contain:
- VIP sender information
- Email server hostnames
- Custom classification rules
EOF
    log_info "Created: README.md"
fi

# Initialize git if not already
if [ ! -d "$TARGET_DIR/.git" ]; then
    log_step "Initializing git repository..."
    (cd "$TARGET_DIR" && git init -q)
    log_info "Git repository initialized"
fi

echo ""
echo "=========================================="
echo "  Config Overlay Created!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Edit the config files:"
echo "   ${BLUE}vim $TARGET_DIR/accounts.yaml${NC}"
echo "   ${BLUE}vim $TARGET_DIR/vip_senders.yaml${NC}"
echo ""
echo "2. Set CONFIG_DIR in your environment:"
echo "   ${BLUE}export CONFIG_DIR=$TARGET_DIR${NC}"
echo ""
echo "3. (Optional) Create a private GitHub repo:"
echo "   ${BLUE}cd $TARGET_DIR${NC}"
echo "   ${BLUE}git remote add origin git@github.com:you/mail-done-config.git${NC}"
echo "   ${BLUE}git add -A && git commit -m 'Initial config'${NC}"
echo "   ${BLUE}git push -u origin main${NC}"
echo ""
echo "4. For deployment, clone this repo on your server and set"
echo "   CONFIG_DIR in the server's .env file."
echo ""
