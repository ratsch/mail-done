#!/bin/bash
#
# Process work email using mail-done
#
# Usage:
#   ./process.sh                    # Use default account (work)
#   ./process.sh --account personal # Use specific account
#
# Optional environment variables:
#   CONFIG_DIR - Path to config overlay (mail-done-config/config)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Export CONFIG_DIR if set (for config overlay support)
if [ -n "$CONFIG_DIR" ]; then
    export CONFIG_DIR
    echo "Using CONFIG_DIR: $CONFIG_DIR"
fi

# Parse account argument (default: work)
ACCOUNT="work"
while [[ $# -gt 0 ]]; do
    case $1 in
        --account)
            ACCOUNT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "Processing $ACCOUNT email"
echo ""

# Main inbox processing
poetry run python process_inbox.py --parallel-workers 1 --safe-move --create-folders --limit-past 30 --account "$ACCOUNT" 2>&1 | tee -a process.log

echo ""
echo -n "Updating locations and embeddings..."
poetry run python process_inbox.py --parallel-workers 3 --embeddings-only --limit-past 180 --account "$ACCOUNT" >> process.log 2>&1
echo -n "."

# Lifecycle updates for various folders
poetry run python process_inbox.py --lifecycle-only --folder "MD" --recursive --limit-past 365 --account "$ACCOUNT" >> process.log 2>&1
echo -n "."
poetry run python process_inbox.py --lifecycle-only --folder "Archive" --limit-past 365 --account "$ACCOUNT" >> process.log 2>&1
echo -n "."
poetry run python process_inbox.py --lifecycle-only --folder "Deleted Items" --limit-past 365 --account "$ACCOUNT" >> process.log 2>&1
echo -n "."
poetry run python process_inbox.py --lifecycle-only --folder "Sent Items" --limit-past 365 --account "$ACCOUNT" >> process.log 2>&1
echo -n "."
poetry run python process_inbox.py --lifecycle-only --folder "Drafts" --limit-past 365 --account "$ACCOUNT" >> process.log 2>&1
echo -n "."

# Embeddings for archive folders
poetry run python process_inbox.py --folder "Archive" --parallel-workers 3 --embeddings-only --limit-past 1000 --account "$ACCOUNT" >> process.log 2>&1
echo -n "."
poetry run python process_inbox.py --folder "Sent Items" --parallel-workers 3 --embeddings-only --limit-past 1000 --account "$ACCOUNT" >> process.log 2>&1
echo " Done."

echo ""
echo "Updating Drafts life cycle"
poetry run python process_inbox.py --update-draft-lifecycle --account "$ACCOUNT"

echo ""
echo "List drafts"
poetry run python process_inbox.py --list-drafts --account "$ACCOUNT"
