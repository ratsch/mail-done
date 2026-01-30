# Email Processing Guide

> **Prerequisites:** [Deployment Guide](DEPLOYMENT.md) | **Related:** [MCP Server](MCP.md), [API Reference](API.md)

`process_inbox.py` is the main script for processing emails through the classification and action pipeline.

## Quick Start

### Safe Preview (Dry Run)

```bash
# Preview what would happen to 10 emails
python3.11 process_inbox.py --dry-run --limit 10

# Preview all unread emails
python3.11 process_inbox.py --dry-run --new-only
```

### Process Emails

```bash
# Process only unread emails
python3.11 process_inbox.py --new-only

# Process specific number of emails
python3.11 process_inbox.py --limit 100

# Process all emails (careful!)
python3.11 process_inbox.py --all
```

## Processing Pipeline

Each email goes through:

```
1. Parse Email
   ├── Extract headers, body, attachments
   └── Convert to markdown
        ↓
2. Preprocessing
   ├── Detect forwarded emails
   └── Extract original sender
        ↓
3. VIP Detection
   ├── Match against VIP list
   └── Assign priority level
        ↓
4. Rule-Based Classification
   ├── Match conditions (sender, domain, subject)
   └── Assign category and actions
        ↓
5. AI Classification (optional)
   ├── LLM-based categorization
   └── Confidence scoring
        ↓
6. Generate Embeddings (optional)
   └── Vector for semantic search
        ↓
7. Execute Actions
   ├── Move to folder
   ├── Apply color/flag
   └── Add labels
```

## Command Line Options

```
Usage: process_inbox.py [options]

Selection:
  --new-only          Process only UNSEEN emails
  --limit N           Process only N emails (default: 100)
  --all               Process ALL emails (overrides --limit)
  --folder NAME       Which folder to process (default: INBOX)
  --recursive         Process folder and subfolders

Processing:
  --dry-run           Show what would happen (NO changes)
  --reprocess         Re-run AI on already-processed emails
  --actions-only      Only execute pending actions

Phases:
  --skip-ai           Skip AI classification
  --skip-embeddings   Skip embedding generation
  --skip-tracking     Skip response tracking
  --skip-vip          Skip VIP detection
  --skip-rules        Skip rule-based classification

AI Options:
  --use-two-stage     Use two-stage classifier (gpt-4o-mini → gpt-4o)
  --generate-drafts   Generate draft replies from AI suggestions

Output:
  --verbose           Show detailed processing info
  --quiet             Minimal output
```

## Common Workflows

### Daily Processing

```bash
# Morning: Process new emails
python3.11 process_inbox.py --new-only

# Check what's pending
python3.11 process_inbox.py --dry-run --new-only
```

### Initial Setup

```bash
# Step 1: Preview on sample
python3.11 process_inbox.py --dry-run --limit 100

# Step 2: Review output, adjust rules if needed
vim config/classification_rules.yaml

# Step 3: Process in batches
python3.11 process_inbox.py --limit 500
python3.11 process_inbox.py --limit 500  # Repeat

# Step 4: Process remaining
python3.11 process_inbox.py --all
```

### Testing New Rules

```bash
# After editing rules
python3.11 process_inbox.py --dry-run --limit 50

# If satisfied
python3.11 process_inbox.py --limit 50
```

### Reprocess with AI

```bash
# Re-run AI on emails that were previously processed
python3.11 process_inbox.py --reprocess --limit 100

# Skip embeddings to save costs
python3.11 process_inbox.py --reprocess --skip-embeddings
```

### Generate Embeddings Only

```bash
# Add embeddings to emails missing them
python3.11 process_inbox.py --skip-ai --limit 500
```

### Execute Pending Actions

```bash
# Apply actions from previous dry-run or failed execution
python3.11 process_inbox.py --actions-only
```

## Configuration Files

### `config/vip_senders.yaml`

Define VIP senders with priority levels:

```yaml
urgent:
  - email: "ceo@company.com"
    name: "CEO"
  - domain: "important-client.com"

high:
  - email: "manager@company.com"
    name: "Manager"
  - pattern: ".*@board.company.com"

medium:
  - domain: "partner-company.com"
```

VIP levels apply Apple Mail colors:
- 🔴 Red (1) = urgent
- 🟠 Orange (2) = high
- 🟡 Yellow (3) = medium

### `config/classification_rules.yaml`

Rule-based classification:

```yaml
rules:
  - name: sentry-alerts
    priority: 10
    conditions:
      sender:
        match_type: contains
        value: "@getsentry.com"
    actions:
      - type: move
        folder: Notifications/Sentry

  - name: newsletter
    priority: 50
    conditions:
      subject:
        match_type: contains
        value: newsletter
        case_sensitive: false
    actions:
      - type: move
        folder: Newsletters
      - type: color
        value: 6  # Gray
```

Condition match types:
- `equals`: Exact match
- `contains`: Substring match
- `regex`: Regular expression
- `domain`: Match email domain

Action types:
- `move`: Move to folder
- `color`: Apply Apple Mail color (1-7)
- `label`: Add IMAP keyword
- `archive`: Move to Archive
- `keep`: Keep in inbox (stop processing)

### `config/ai_category_actions.yaml`

Actions for AI categories:

```yaml
categories:
  work:
    color: null  # No color
    folder: null  # Keep in inbox

  newsletter:
    color: 6
    folder: Newsletters

  notification:
    color: 7
    folder: Notifications

  review-request:
    color: 5  # Purple
    folder: null
```

### `config/preprocessing_rules.yaml`

Extract original senders from forwarded emails:

```yaml
forwarding_addresses:
  - "forward@yourdomain.com"
  - "alerts@yourdomain.com"

extraction_patterns:
  - pattern: "From: (?P<sender>.*?)\n"
    type: header
  - pattern: "Original Message.*?From: (?P<sender>.*?)\n"
    type: body
```

## Output Example

### Dry Run

```
📧 Email Processing Pipeline
======================================================================
Folder: INBOX
Mode: New emails only
Limit: 20
Dry Run: True
======================================================================

[11354] noreply@getsentry.com
        Subject: METAGRAPH-API - Database error
        Action: Rule: Move → Notifications/Sentry
        🔍 DRY RUN: Would execute 1 action(s)

[11355] colleague@university.edu
        Subject: Important Meeting
        🔴 VIP: URGENT
        Action: VIP: Color 1 (urgent)
        🔍 DRY RUN: Would execute 1 action(s)

======================================================================
📊 PROCESSING REPORT
======================================================================
Total Emails Processed: 20
Processing Time: 1.2s

🔴 VIP Emails Detected: 3 (15.0%)
📋 Emails Matched Rules: 8 (40.0%)

🔍 DRY RUN: No changes were made
======================================================================
```

### Live Mode

```
⚡ ACTIONS EXECUTED:
   move: 30
   color: 25

📁 Folders Used:
     15 → Notifications/Sentry
      8 → Newsletters/Science
      5 → Social/LinkedIn

🎨 Colors Applied:
     12 → Red (VIP Urgent)
      8 → Orange (VIP High)
      3 → Yellow (VIP Medium)

✅ Changes applied to IMAP server
```

## Safety Features

### Dry Run Mode

Shows exactly what would happen without making changes:
- Preview classification results
- See which actions would execute
- Test new rules safely

### Confirmation Prompts

Interactive confirmations for risky operations:
- Processing > 100 emails
- Processing ALL emails
- Non-dry-run mode

### Detailed Logging

All operations are logged:
- Email processing results
- Actions taken
- Errors encountered
- Performance statistics

## Performance

Typical speeds:
- **15-30 emails/second** for rule-based processing
- **2-5 emails/second** with AI classification
- **5-10 emails/second** with embedding generation

Estimates:
- 100 emails: ~5 seconds (rules only)
- 1,000 emails: ~45 seconds
- 10,000 emails: ~7 minutes

## Troubleshooting

### Rules Not Matching

1. Check pattern syntax in YAML
2. Test with `--verbose` flag
3. Verify case sensitivity settings
4. Use `--limit 1` to debug single email

### VIPs Not Detected

1. Check `config/vip_senders.yaml` syntax
2. Verify email addresses are correct
3. Check if preprocessing is needed for forwards

### Actions Not Applied

1. Remove `--dry-run` flag
2. Check IMAP write permissions
3. Verify folder names are valid

### AI Classification Failing

1. Check `OPENAI_API_KEY` is set
2. Verify API quota/credits
3. Check model availability
4. Review error logs

## Automation

### Cron Job

```bash
# Process new emails every hour
0 * * * * cd /path/to/mail-done && poetry run python process_inbox.py --new-only >> logs/process.log 2>&1
```

### Systemd Service

```ini
[Unit]
Description=mail-done email processor
After=network.target

[Service]
Type=oneshot
WorkingDirectory=/path/to/mail-done
ExecStart=/usr/bin/poetry run python process_inbox.py --new-only
User=mail-done

[Timer]
OnCalendar=hourly
Persistent=true

[Install]
WantedBy=timers.target
```

## Undo Actions

### Moved Emails

1. Open your email client
2. Navigate to destination folder
3. Select emails
4. Move back to INBOX

### Applied Colors

In Apple Mail:
1. Select email
2. Message menu → Flag → Clear Flag

### Reset Classification

```bash
# Reprocess to re-classify
python3.11 process_inbox.py --reprocess --limit N
```
