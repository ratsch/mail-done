# Configuration Guide

This document covers all configuration options for mail-done, including the overlay system for private configurations.

## Configuration Overview

mail-done uses a layered configuration system:

```
┌─────────────────────────────────────┐
│     Environment Variables           │  ← Highest priority
├─────────────────────────────────────┤
│     CONFIG_DIR Overlay              │  ← Private/institutional configs
├─────────────────────────────────────┤
│     config/ Directory               │  ← Default configs (in repo)
└─────────────────────────────────────┘
```

This allows you to:
- Keep sensitive configs (VIP lists, institutional rules) in a private repository
- Override defaults without modifying the main codebase
- Use different configs for development vs production

## Configuration Overlay System

### How It Works

Set the `CONFIG_DIR` environment variable to point to your private configuration directory:

```bash
export CONFIG_DIR=/path/to/mail-done-config
```

When loading a config file (e.g., `vip_senders.yaml`), the system:

1. First checks `$CONFIG_DIR/vip_senders.yaml`
2. If not found, falls back to `config/vip_senders.yaml`

This applies to ALL configuration files:
- `vip_senders.yaml`
- `classification_rules.yaml`
- `ai_category_actions.yaml`
- `preprocessing_rules.yaml`
- `accounts.yaml`
- `model_config.yaml`
- `clients.yaml`

### Prompt Overlays

For custom AI prompts, use the `PROMPTS_DIR` variable:

```bash
export PROMPTS_DIR=/path/to/mail-done-config/prompts
```

This allows overriding the default prompts in `backend/core/ai/prompts/` with institution-specific versions.

### Example Directory Structure

```
mail-done-config/           # Your private config repo
├── vip_senders.yaml        # Your VIP sender definitions
├── classification_rules.yaml
├── ai_category_actions.yaml
├── accounts.yaml           # Your email account credentials
├── model_config.yaml
└── prompts/                # Custom AI prompts
    ├── classification.py
    └── application.py
```

### Setting Up a Private Config Repo

1. Create a new private repository:
   ```bash
   mkdir ~/git/mail-done-config
   cd ~/git/mail-done-config
   git init
   ```

2. Copy configs you want to customize:
   ```bash
   cp ~/git/services/mail-done/config/vip_senders.yaml .
   cp ~/git/services/mail-done/config/accounts.yaml .
   # Edit as needed
   ```

3. Set environment variable:
   ```bash
   # Add to .env or shell profile
   export CONFIG_DIR=~/git/mail-done-config
   ```

4. Keep configs in sync:
   ```bash
   # Check for new config options in main repo
   diff ~/git/services/mail-done/config/ ~/git/mail-done-config/
   ```

---

## Configuration Files Reference

### vip_senders.yaml

Defines VIP senders who receive priority handling.

```yaml
# VIP sender definitions
vip_senders:
  # By email address
  - email: "important.person@example.com"
    name: "Important Person"
    priority: high
    category_override: "vip-correspondence"

  # By domain (all emails from this domain)
  - domain: "funding-agency.gov"
    priority: high
    category_override: "funding"

  # By pattern (regex)
  - pattern: ".*@.*\\.edu$"
    priority: medium
    note: "Academic senders"

# Priority levels: critical, high, medium, low
# category_override: Force emails to this category regardless of AI classification
```

### classification_rules.yaml

Rule-based pre-classification before AI analysis.

```yaml
# Rules are evaluated in order; first match wins
rules:
  # Match by sender domain
  - name: "Nature journals"
    match:
      sender_domain: "nature.com"
    action:
      category: "journal-notification"
      skip_ai: true  # Don't send to LLM

  # Match by subject pattern
  - name: "Calendar invites"
    match:
      subject_contains: ["invitation", "calendar", "meeting request"]
      has_attachment_type: "text/calendar"
    action:
      category: "calendar"
      skip_ai: true

  # Match by sender and subject
  - name: "GitHub notifications"
    match:
      sender_domain: "github.com"
      subject_pattern: "^\\[.*\\]"
    action:
      category: "notifications-github"
      skip_ai: true

  # Complex rules
  - name: "PhD applications"
    match:
      subject_contains: ["phd", "doctoral", "graduate position"]
      body_contains: ["application", "cv", "resume"]
    action:
      category: "application-phd"
      skip_ai: false  # Still run AI for detailed analysis
```

### ai_category_actions.yaml

IMAP actions to execute for each AI-assigned category.

```yaml
# Actions per category
categories:
  application-phd:
    move_to: "MD/Applications/PhD"
    add_labels: ["Application", "PhD"]
    flag: true
    color: blue

  application-postdoc:
    move_to: "MD/Applications/Postdoc"
    add_labels: ["Application", "Postdoc"]
    flag: true

  spam:
    move_to: "Junk"
    mark_read: true

  newsletter:
    move_to: "MD/Newsletters"
    mark_read: true

  # Default for unmatched categories
  _default:
    move_to: "MD/Inbox"

# Folder prefix by account (for multi-account setups)
account_prefixes:
  work: "MD/"
  personal: "Processed/"
```

### accounts.yaml

Multi-account email configuration.

```yaml
accounts:
  work:
    imap_host: "outlook.office365.com"
    imap_port: 993
    username: "user@company.com"
    # Password from environment: IMAP_PASSWORD_WORK
    auth_method: "oauth2"  # or "password"
    oauth2_client_id: "..."
    oauth2_tenant_id: "..."
    folders:
      inbox: "INBOX"
      processed: "Processed"
    enabled: true

  personal:
    imap_host: "imap.gmail.com"
    imap_port: 993
    username: "user@gmail.com"
    # Password from environment: IMAP_PASSWORD_PERSONAL
    auth_method: "password"  # App password
    folders:
      inbox: "INBOX"
      processed: "Processed"
    enabled: true

  # Account-specific routing
  routing:
    # Move cross-account: emails from personal that belong in work
    - from_account: "personal"
      to_account: "work"
      match:
        sender_domain: "company.com"
```

### model_config.yaml

LLM model selection per email category.

```yaml
# Default model for all categories
default:
  provider: "openai"
  model: "gpt-4o-mini"
  temperature: 0.1

# Override for specific categories
categories:
  # Use more powerful model for applications
  application-phd:
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.0

  application-postdoc:
    provider: "openai"
    model: "gpt-4o"
    temperature: 0.0

  # Use fast model for simple categories
  newsletter:
    provider: "openai"
    model: "gpt-4o-mini"
    temperature: 0.1

  # Use Azure OpenAI for specific categories
  confidential:
    provider: "azure"
    deployment: "gpt-4o-private"
    temperature: 0.0

# Two-stage classification
two_stage:
  enabled: true
  # First stage: fast triage
  stage1:
    provider: "openai"
    model: "gpt-4o-mini"
  # Second stage: detailed analysis (only for complex cases)
  stage2:
    provider: "openai"
    model: "gpt-4o"
```

### preprocessing_rules.yaml

Rules for detecting forwarded emails and extracting original senders.

```yaml
# Forwarding detection patterns
forwarding_patterns:
  - pattern: "^Fwd:"
    type: "forward"
  - pattern: "^FW:"
    type: "forward"
  - pattern: "^Forwarded message"
    type: "forward"

# Sender extraction from forwarded emails
sender_extraction:
  # Patterns to find original sender in body
  - pattern: "From: (?P<name>.*?) <(?P<email>.*?)>"
  - pattern: "From: (?P<email>[\\w.-]+@[\\w.-]+)"

# Preprocessing rules
rules:
  # Extract sender from assistant-forwarded emails
  - name: "Assistant forwards"
    match:
      sender_email: "assistant@company.com"
      subject_contains: "FW:"
    action:
      extract_original_sender: true
      preserve_forward_chain: true
```

### clients.yaml

API client authentication configuration.

```yaml
clients:
  # MCP server client
  mcp_server:
    api_key_env: "MCP_API_KEY"
    rate_limit: 100  # requests per minute
    allowed_endpoints:
      - "/api/search/*"
      - "/api/emails/*"
      - "/api/documents/*"

  # Web UI client
  web_ui:
    api_key_env: "WEB_UI_API_KEY"
    rate_limit: 200

  # External integration
  external:
    api_key_env: "EXTERNAL_API_KEY"
    rate_limit: 50
    allowed_endpoints:
      - "/api/search/semantic"
```

---

## Environment Variables

### Required Variables

```bash
# Database
DATABASE_URL=postgresql://user:pass@host:5432/maildb
DB_ENCRYPTION_KEY=<fernet-key>  # Generate: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Email (primary account)
IMAP_HOST=imap.gmail.com
IMAP_PORT=993
IMAP_USERNAME=user@gmail.com
IMAP_PASSWORD=<app-password>

# LLM
OPENAI_API_KEY=sk-...
```

### Optional Variables

```bash
# Configuration overlay
CONFIG_DIR=/path/to/private-config
PROMPTS_DIR=/path/to/private-config/prompts

# Multi-account passwords
IMAP_PASSWORD_WORK=...
IMAP_PASSWORD_PERSONAL=...

# Azure OpenAI (alternative to OpenAI)
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_VERSION=2025-04-01-preview

# Google Drive export
GOOGLE_DRIVE_FOLDER_ID=<folder-id>
GOOGLE_SERVICE_ACCOUNT_FILE=/path/to/service-account.json

# MCP Server
MCP_API_KEY=<secure-key>
MCP_ENABLE_FILE_DOWNLOADS=true
MCP_FILE_CACHE_DIR=~/Downloads/.mcp_cache

# API
API_KEY=<backend-api-key>
API_KEY_USER_EMAIL=admin@example.com  # Email for API key user

# Safety
DRY_RUN=false  # Set to true to disable IMAP modifications

# Logging
LOG_LEVEL=INFO
```

---

## Configuration Best Practices

### 1. Separate Sensitive Configs

Keep credentials and institutional data in a private overlay:

```bash
# Public repo (mail-done)
config/
├── vip_senders.example.yaml
├── classification_rules.yaml  # Generic rules
└── ai_category_actions.yaml

# Private repo (mail-done-config)
├── vip_senders.yaml          # Actual VIP list
├── accounts.yaml             # Account credentials
└── classification_rules.yaml # Institution-specific rules
```

### 2. Use Environment Variables for Secrets

Never put passwords or API keys in config files:

```yaml
# BAD - don't do this
accounts:
  work:
    password: "my-secret-password"  # NO!

# GOOD - reference environment variable
accounts:
  work:
    password_env: "IMAP_PASSWORD_WORK"  # Loaded from $IMAP_PASSWORD_WORK
```

### 3. Version Control Your Configs

Keep your private config repo in git (just keep it private):

```bash
cd ~/git/mail-done-config
git add .
git commit -m "Update VIP list"
git push origin main
```

### 4. Document Custom Configs

Add a README to your private config repo:

```markdown
# mail-done-config

Private configuration for mail-done.

## Files

- `vip_senders.yaml` - VIP list (updated monthly)
- `accounts.yaml` - Email accounts
- `prompts/application.py` - Custom application analysis prompt

## Setup

Set `CONFIG_DIR` environment variable:
```bash
export CONFIG_DIR=~/git/mail-done-config
```
```

### 5. Test Configuration Changes

Use dry-run mode to test config changes:

```bash
python3.11 process_inbox.py --dry-run --limit 5
```

This processes emails without making IMAP changes, so you can verify classifications.
