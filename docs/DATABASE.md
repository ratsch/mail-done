# Database Documentation

mail-done uses PostgreSQL with pgvector extension for semantic search capabilities.

## Requirements

- PostgreSQL 16 or later
- pgvector extension (for vector similarity search)
- uuid-ossp extension (for UUID generation)
- pg_trgm extension (for text search)

## Quick Setup

### Using Docker (Recommended)

The deployment includes PostgreSQL with all extensions:

```bash
docker compose -f deploy/docker-compose.yml up -d db
```

### Manual Installation

```bash
# Install PostgreSQL 16
# (varies by OS)

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make && sudo make install

# Create database
createdb email_processor

# Enable extensions
psql email_processor <<EOF
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;
EOF
```

## Connection

Set the `DATABASE_URL` environment variable:

```bash
DATABASE_URL=postgresql://postgres:password@localhost:5432/email_processor
```

## Schema

The database schema is managed by Alembic migrations.

### Run Migrations

```bash
# Apply all migrations
poetry run alembic upgrade head

# Show current revision
poetry run alembic current

# Generate new migration
poetry run alembic revision --autogenerate -m "description"
```

### Core Tables

#### `emails`

Main email storage:

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| message_id | VARCHAR | IMAP message ID |
| from_address | VARCHAR | Sender email (encrypted) |
| from_name | VARCHAR | Sender name (encrypted) |
| to_addresses | JSONB | Recipients (encrypted) |
| subject | VARCHAR | Email subject (encrypted) |
| body_text | TEXT | Plain text body (encrypted) |
| body_html | TEXT | HTML body (encrypted) |
| body_markdown | TEXT | Markdown body (encrypted) |
| date | TIMESTAMP | Email date |
| folder | VARCHAR | IMAP folder |
| imap_uid | INTEGER | IMAP UID |
| account | VARCHAR | Email account |
| has_attachments | BOOLEAN | Has attachments |
| is_seen | BOOLEAN | Read status |
| is_flagged | BOOLEAN | Flagged status |
| raw_headers | JSONB | Original headers |
| created_at | TIMESTAMP | Record created |
| updated_at | TIMESTAMP | Record updated |

#### `email_metadata`

AI classification and user metadata:

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| email_id | UUID | Foreign key to emails |
| vip_level | VARCHAR | VIP priority level |
| ai_category | VARCHAR | AI-assigned category |
| ai_confidence | FLOAT | Classification confidence |
| ai_summary | TEXT | AI-generated summary |
| ai_suggested_reply | TEXT | Suggested reply |
| needs_reply | BOOLEAN | Needs reply flag |
| awaiting_reply | BOOLEAN | Awaiting response |
| is_urgent | BOOLEAN | Urgent flag |
| intended_color | INTEGER | Apple Mail color code |
| project_tags | JSONB | User-assigned tags |
| user_notes | TEXT | User notes |

#### `email_embeddings`

Vector embeddings for semantic search:

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| email_id | UUID | Foreign key to emails |
| embedding | vector(1536) | OpenAI embedding vector |
| model | VARCHAR | Embedding model used |
| created_at | TIMESTAMP | When generated |

#### `email_classifications`

Classification history:

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| email_id | UUID | Foreign key to emails |
| classifier_type | VARCHAR | Classifier type (ai/rule/vip) |
| category | VARCHAR | Assigned category |
| confidence | FLOAT | Classification confidence |
| model | VARCHAR | Model used |
| tokens_used | INTEGER | Tokens consumed |
| cost | FLOAT | API cost |
| created_at | TIMESTAMP | Classification time |

#### `sender_history`

Sender statistics:

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| email_address | VARCHAR | Sender email (encrypted) |
| sender_name | VARCHAR | Display name |
| email_count | INTEGER | Total emails received |
| last_seen | TIMESTAMP | Most recent email |
| sender_type | VARCHAR | Computed type |
| is_vip | BOOLEAN | VIP status |
| avg_reply_time_hours | FLOAT | Average reply time |

#### `cost_tracking`

API cost tracking:

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| operation | VARCHAR | Operation type |
| model | VARCHAR | Model used |
| tokens_input | INTEGER | Input tokens |
| tokens_output | INTEGER | Output tokens |
| cost | FLOAT | Computed cost |
| created_at | TIMESTAMP | When recorded |

## Encryption

Sensitive fields are encrypted at rest using Fernet symmetric encryption.

### Encrypted Fields

- `from_address`
- `from_name`
- `to_addresses`
- `subject`
- `body_text`
- `body_html`
- `body_markdown`
- Sender email addresses

### Encryption Key

Set the `DB_ENCRYPTION_KEY` environment variable:

```bash
# Generate a Fernet key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Important**: Back up this key securely. Lost key = unrecoverable data.

### Key Rotation

See `docs/DB_ENCRYPTION_KEY_ROTATION.md` for key rotation procedures.

## Vector Search

### Creating Embeddings

Embeddings are generated using OpenAI's text-embedding-3-small model (1536 dimensions).

```python
# In email processing
embedding = await openai.embeddings.create(
    model="text-embedding-3-small",
    input=email_text
)
# Stored in email_embeddings table
```

### Similarity Search

pgvector supports multiple distance functions:

```sql
-- Cosine similarity (recommended)
SELECT * FROM email_embeddings
ORDER BY embedding <=> query_vector
LIMIT 10;

-- L2 distance
SELECT * FROM email_embeddings
ORDER BY embedding <-> query_vector
LIMIT 10;
```

### Vector Index

For performance, create an IVFFlat or HNSW index:

```sql
-- IVFFlat index (faster build, good recall)
CREATE INDEX ON email_embeddings
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- HNSW index (slower build, better recall)
CREATE INDEX ON email_embeddings
USING hnsw (embedding vector_cosine_ops);
```

## Performance

### Recommended Indexes

```sql
-- Email queries
CREATE INDEX idx_emails_folder ON emails(folder);
CREATE INDEX idx_emails_date ON emails(date DESC);
CREATE INDEX idx_emails_account ON emails(account);
CREATE INDEX idx_emails_message_id ON emails(message_id);

-- Metadata queries
CREATE INDEX idx_metadata_category ON email_metadata(ai_category);
CREATE INDEX idx_metadata_vip ON email_metadata(vip_level);
CREATE INDEX idx_metadata_needs_reply ON email_metadata(needs_reply) WHERE needs_reply = true;

-- Full-text search
CREATE INDEX idx_emails_subject_trgm ON emails USING gin (subject gin_trgm_ops);
```

### Connection Pooling

For production, use connection pooling:

```bash
# PgBouncer or similar
DATABASE_URL=postgresql://postgres:password@localhost:6432/email_processor?pool_mode=transaction
```

## Backup & Recovery

### Backup

```bash
# Full backup
pg_dump -U postgres email_processor > backup_$(date +%Y%m%d).sql

# Compressed
pg_dump -U postgres email_processor | gzip > backup_$(date +%Y%m%d).sql.gz

# Docker
docker exec mail-done-db pg_dump -U postgres email_processor > backup.sql
```

### Restore

```bash
# Full restore
psql -U postgres email_processor < backup.sql

# Docker
docker exec -i mail-done-db psql -U postgres email_processor < backup.sql
```

### Point-in-Time Recovery

For production, enable WAL archiving for point-in-time recovery.

## Monitoring

### Database Size

```sql
SELECT pg_size_pretty(pg_database_size('email_processor'));
```

### Table Sizes

```sql
SELECT
    relname as table,
    pg_size_pretty(pg_total_relation_size(relid)) as total_size
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

### Index Usage

```sql
SELECT
    indexrelname,
    idx_scan,
    idx_tup_read
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

## Troubleshooting

### "pgvector extension not found"

```sql
CREATE EXTENSION vector;
```

If not installed, install pgvector:
```bash
# Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# From source
git clone https://github.com/pgvector/pgvector.git
cd pgvector && make && sudo make install
```

### "Encryption key invalid"

1. Check `DB_ENCRYPTION_KEY` is set correctly
2. Ensure no whitespace in the key
3. Key must be valid Fernet format (base64-encoded 32 bytes)

### "Vector dimension mismatch"

Ensure embeddings use consistent dimensions:
```sql
-- Check current dimensions
SELECT vector_dims(embedding) FROM email_embeddings LIMIT 1;
```

The default is 1536 (OpenAI text-embedding-3-small).
