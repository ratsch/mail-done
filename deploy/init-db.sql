-- Initialize PostgreSQL with required extensions for mail-done
--
-- This script runs on first database creation (Docker entrypoint).
-- It only creates EXTENSIONS - tables and indexes are created by Alembic migrations.
--
-- The full database setup flow:
--   1. Docker creates database and runs this script (extensions only)
--   2. API container runs: alembic upgrade head (creates tables/indexes)
--
-- For manual setup, see: scripts/db_migrate.sh

-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable other useful extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search

-- Log success
DO $$
BEGIN
    RAISE NOTICE 'mail-done database initialized with pgvector extension';
END $$;
