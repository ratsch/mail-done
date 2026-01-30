-- Initialize PostgreSQL with required extensions for mail-done
-- This script runs on first database creation

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
