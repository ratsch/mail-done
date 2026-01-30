# Use Python 3.11 as required by project
FROM python:3.11-slim

# Force cache bust for deployment
ARG BUILDTIME=unknown
RUN echo "Build time: $BUILDTIME"

# Install build dependencies for numpy and pgvector
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install poetry (version 2.2+)
RUN pip install --no-cache-dir poetry==2.2.1

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies (Poetry 2.2+ syntax)
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Copy application code
COPY . .

# Expose port
EXPOSE 8080

# Start application (use shell form to allow $PORT expansion)
# Run migrations at startup only if alembic.ini exists
CMD sh -c "[ -f alembic.ini ] && poetry run alembic upgrade head; poetry run uvicorn backend.api.main:app --host 0.0.0.0 --port ${PORT:-8080}"

