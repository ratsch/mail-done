# Use Python 3.11 as required by project
# This Dockerfile includes Tailscale for direct network exposure
FROM python:3.11-slim

# Force cache bust for deployment
ARG BUILDTIME=unknown
RUN echo "Build time: $BUILDTIME"

# Install build dependencies and Tailscale
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    postgresql-client \
    libpq-dev \
    curl \
    && curl -fsSL https://tailscale.com/install.sh | sh \
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

# Create startup script
COPY start-ts.sh /start.sh
RUN chmod +x /start.sh

# Expose port
EXPOSE 8000

CMD ["/start.sh"]
