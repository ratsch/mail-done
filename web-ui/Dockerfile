FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .
COPY static/ ./static/
COPY shared_auth/ ./shared_auth/

# Create non-root user for security
RUN useradd -m -u 1000 webui && chown -R webui:webui /app
USER webui

# Environment defaults (can be overridden)
ENV ENVIRONMENT=production
ENV ENABLE_MACOS_FEATURES=false
ENV RATE_LIMIT_RPM=60
ENV RATE_LIMIT_AUTH_PM=5
ENV RATE_LIMIT_LOCKOUT_SECONDS=300

# Expose port
EXPOSE 8080

# Health check using Python's urllib (no extra deps needed)
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=2)"

# Run the application
CMD ["python", "app.py"]
