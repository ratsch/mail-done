#!/bin/bash
# Quick start script for Mail-Done Web UI

echo "🚀 Starting Mail-Done Web UI..."
echo ""

# Detect docker compose command (new vs old syntax)
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
elif docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    echo "❌ Docker Compose not found!"
    echo "   Please install Docker Desktop or Docker Compose"
    exit 1
fi

echo "Using: $DOCKER_COMPOSE"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found!"
    echo "   Creating from env-template..."

    if [ -f "env-template" ]; then
        cp env-template .env
        echo "   ✅ Created .env file"
        echo ""
        echo "   ⚠️  IMPORTANT: Edit .env and set your BACKEND_API_URL!"
        echo "   Then run this script again."
        echo ""
        exit 1
    else
        echo "   ❌ env-template not found. Please create .env manually."
        exit 1
    fi
fi

# Check if Backend API URL is set to default
if grep -q "BACKEND_API_URL=http://localhost:8000" .env; then
    echo "ℹ️  Using default BACKEND_API_URL: http://localhost:8000"
    echo "   Edit .env to change the backend URL if needed."
    echo ""
fi

echo "📦 Building Docker image..."
$DOCKER_COMPOSE build

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo ""
echo "🐳 Starting Docker container..."
$DOCKER_COMPOSE up -d

if [ $? -ne 0 ]; then
    echo "❌ Failed to start container"
    exit 1
fi

echo ""
echo "✅ Web UI is starting!"
echo ""
echo "   🌐 Access at: http://localhost:8080"
echo "   📊 Health check: http://localhost:8080/health"
echo ""
echo "   View logs: $DOCKER_COMPOSE logs -f"
echo "   Stop: $DOCKER_COMPOSE down"
echo ""
