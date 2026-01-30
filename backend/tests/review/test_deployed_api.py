"""
Tests against deployed API.

These tests require:
1. API to be deployed
2. Valid test credentials
3. Test database with seed data

Run with: API_URL=https://your-api-url pytest backend/tests/review/test_deployed_api.py
"""
import pytest
import os
import httpx
from typing import Optional


API_URL = os.getenv("API_URL")
if not API_URL:
    pytest.skip("API_URL environment variable not set", allow_module_level=True)
TEST_EMAIL = os.getenv("TEST_EMAIL")  # GSuite email for testing
TEST_TOKEN = os.getenv("TEST_TOKEN")  # Pre-generated JWT token for testing


@pytest.fixture
def api_client():
    """Create HTTP client for deployed API."""
    return httpx.AsyncClient(base_url=API_URL, timeout=30.0)


@pytest.mark.asyncio
@pytest.mark.skipif(not TEST_TOKEN, reason="TEST_TOKEN not set")
class TestDeployedAPI:
    """Test deployed API endpoints."""
    
    async def test_health_check(self, api_client):
        """Test health check endpoint."""
        response = await api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    async def test_auth_me(self, api_client):
        """Test GET /auth/me with deployed token."""
        if not TEST_TOKEN:
            pytest.skip("TEST_TOKEN not set")
        
        response = await api_client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "email" in data
        assert "role" in data
    
    async def test_list_applications(self, api_client):
        """Test GET /applications."""
        if not TEST_TOKEN:
            pytest.skip("TEST_TOKEN not set")
        
        response = await api_client.get(
            "/applications",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params={"page": 1, "limit": 10}
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
    
    async def test_stats_overview(self, api_client):
        """Test GET /stats/overview."""
        if not TEST_TOKEN:
            pytest.skip("TEST_TOKEN not set")
        
        response = await api_client.get(
            "/stats/overview",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_applications" in data


@pytest.mark.asyncio
@pytest.mark.skipif(not TEST_EMAIL, reason="TEST_EMAIL not set")
class TestDeployedOAuthFlow:
    """Test OAuth flow against deployed API."""
    
    async def test_oauth_init(self, api_client):
        """Test OAuth initiation."""
        response = await api_client.get("/auth/google/init", follow_redirects=False)
        assert response.status_code == 307
        assert "accounts.google.com" in response.headers.get("Location", "")


@pytest.mark.asyncio
class TestDeployedAPIPerformance:
    """Test API performance."""
    
    async def test_response_time(self, api_client):
        """Test that API responds within acceptable time."""
        import time
        
        start = time.time()
        response = await api_client.get("/health")
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 2.0  # Should respond within 2 seconds
    
    async def test_list_applications_performance(self, api_client):
        """Test list applications performance."""
        if not TEST_TOKEN:
            pytest.skip("TEST_TOKEN not set")
        
        import time
        
        start = time.time()
        response = await api_client.get(
            "/applications",
            headers={"Authorization": f"Bearer {TEST_TOKEN}"},
            params={"page": 1, "limit": 20}
        )
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0  # Should respond within 1 second

