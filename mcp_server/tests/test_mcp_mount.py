"""Tests for MCP SSE mount — auth and route matching."""
import os
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from starlette.testclient import TestClient

# Set required env vars before import
os.environ.setdefault("MCP_API_KEY", "test-admin-key-12345")
os.environ.setdefault("MCP_ALLOWED_ACCOUNT", "personal,work")
os.environ.setdefault("MCP_REMOTE_KEYS", "test-assistant-key:assistant:work")
os.environ.setdefault("BACKEND_API_KEY", "dummy")


@pytest.fixture
def app():
    """Create the MCP SSE Starlette app for testing."""
    # Patch create_server to avoid needing the full MCP server
    mock_server = MagicMock()
    mock_server.create_initialization_options.return_value = {}

    with patch("mcp_server.server.create_server", return_value=mock_server):
        from backend.api.mcp_mount import create_mcp_sse_app, MCP_API_KEYS
        MCP_API_KEYS.clear()  # Reset between tests
        starlette_app = create_mcp_sse_app()
        assert starlette_app is not None
        yield starlette_app, MCP_API_KEYS


class TestAuthentication:
    """Test Bearer token authentication on both SSE and messages routes."""

    def test_sse_without_auth_returns_401(self, app):
        starlette_app, _ = app
        client = TestClient(starlette_app)
        response = client.get("/sse")
        assert response.status_code == 401
        assert response.json() == {"error": "Unauthorized"}

    def test_sse_with_wrong_token_returns_401(self, app):
        starlette_app, _ = app
        client = TestClient(starlette_app)
        response = client.get("/sse", headers={"Authorization": "Bearer wrong-key"})
        assert response.status_code == 401

    def test_messages_without_auth_returns_401(self, app):
        """Critical: /messages/ must be authenticated (was previously open)."""
        starlette_app, _ = app
        client = TestClient(starlette_app)
        response = client.post("/messages/", json={})
        assert response.status_code == 401
        assert response.json() == {"error": "Unauthorized"}

    def test_messages_with_session_id_without_auth_returns_401(self, app):
        """Test /messages/?session_id=xxx without auth."""
        starlette_app, _ = app
        client = TestClient(starlette_app)
        response = client.post("/messages/?session_id=abc123", json={})
        assert response.status_code == 401

    def test_messages_with_path_without_auth_returns_401(self, app):
        """Test /messages/some/path without auth."""
        starlette_app, _ = app
        client = TestClient(starlette_app)
        response = client.post("/messages/some/path", json={})
        assert response.status_code == 401


class TestAPIKeyLoading:
    """Test that API keys are loaded correctly from environment."""

    def test_admin_key_loaded(self, app):
        _, keys = app
        assert "test-admin-key-12345" in keys
        assert keys["test-admin-key-12345"]["name"] == "admin"
        assert "personal" in keys["test-admin-key-12345"]["accounts"]
        assert "work" in keys["test-admin-key-12345"]["accounts"]

    def test_assistant_key_loaded(self, app):
        _, keys = app
        assert "test-assistant-key" in keys
        assert keys["test-assistant-key"]["name"] == "assistant"
        assert keys["test-assistant-key"]["accounts"] == "work"

    def test_no_personal_for_assistant(self, app):
        _, keys = app
        assert "personal" not in keys["test-assistant-key"]["accounts"]


class TestRouteMatching:
    """Test that /messages/ route correctly matches various URL patterns."""

    def test_messages_root_matches(self, app):
        """POST /messages/ should be routed (not 404) and authenticated (not 401)."""
        starlette_app, _ = app
        client = TestClient(starlette_app, raise_server_exceptions=False)
        response = client.post(
            "/messages/",
            headers={"Authorization": "Bearer test-admin-key-12345"},
            json={}
        )
        # Route matched (not 404) and auth passed (not 401).
        # May return 500 because no real SSE session exists — that's expected.
        assert response.status_code != 404
        assert response.status_code != 401

    def test_messages_with_query_string_matches(self, app):
        """POST /messages/?session_id=xxx should be routed."""
        starlette_app, _ = app
        client = TestClient(starlette_app, raise_server_exceptions=False)
        response = client.post(
            "/messages/?session_id=abc123def",
            headers={"Authorization": "Bearer test-admin-key-12345"},
            json={}
        )
        assert response.status_code != 404
        assert response.status_code != 401

    def test_messages_with_subpath_matches(self, app):
        """POST /messages/sub/path should be routed."""
        starlette_app, _ = app
        client = TestClient(starlette_app, raise_server_exceptions=False)
        response = client.post(
            "/messages/sub/path",
            headers={"Authorization": "Bearer test-admin-key-12345"},
            json={}
        )
        assert response.status_code != 404
        assert response.status_code != 401

    def test_unknown_route_returns_404_or_405(self, app):
        """GET /nonexistent should not match."""
        starlette_app, _ = app
        client = TestClient(starlette_app)
        response = client.get("/nonexistent")
        assert response.status_code in (404, 405)


class TestContextVarIsolation:
    """Test that account restriction uses contextvars correctly."""

    def test_override_function_returns_none_by_default(self):
        from backend.api.mcp_mount import get_mcp_allowed_account_override
        # Outside of any SSE handler, should return None
        assert get_mcp_allowed_account_override() is None

    def test_contextvar_set_and_reset(self):
        from backend.api.mcp_mount import _current_allowed_account, get_mcp_allowed_account_override
        # Set
        token = _current_allowed_account.set("work")
        assert get_mcp_allowed_account_override() == "work"
        # Reset
        _current_allowed_account.reset(token)
        assert get_mcp_allowed_account_override() is None

    def test_api_client_reads_contextvar(self):
        """EmailAPIClient should pick up contextvar override when available."""
        from backend.api.mcp_mount import _current_allowed_account

        token = _current_allowed_account.set("work")
        try:
            from mcp_server.api_client import _get_mcp_allowed_account
            result = _get_mcp_allowed_account()
            assert result == "work"
        finally:
            _current_allowed_account.reset(token)
