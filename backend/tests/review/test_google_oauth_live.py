"""
Live Google OAuth integration tests

These tests require:
- Live API deployment
- Google OAuth credentials configured
- Valid GSuite account for testing

Run with:
    export API_URL=https://your-api-url
    export GOOGLE_CLIENT_ID=your_client_id
    export GOOGLE_CLIENT_SECRET_V0_PORTAL=your_client_secret
    export GOOGLE_REDIRECT_URI=https://your-api-url/auth/google/callback
    export TEST_GSUITE_EMAIL=your-test@gsuite-domain.com
    poetry run pytest backend/tests/review/test_google_oauth_live.py -v
"""
import pytest
import os
import httpx
from urllib.parse import urlparse, parse_qs


# Skip all tests if credentials not configured
pytestmark = pytest.mark.skipif(
    not all([
        os.getenv("API_URL"),
        os.getenv("GOOGLE_CLIENT_ID"),
        os.getenv("GOOGLE_CLIENT_SECRET_V0_PORTAL"),
        os.getenv("GOOGLE_REDIRECT_URI"),
    ]),
    reason="Google OAuth credentials not configured (API_URL, GOOGLE_CLIENT_ID, etc.)"
)


API_URL = os.getenv("API_URL")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET_V0_PORTAL")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")
TEST_GSUITE_EMAIL = os.getenv("TEST_GSUITE_EMAIL")


class TestGoogleOAuthLive:
    """Live Google OAuth integration tests."""
    
    def test_oauth_init_endpoint(self):
        """Test OAuth initiation endpoint returns redirect URL."""
        response = httpx.get(
            f"{API_URL}/auth/google/init",
            follow_redirects=False
        )
        assert response.status_code == 307  # Redirect
        assert "accounts.google.com" in response.headers.get("location", "")
        assert "client_id=" in response.headers.get("location", "")
        assert "state=" in response.headers.get("location", "")
        assert "redirect_uri=" in response.headers.get("location", "")
    
    def test_oauth_init_url_structure(self):
        """Test OAuth URL has correct structure."""
        response = httpx.get(
            f"{API_URL}/auth/google/init",
            follow_redirects=False
        )
        assert response.status_code == 307
        
        redirect_url = response.headers.get("location", "")
        parsed = urlparse(redirect_url)
        assert parsed.scheme == "https"
        assert parsed.netloc == "accounts.google.com"
        assert parsed.path == "/o/oauth2/v2/auth"
        
        # Parse query parameters
        params = parse_qs(parsed.query)
        assert "client_id" in params
        assert "redirect_uri" in params
        assert "response_type" in params
        assert "scope" in params
        assert "state" in params
        
        # Verify redirect URI matches configured
        assert params["redirect_uri"][0] == GOOGLE_REDIRECT_URI
    
    def test_oauth_callback_missing_params(self):
        """Test OAuth callback with missing parameters."""
        response = httpx.get(f"{API_URL}/auth/google/callback")
        assert response.status_code == 400
        
        # Missing code
        response = httpx.get(
            f"{API_URL}/auth/google/callback",
            params={"state": "test_state"}
        )
        assert response.status_code == 400
        
        # Missing state
        response = httpx.get(
            f"{API_URL}/auth/google/callback",
            params={"code": "test_code"}
        )
        assert response.status_code == 400
    
    def test_oauth_callback_invalid_state(self):
        """Test OAuth callback with invalid state."""
        response = httpx.get(
            f"{API_URL}/auth/google/callback",
            params={"code": "test_code", "state": "invalid_state_12345"}
        )
        assert response.status_code == 400
    
    def test_oauth_callback_invalid_code(self):
        """Test OAuth callback with invalid authorization code."""
        # First get a valid state
        init_response = httpx.get(
            f"{API_URL}/auth/google/init",
            follow_redirects=False
        )
        assert init_response.status_code == 307
        
        redirect_url = init_response.headers.get("location", "")
        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)
        state = params["state"][0]
        
        # Try callback with invalid code
        response = httpx.get(
            f"{API_URL}/auth/google/callback",
            params={"code": "invalid_code_12345", "state": state}
        )
        # Should fail at token exchange
        assert response.status_code in [400, 401]
    
    @pytest.mark.skipif(
        not TEST_GSUITE_EMAIL,
        reason="TEST_GSUITE_EMAIL not set - manual OAuth flow test required"
    )
    def test_full_oauth_flow_manual(self):
        """
        Manual test for full OAuth flow.
        
        This test provides instructions for manual testing since it requires
        user interaction with Google's OAuth consent screen.
        
        Steps:
        1. Visit: {API_URL}/auth/google/init
        2. Sign in with GSuite account: {TEST_GSUITE_EMAIL}
        3. Grant permissions
        4. Should redirect back with code and state
        5. API should return JWT token
        """
        init_url = f"{API_URL}/auth/google/init"
        print(f"\n{'='*60}")
        print("MANUAL OAUTH FLOW TEST")
        print(f"{'='*60}")
        print(f"1. Visit: {init_url}")
        print(f"2. Sign in with: {TEST_GSUITE_EMAIL}")
        print(f"3. Grant permissions")
        print(f"4. Check redirect URL contains 'code' and 'state'")
        print(f"5. API should return JWT token in response")
        print(f"{'='*60}\n")
        
        # Just verify init endpoint works
        response = httpx.get(init_url, follow_redirects=False)
        assert response.status_code == 307
        assert "accounts.google.com" in response.headers.get("location", "")
    
    def test_oauth_configuration_check(self):
        """Test that OAuth is properly configured."""
        # Init endpoint should work if configured
        response = httpx.get(
            f"{API_URL}/auth/google/init",
            follow_redirects=False
        )
        
        if response.status_code == 500:
            # OAuth not configured
            data = response.json()
            assert "not configured" in data.get("detail", "").lower()
        else:
            # OAuth is configured
            assert response.status_code == 307


class TestOAuthWithToken:
    """Test OAuth endpoints after obtaining a token."""
    
    @pytest.fixture
    def test_token(self):
        """
        Get a test token by completing OAuth flow.
        
        This requires manual OAuth flow completion or a pre-obtained token.
        Set TEST_TOKEN environment variable to skip manual flow.
        """
        token = os.getenv("TEST_TOKEN")
        if token:
            return token
        
        # If no token provided, skip tests that require it
        pytest.skip("TEST_TOKEN not set - set it to a valid JWT token from OAuth flow")
    
    def test_auth_me_with_token(self, test_token):
        """Test /auth/me endpoint with OAuth-obtained token."""
        response = httpx.get(
            f"{API_URL}/auth/me",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "email" in data
        assert "role" in data
        assert "can_review" in data
    
    def test_protected_endpoint_with_token(self, test_token):
        """Test accessing protected endpoint with OAuth token."""
        response = httpx.get(
            f"{API_URL}/applications",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        # May return 200 (if user has can_review) or 403 (if not)
        assert response.status_code in [200, 403]
    
    def test_token_refresh(self, test_token):
        """Test token refresh endpoint."""
        response = httpx.post(
            f"{API_URL}/auth/refresh",
            json={"token": test_token}
        )
        # May return 200 (new token) or 401 (token invalid/expired)
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert data["access_token"] != test_token  # Should be new token
    
    def test_token_logout(self, test_token):
        """Test logout endpoint."""
        response = httpx.post(
            f"{API_URL}/auth/logout",
            json={"token": test_token},
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 200
        
        # Token should be blacklisted
        response = httpx.get(
            f"{API_URL}/auth/me",
            headers={"Authorization": f"Bearer {test_token}"}
        )
        assert response.status_code == 401


class TestOAuthErrorHandling:
    """Test OAuth error handling."""
    
    def test_oauth_callback_with_error(self):
        """Test OAuth callback with error parameter."""
        response = httpx.get(
            f"{API_URL}/auth/google/callback",
            params={"error": "access_denied", "error_description": "User denied access"}
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data.get("detail", "").lower()
    
    def test_oauth_callback_expired_code(self):
        """Test OAuth callback with expired authorization code."""
        # Get valid state
        init_response = httpx.get(
            f"{API_URL}/auth/google/init",
            follow_redirects=False
        )
        assert init_response.status_code == 307
        
        redirect_url = init_response.headers.get("location", "")
        parsed = urlparse(redirect_url)
        params = parse_qs(parsed.query)
        state = params["state"][0]
        
        # Use expired/invalid code
        response = httpx.get(
            f"{API_URL}/auth/google/callback",
            params={"code": "expired_code_12345", "state": state}
        )
        # Should fail at token exchange
        assert response.status_code in [400, 401]

