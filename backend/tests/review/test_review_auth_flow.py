"""
Authentication flow tests
"""
import pytest
from unittest.mock import patch, Mock
from datetime import datetime, timedelta


class TestOAuthFlow:
    """Test OAuth authentication flow."""
    
    def test_oauth_init(self, client):
        """Test OAuth initiation endpoint."""
        # Mock the module-level variables
        with patch('backend.api.routes.review_auth.GOOGLE_CLIENT_ID', 'test_client_id'), \
             patch('backend.api.routes.review_auth.GOOGLE_REDIRECT_URI', 'http://localhost:3000/callback'):
            response = client.get("/auth/google/init", follow_redirects=False)
            assert response.status_code == 307  # Redirect
            assert "accounts.google.com" in response.headers["Location"]
            assert "state=" in response.headers["Location"]
    
    def test_oauth_callback_success(self, client, test_db, mock_google_oauth):
        """Test successful OAuth callback."""
        from backend.api.review_auth import generate_oauth_state
        
        state = generate_oauth_state()
        
        with patch('backend.api.routes.review_auth.GOOGLE_CLIENT_ID', 'test_client_id'), \
             patch('backend.api.routes.review_auth.GOOGLE_CLIENT_SECRET', 'test_secret'), \
             patch('backend.api.routes.review_auth.GOOGLE_REDIRECT_URI', 'http://localhost:3000/callback'):
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = {
                    "id_token": "mock_id_token",
                    "access_token": "mock_access_token"
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response
                
                response = client.get(
                    "/auth/google/callback",
                    params={"code": "test_code", "state": state}
                )
                # Should create user and return token
                assert response.status_code == 200
                data = response.json()
                assert "access_token" in data
                assert "user" in data
    
    def test_oauth_callback_invalid_state(self, client):
        """Test OAuth callback with invalid state."""
        response = client.get(
            "/auth/google/callback",
            params={"code": "test_code", "state": "invalid_state"}
        )
        assert response.status_code == 400
    
    def test_oauth_callback_account_locked(self, client, test_db, reviewer_user, mock_google_oauth):
        """Test OAuth callback with locked account."""
        from backend.api.review_auth import generate_oauth_state, handle_failed_login
        
        # Lock the account
        for _ in range(5):
            handle_failed_login(reviewer_user.email, test_db)
        test_db.refresh(reviewer_user)
        assert reviewer_user.locked_until is not None
        
        state = generate_oauth_state()
        
        with patch('backend.api.routes.review_auth.GOOGLE_CLIENT_ID', 'test_client_id'), \
             patch('backend.api.routes.review_auth.GOOGLE_CLIENT_SECRET', 'test_secret'), \
             patch('backend.api.routes.review_auth.GOOGLE_REDIRECT_URI', 'http://localhost:3000/callback'):
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = {
                    "id_token": "mock_id_token"
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response
                
                # Mock verify_google_token to return locked user's email
                with patch("backend.api.routes.review_auth.verify_google_token") as mock_verify:
                    mock_verify.return_value = {
                        "sub": reviewer_user.gsuite_id,
                        "email": reviewer_user.email,
                        "name": reviewer_user.full_name
                    }
                    
                    response = client.get(
                        "/auth/google/callback",
                        params={"code": "test_code", "state": state}
                    )
                    assert response.status_code == 403
    
    def test_oauth_callback_suspended_account(self, client, test_db, reviewer_user, mock_google_oauth):
        """Test OAuth callback with suspended account."""
        from backend.api.review_auth import generate_oauth_state
        
        # Suspend account
        reviewer_user.is_active = False
        test_db.commit()
        
        state = generate_oauth_state()
        
        with patch('backend.api.routes.review_auth.GOOGLE_CLIENT_ID', 'test_client_id'), \
             patch('backend.api.routes.review_auth.GOOGLE_CLIENT_SECRET', 'test_secret'), \
             patch('backend.api.routes.review_auth.GOOGLE_REDIRECT_URI', 'http://localhost:3000/callback'):
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_response = Mock()
                mock_response.json.return_value = {
                    "id_token": "mock_id_token"
                }
                mock_response.raise_for_status = Mock()
                mock_post.return_value = mock_response
                
                with patch("backend.api.routes.review_auth.verify_google_token") as mock_verify:
                    mock_verify.return_value = {
                        "sub": reviewer_user.gsuite_id,
                        "email": reviewer_user.email,
                        "name": reviewer_user.full_name
                    }
                    
                    response = client.get(
                        "/auth/google/callback",
                        params={"code": "test_code", "state": state}
                    )
                    assert response.status_code == 403


class TestAuthorization:
    """Test authorization checks."""
    
    def test_reviewer_endpoint_requires_permission(self, client, regular_token):
        """Test that reviewer endpoints require can_review permission."""
        response = client.get(
            "/applications",
            headers={"Authorization": f"Bearer {regular_token}"}
        )
        assert response.status_code == 403
    
    def test_admin_endpoint_requires_admin_role(self, client, reviewer_token):
        """Test that admin endpoints require admin role."""
        response = client.get(
            "/admin/users",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 403
    
    def test_permission_change_takes_effect(self, client, admin_token, reviewer_user, reviewer_token, test_db):
        """Test that permission changes take effect immediately."""
        # User can access initially
        response = client.get(
            "/applications",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        
        # Revoke permission
        response = client.patch(
            f"/admin/users/{reviewer_user.id}",
            json={"can_review": False},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        
        # User can no longer access
        response = client.get(
            "/applications",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 403

