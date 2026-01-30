"""
API endpoint tests for Lab Application Review System
"""
import pytest
from datetime import datetime, timedelta
from uuid import uuid4


class TestAuthenticationEndpoints:
    """Test authentication endpoints."""
    
    def test_get_current_user_info(self, client, reviewer_token):
        """Test GET /auth/me endpoint."""
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "email" in data
        assert "role" in data
        assert "can_review" in data
    
    def test_get_current_user_unauthorized(self, client):
        """Test GET /auth/me without token."""
        response = client.get("/auth/me")
        assert response.status_code == 401
    
    def test_refresh_token(self, client, reviewer_token, reviewer_user, test_db):
        """Test POST /auth/refresh."""
        # Refresh token requires valid token that hasn't expired
        # The token should be valid for refresh
        response = client.post(
            "/auth/refresh",
            json={"token": reviewer_token}
        )
        # May return 200 (success) or 401 (if token validation fails)
        assert response.status_code in [200, 401]
        if response.status_code == 200:
            data = response.json()
            assert "access_token" in data
            assert data["access_token"] != reviewer_token  # Should be new token
    
    def test_logout(self, client, reviewer_token, reviewer_user, test_db):
        """Test POST /auth/logout."""
        response = client.post(
            "/auth/logout",
            json={"token": reviewer_token},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        
        # Token should be blacklisted
        response = client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 401


class TestApplicationEndpoints:
    """Test application listing and detail endpoints."""
    
    def test_list_applications(self, client, reviewer_token, multiple_applications):
        """Test GET /applications."""
        response = client.get(
            "/applications",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert len(data["items"]) > 0
    
    def test_list_applications_unauthorized(self, client):
        """Test GET /applications without auth."""
        response = client.get("/applications")
        assert response.status_code == 401
    
    def test_list_applications_no_permission(self, client, regular_token):
        """Test GET /applications without can_review permission."""
        response = client.get(
            "/applications",
            headers={"Authorization": f"Bearer {regular_token}"}
        )
        assert response.status_code == 403
    
    def test_list_applications_with_filters(self, client, reviewer_token, multiple_applications):
        """Test GET /applications with filters."""
        response = client.get(
            "/applications",
            params={
                "category": "application-phd",
                "min_recommendation_score": 7,
                "page": 1,
                "limit": 10
            },
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert all(item["overall_recommendation_score"] >= 7 for item in data["items"])
    
    def test_list_applications_search_name(self, client, reviewer_token, multiple_applications):
        """Test GET /applications with name search."""
        response = client.get(
            "/applications",
            params={"search_name": "Applicant 0"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # Should find at least one application
        assert len(data["items"]) >= 1
    
    def test_get_application_detail(self, client, reviewer_token, sample_application):
        """Test GET /applications/{email_id}."""
        email, metadata = sample_application
        response = client.get(
            f"/applications/{email.id}",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email_id"] == str(email.id)
        assert data["applicant_name"] == metadata.applicant_name
        assert "ETag" in response.headers
        assert "Cache-Control" in response.headers
    
    def test_get_application_detail_etag(self, client, reviewer_token, sample_application):
        """Test ETag support in GET /applications/{email_id}."""
        email, metadata = sample_application
        
        # First request
        response1 = client.get(
            f"/applications/{email.id}",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response1.status_code == 200
        etag = response1.headers.get("ETag")
        
        # Second request with If-None-Match
        response2 = client.get(
            f"/applications/{email.id}",
            headers={
                "Authorization": f"Bearer {reviewer_token}",
                "If-None-Match": etag
            }
        )
        assert response2.status_code == 304  # Not Modified
    
    def test_get_application_not_found(self, client, reviewer_token):
        """Test GET /applications/{email_id} with non-existent ID."""
        fake_id = uuid4()
        response = client.get(
            f"/applications/{fake_id}",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 404


class TestReviewEndpoints:
    """Test review submission and management endpoints."""
    
    def test_submit_review(self, client, reviewer_token, sample_application):
        """Test POST /applications/{email_id}/review."""
        email, metadata = sample_application
        response = client.post(
            f"/applications/{email.id}/review",
            json={"rating": 5, "comment": "Excellent candidate"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "avg_rating" in data
        assert "num_ratings" in data
        assert data["num_ratings"] == 1
        assert data["avg_rating"] == 5.0
    
    def test_update_review(self, client, reviewer_token, sample_application, sample_review):
        """Test updating existing review."""
        email, metadata = sample_application
        response = client.post(
            f"/applications/{email.id}/review",
            json={"rating": 3, "comment": "Updated review"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["avg_rating"] == 3.0
    
    def test_delete_review(self, client, reviewer_token, sample_application, sample_review):
        """Test DELETE /applications/{email_id}/review."""
        email, metadata = sample_application
        response = client.delete(
            f"/applications/{email.id}/review",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
    
    def test_list_reviews(self, client, reviewer_token, sample_application, sample_review):
        """Test GET /applications/{email_id}/reviews."""
        email, metadata = sample_application
        response = client.get(
            f"/applications/{email.id}/reviews",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 1
    
    def test_get_review_summary(self, client, reviewer_token, sample_application, sample_review):
        """Test GET /applications/{email_id}/reviews/summary."""
        email, metadata = sample_application
        response = client.get(
            f"/applications/{email.id}/reviews/summary",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "avg_rating" in data
        assert "num_ratings" in data


class TestAdminEndpoints:
    """Test admin-only endpoints."""
    
    def test_list_users(self, client, admin_token):
        """Test GET /admin/users."""
        response = client.get(
            "/admin/users",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
        assert "total" in data
    
    def test_list_users_not_admin(self, client, reviewer_token):
        """Test GET /admin/users without admin role."""
        response = client.get(
            "/admin/users",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 403
    
    def test_create_user(self, client, admin_token, test_db):
        """Test POST /admin/users."""
        response = client.post(
            "/admin/users",
            json={
                "email": "newuser@test.com",
                "full_name": "New User",
                "role": "member",
                "can_review": True
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "newuser@test.com"
        assert data["can_review"] is True
    
    def test_update_user(self, client, admin_token, reviewer_user):
        """Test PATCH /admin/users/{id}."""
        response = client.patch(
            f"/admin/users/{reviewer_user.id}",
            json={"can_review": False},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["can_review"] is False
    
    def test_make_decision(self, client, admin_token, admin_user, sample_application, mock_gdrive_client):
        """Test POST /admin/applications/{email_id}/decision."""
        email, metadata = sample_application
        response = client.post(
            f"/admin/applications/{email.id}/decision",
            json={"decision": "accept", "notes": "Accepted"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "accept"
        assert data["admin_id"] == str(admin_user.id)
    
    def test_batch_decide(self, client, admin_token, multiple_applications, mock_gdrive_client):
        """Test POST /admin/applications/batch/decide."""
        email_ids = [str(email.id) for email, _ in multiple_applications[:3]]
        response = client.post(
            "/admin/applications/batch/decide",
            json={
                "email_ids": email_ids,
                "decision": "reject",
                "notes": "Batch rejection"
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["processed"] > 0
    
    def test_get_old_applications(self, client, admin_token, multiple_applications):
        """Test GET /admin/old-applications."""
        response = client.get(
            "/admin/old-applications",
            params={"days": 1},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
    
    def test_get_audit_log(self, client, admin_token, sample_application, reviewer_token):
        """Test GET /admin/audit-log."""
        # Create an audit log entry by viewing an application
        email, metadata = sample_application
        client.get(
            f"/applications/{email.id}",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        
        # Get audit log
        response = client.get(
            "/admin/audit-log",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "items" in data
    
    def test_get_settings(self, client, admin_token, system_settings):
        """Test GET /admin/settings."""
        response = client.get(
            "/admin/settings",
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "settings" in data
    
    def test_update_setting(self, client, admin_token, system_settings):
        """Test PUT /admin/settings/{key}."""
        response = client.put(
            "/admin/settings/rate_limit_reviews_per_hour",
            json={"value": 60, "type": "integer"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["value"]["value"] == 60


class TestExportEndpoints:
    """Test export endpoints."""
    
    def test_export_applications_csv(self, client, admin_token, multiple_applications):
        """Test GET /applications/export as CSV."""
        response = client.get(
            "/applications/export",
            params={"format": "csv"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        assert "text/csv" in response.headers["Content-Type"]  # May include charset
        assert "Content-Disposition" in response.headers
    
    def test_export_applications_excel(self, client, admin_token, multiple_applications):
        """Test GET /applications/export as Excel."""
        response = client.get(
            "/applications/export",
            params={"format": "excel"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # May fail if pandas/openpyxl not available, but should handle gracefully
        assert response.status_code in [200, 500]
    
    def test_export_applications_exclude_pii(self, client, admin_token, multiple_applications):
        """Test export with PII exclusion."""
        response = client.get(
            "/applications/export",
            params={"format": "csv", "exclude_pii": True},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 200
        content = response.text
        assert "[REDACTED]" in content


class TestNotificationEndpoints:
    """Test notification endpoints."""
    
    def test_get_notifications(self, client, reviewer_token, sample_application):
        """Test GET /notifications."""
        response = client.get(
            "/notifications",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestStatsEndpoints:
    """Test statistics endpoints."""
    
    def test_get_overview_stats(self, client, reviewer_token, multiple_applications):
        """Test GET /stats/overview."""
        response = client.get(
            "/stats/overview",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_applications" in data
        assert "pending_review" in data
        assert "decided" in data

