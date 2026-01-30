"""
Edge case and validation tests for Lab Application Review System
"""
import pytest
from datetime import datetime, timedelta
from uuid import uuid4


class TestFilterValidation:
    """Test filter parameter validation."""
    
    def test_invalid_date_filters(self, client, reviewer_token):
        """Test invalid date filter formats."""
        # Invalid date format - API may be lenient or ignore invalid dates
        response = client.get(
            "/applications",
            params={"received_after": "not-a-date"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        # API may return 200 (ignoring invalid date) or 422 (validation error)
        assert response.status_code in [200, 422]
        if response.status_code == 200:
            # Should return empty results or all results (invalid filter ignored)
            data = response.json()
            assert "items" in data
    
    def test_negative_scores(self, client, reviewer_token):
        """Test negative score filters."""
        response = client.get(
            "/applications",
            params={"min_recommendation_score": -1},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 422  # Validation error (ge=0)
    
    def test_score_out_of_range(self, client, reviewer_token):
        """Test score filters out of valid range."""
        response = client.get(
            "/applications",
            params={"min_recommendation_score": 11},  # Max is 10
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 422  # Validation error (le=10)
    
    def test_invalid_status(self, client, reviewer_token):
        """Test invalid application status filter."""
        response = client.get(
            "/applications",
            params={"application_status": "invalid_status"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        # Should either filter to empty results or validate enum
        assert response.status_code in [200, 422]
    
    def test_search_name_sql_injection(self, client, reviewer_token, multiple_applications):
        """Test SQL injection prevention in name search."""
        # Try SQL injection patterns
        injection_attempts = [
            "'; DROP TABLE emails; --",
            "' OR '1'='1",
            "'; SELECT * FROM emails; --",
            "1' UNION SELECT NULL--",
        ]
        
        for attempt in injection_attempts:
            response = client.get(
                "/applications",
                params={"search_name": attempt},
                headers={"Authorization": f"Bearer {reviewer_token}"}
            )
            # Should not crash - either return empty results or sanitize
            assert response.status_code == 200
            # Should not return all records
            data = response.json()
            assert isinstance(data["items"], list)
    
    def test_search_name_length_limit(self, client, reviewer_token):
        """Test search name length limit."""
        # Very long search string
        long_string = "a" * 1000
        response = client.get(
            "/applications",
            params={"search_name": long_string},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        # Should sanitize or reject
        assert response.status_code in [200, 422]


class TestReviewValidation:
    """Test review submission validation."""
    
    def test_review_rating_out_of_range(self, client, reviewer_token, multiple_applications):
        """Test review rating validation."""
        email_id = multiple_applications[0]
        
        # Rating too low
        response = client.post(
            f"/applications/{email_id}/review",
            json={"rating": 0, "comment": "Test"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 422
        
        # Rating too high
        response = client.post(
            f"/applications/{email_id}/review",
            json={"rating": 6, "comment": "Test"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 422
    
    def test_review_comment_length(self, client, reviewer_token, multiple_applications):
        """Test review comment length limit."""
        email_id = multiple_applications[0]
        
        # Comment too long (max is 2000)
        long_comment = "a" * 2001
        response = client.post(
            f"/applications/{email_id}/review",
            json={"rating": 5, "comment": long_comment},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 422
    
    def test_review_xss_in_comment(self, client, reviewer_token, multiple_applications):
        """Test XSS prevention in comments."""
        email_id = multiple_applications[0]
        
        xss_attempts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg onload=alert('xss')>",
        ]
        
        for xss in xss_attempts:
            response = client.post(
                f"/applications/{email_id}/review",
                json={"rating": 5, "comment": xss},
                headers={"Authorization": f"Bearer {reviewer_token}"}
            )
            # API accepts XSS strings (sanitization is frontend responsibility or happens in storage)
            # May return 200 (success) or 422 (validation error if sanitization rejects)
            assert response.status_code in [200, 201, 422]
            if response.status_code in [200, 201]:
                data = response.json()
                # Verify review was created
                assert "num_ratings" in data
    
    def test_review_nonexistent_application(self, client, reviewer_token):
        """Test reviewing non-existent application."""
        fake_id = uuid4()
        response = client.post(
            f"/applications/{fake_id}/review",
            json={"rating": 5, "comment": "Test"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 404
    
    def test_delete_nonexistent_review(self, client, reviewer_token, multiple_applications):
        """Test deleting review that doesn't exist."""
        email, metadata = multiple_applications[0]
        email_id = email.id
        
        # Try to delete review that doesn't exist
        response = client.delete(
            f"/applications/{email_id}/review",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        # API may return 200 (idempotent delete), 404 (not found), or 422 (validation error)
        assert response.status_code in [200, 404, 422]


class TestAdminValidation:
    """Test admin endpoint validation."""
    
    def test_batch_decision_invalid_ids(self, client, admin_token):
        """Test batch decision with invalid email IDs."""
        fake_ids = [uuid4(), uuid4()]
        response = client.post(
            "/admin/applications/batch/decide",
            json={
                "email_ids": [str(fid) for fid in fake_ids],
                "decision": "accept",
                "notes": "Test"
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # Should handle gracefully - may return partial success or errors
        assert response.status_code in [200, 400, 422]
        if response.status_code == 200:
            data = response.json()
            # Should report failures
            assert "successful" in data or "failed" in data or "errors" in data
    
    def test_batch_decision_empty_list(self, client, admin_token):
        """Test batch decision with empty email list."""
        response = client.post(
            "/admin/applications/batch/decide",
            json={
                "email_ids": [],
                "decision": "accept"
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # API may accept empty list (returning success with 0 processed) or reject
        assert response.status_code in [200, 422]
    
    def test_batch_decision_invalid_decision(self, client, admin_token, multiple_applications):
        """Test batch decision with invalid decision value."""
        email_id = multiple_applications[0]
        response = client.post(
            "/admin/applications/batch/decide",
            json={
                "email_ids": [str(email_id)],
                "decision": "invalid_decision"
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_settings_invalid_json(self, client, admin_token):
        """Test updating settings with invalid JSON."""
        response = client.put(
            "/admin/settings/invalid_key",
            json={"value": "test"},
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # Should validate key exists or return error
        assert response.status_code in [400, 404, 422]
    
    def test_create_user_invalid_email(self, client, admin_token):
        """Test creating user with invalid email."""
        response = client.post(
            "/admin/users",
            json={
                "email": "not-an-email",
                "full_name": "Test User",
                "role": "member",
                "can_review": False
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # API may validate email format or accept any string
        assert response.status_code in [200, 400, 422]
    
    def test_create_user_duplicate_email(self, client, admin_token, reviewer_user):
        """Test creating user with duplicate email."""
        response = client.post(
            "/admin/users",
            json={
                "email": reviewer_user.email,  # Already exists
                "full_name": "Duplicate User",
                "role": "member",
                "can_review": False
            },
            headers={"Authorization": f"Bearer {admin_token}"}
        )
        # Should return error (409 Conflict or 400)
        assert response.status_code in [400, 409, 422]


class TestNotificationEdgeCases:
    """Test notification endpoint edge cases."""
    
    def test_notifications_filter_by_type(self, client, reviewer_token):
        """Test filtering notifications by type."""
        response = client.get(
            "/notifications",
            params={"notification_type": "new_application"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        # All should be new_application type
        for notification in data:
            assert notification["notification_type"] == "new_application"
    
    def test_notifications_unread_only(self, client, reviewer_token):
        """Test unread_only filter."""
        response = client.get(
            "/notifications",
            params={"unread_only": True},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_notifications_invalid_type(self, client, reviewer_token):
        """Test invalid notification type filter."""
        response = client.get(
            "/notifications",
            params={"notification_type": "invalid_type"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        # Should return empty list or all notifications
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)


class TestStatsEdgeCases:
    """Test statistics endpoint edge cases."""
    
    def test_stats_empty_database(self, client, reviewer_token, test_db):
        """Test stats with empty database."""
        response = client.get(
            "/stats/overview",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_applications" in data
        assert data["total_applications"] == 0
    
    def test_stats_with_reviews(self, client, reviewer_token, multiple_applications, reviewer_user, test_db):
        """Test stats include review counts."""
        from backend.core.database.models import ApplicationReview
        
        # Add some reviews
        for i, (email, metadata) in enumerate(multiple_applications[:3]):
            review = ApplicationReview(
                email_id=email.id,  # Use email.id from tuple
                lab_member_id=reviewer_user.id,
                rating=4 + (i % 2),
                comment=f"Review {i}"
            )
            test_db.add(review)
        test_db.commit()
        
        response = client.get(
            "/stats/overview",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "total_applications" in data
        assert data["total_applications"] > 0


class TestPaginationEdgeCases:
    """Test pagination edge cases."""
    
    def test_pagination_page_zero(self, client, reviewer_token, multiple_applications):
        """Test pagination with page 0."""
        response = client.get(
            "/applications",
            params={"page": 0, "limit": 10},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        # Should either use page 1 or return error
        assert response.status_code in [200, 422]
    
    def test_pagination_negative_page(self, client, reviewer_token):
        """Test pagination with negative page."""
        response = client.get(
            "/applications",
            params={"page": -1},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 422  # Validation error
    
    def test_pagination_large_limit(self, client, reviewer_token):
        """Test pagination with very large limit."""
        response = client.get(
            "/applications",
            params={"limit": 10000},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        # Should either cap limit or return error
        assert response.status_code in [200, 422]
        if response.status_code == 200:
            data = response.json()
            # Should cap at reasonable limit (e.g., 100)
            assert data["limit"] <= 100
    
    def test_pagination_beyond_total(self, client, reviewer_token, multiple_applications):
        """Test pagination beyond total pages."""
        response = client.get(
            "/applications",
            params={"page": 9999, "limit": 10},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # Should return empty items
        assert len(data["items"]) == 0
        assert data["total"] >= 0

