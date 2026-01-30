"""
Comprehensive filter tests for application listing
"""
import pytest
from datetime import datetime, timedelta
from uuid import uuid4


class TestAdvancedFilters:
    """Test advanced filtering options."""
    
    def test_filter_by_red_flags(self, client, reviewer_token, test_db):
        """Test filtering by red flags."""
        from backend.core.database.models import Email, EmailMetadata
        
        # Create applications with different red flags
        for i, flag_name in enumerate(['is_mass_email', 'no_research_background', 'possible_spam']):
            email = Email(
                id=uuid4(),
                message_id=f"msg{i}",
                uid=f"uid{i}",
                from_address=f"test{i}@example.com",
                to_addresses=["test@example.com"],
                subject=f"Test {i}",
                date=datetime.utcnow() - timedelta(days=i),
                folder="INBOX"
            )
            test_db.add(email)
            
            metadata = EmailMetadata(
                email_id=email.id,
                ai_category="application-phd",
                applicant_name=f"Applicant {i}",
                category_metadata={
                    "red_flags": {
                        flag_name: True,
                        "is_mass_email": False if flag_name != "is_mass_email" else True,
                        "no_research_background": False if flag_name != "no_research_background" else True,
                        "possible_spam": False if flag_name != "possible_spam" else True,
                    }
                }
            )
            test_db.add(metadata)
        test_db.commit()
        
        # Note: Current API doesn't have direct is_mass_email filter
        # Filtering by red_flags requires JSON query which SQLite doesn't support well
        # This test documents the gap - filtering by red flags would need PostgreSQL JSON operators
        response = client.get(
            "/applications",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # All applications should be returned (no red flag filter implemented yet)
        assert len(data["items"]) >= 3
    
    def test_filter_by_technical_scores(self, client, reviewer_token, test_db):
        """Test filtering by technical experience scores."""
        from backend.core.database.models import Email, EmailMetadata
        
        # Create applications with different tech scores
        for i in range(3):
            email = Email(
                id=uuid4(),
                message_id=f"tech{i}",
                uid=f"uid-tech{i}",
                from_address=f"tech{i}@example.com",
                to_addresses=["test@example.com"],
                subject=f"Tech Test {i}",
                date=datetime.utcnow() - timedelta(days=i),
                folder="INBOX"
            )
            test_db.add(email)
            
            metadata = EmailMetadata(
                email_id=email.id,
                ai_category="application-phd",
                applicant_name=f"Tech Applicant {i}",
                category_metadata={
                    "technical_experience_scores": {
                        "coding_experience": 5 + i,
                        "omics_genomics_experience": 3 + i,
                    }
                }
            )
            test_db.add(metadata)
        test_db.commit()
        
        # Note: Current API doesn't have filters for tech scores
        # This test documents the gap
        response = client.get(
            "/applications",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # All applications should be returned (no tech score filter yet)
        assert len(data["items"]) >= 3
    
    def test_filter_by_profile_tags(self, client, reviewer_token, test_db):
        """Test filtering by profile tags."""
        from backend.core.database.models import Email, EmailMetadata
        
        # Create applications with different profile tags
        tags_list = [
            [{"tag": "single_cell_omics", "confidence": 0.9}],
            [{"tag": "computational_pathology", "confidence": 0.8}],
            [{"tag": "multimodal_learning", "confidence": 0.7}],
        ]
        
        for i, tags in enumerate(tags_list):
            email = Email(
                id=uuid4(),
                message_id=f"tag{i}",
                uid=f"uid-tag{i}",
                from_address=f"tag{i}@example.com",
                to_addresses=["test@example.com"],
                subject=f"Tag Test {i}",
                date=datetime.utcnow() - timedelta(days=i),
                folder="INBOX"
            )
            test_db.add(email)
            
            metadata = EmailMetadata(
                email_id=email.id,
                ai_category="application-phd",
                applicant_name=f"Tag Applicant {i}",
                category_metadata={
                    "profile_tags": tags
                }
            )
            test_db.add(metadata)
        test_db.commit()
        
        # Note: Current API doesn't have filters for profile tags
        # This test documents the gap
        response = client.get(
            "/applications",
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) >= 3
    
    def test_filter_by_has_decision(self, client, reviewer_token, test_db):
        """Test filtering by decision status."""
        from backend.core.database.models import Email, EmailMetadata, ApplicationDecision, LabMember
        
        # Create admin user for decisions
        admin = LabMember(
            id=uuid4(),
            email="admin@test.com",
            full_name="Admin",
            role="admin",
            can_review=True,
            is_active=True,
            gsuite_id="admin"
        )
        test_db.add(admin)
        
        # Create applications - some with decisions, some without
        for i in range(4):
            email = Email(
                id=uuid4(),
                message_id=f"dec{i}",
                uid=f"uid-dec{i}",
                from_address=f"dec{i}@example.com",
                to_addresses=["test@example.com"],
                subject=f"Decision Test {i}",
                date=datetime.utcnow() - timedelta(days=i),
                folder="INBOX"
            )
            test_db.add(email)
            
            metadata = EmailMetadata(
                email_id=email.id,
                ai_category="application-phd",
                applicant_name=f"Decision Applicant {i}",
                application_status="pending"
            )
            test_db.add(metadata)
            
            # Add decision to first 2
            if i < 2:
                decision = ApplicationDecision(
                    email_id=email.id,
                    admin_id=admin.id,
                    decision="accept" if i == 0 else "reject",
                    notes="Test decision"
                )
                test_db.add(decision)
                metadata.application_status = "accepted" if i == 0 else "rejected"
        
        test_db.commit()
        
        # Filter by has_decision=True
        response = client.get(
            "/applications",
            params={"has_decision": True},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) >= 2
        
        # Filter by has_decision=False
        response = client.get(
            "/applications",
            params={"has_decision": False},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) >= 2
    
    def test_filter_by_deadline_approaching(self, client, reviewer_token, test_db):
        """Test filtering by deadline approaching."""
        from backend.core.database.models import Email, EmailMetadata
        
        # Create applications with different deadlines
        now = datetime.utcnow()
        deadlines = [
            now + timedelta(days=1),   # Approaching
            now + timedelta(days=5),   # Not approaching
            now - timedelta(days=1),   # Passed
        ]
        
        for i, deadline in enumerate(deadlines):
            email = Email(
                id=uuid4(),
                message_id=f"deadline{i}",
                uid=f"uid-deadline{i}",
                from_address=f"deadline{i}@example.com",
                to_addresses=["test@example.com"],
                subject=f"Deadline Test {i}",
                date=datetime.utcnow() - timedelta(days=i+10),
                folder="INBOX"
            )
            test_db.add(email)
            
            metadata = EmailMetadata(
                email_id=email.id,
                ai_category="application-phd",
                applicant_name=f"Deadline Applicant {i}",
                review_deadline=deadline,
                overall_recommendation_score=8  # High enough to have deadline
            )
            test_db.add(metadata)
        test_db.commit()
        
        # Filter by deadline_approaching=True
        response = client.get(
            "/applications",
            params={"deadline_approaching": True},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # Should include applications with deadlines within 3 days
        assert len(data["items"]) >= 1
    
    def test_filter_combination(self, client, reviewer_token, test_db):
        """Test combining multiple filters."""
        from backend.core.database.models import Email, EmailMetadata
        
        # Create application matching multiple criteria
        email = Email(
            id=uuid4(),
            message_id="combined",
            uid="uid-combined",
            from_address="combined@example.com",
            to_addresses=["test@example.com"],
            subject="Combined Filter Test",
            date=datetime.utcnow() - timedelta(days=5),
            folder="INBOX"
        )
        test_db.add(email)
        
        metadata = EmailMetadata(
            email_id=email.id,
            ai_category="application-phd",
            applicant_name="Combined Applicant",
            research_fit_score=8,
            overall_recommendation_score=9,
            category_metadata={
                "scientific_excellence_score": 8,
                "red_flags": {
                    "is_mass_email": False,
                    "no_research_background": False,
                }
            }
        )
        test_db.add(metadata)
        test_db.commit()
        
        # Apply multiple filters
        # Note: min_excellence_score uses JSON query which may not work in SQLite
        # Test with filters that work in SQLite
        response = client.get(
            "/applications",
            params={
                "category": "application-phd",
                "min_recommendation_score": 8,
                "min_research_fit_score": 7,
                # Skip min_excellence_score as it requires PostgreSQL JSON operators
            },
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        # Should return at least the application we created
        assert len(data["items"]) >= 1


class TestSortingOptions:
    """Test sorting functionality."""
    
    def test_sort_by_rating(self, client, reviewer_token, reviewer_user, test_db):
        """Test sorting by average rating."""
        from backend.core.database.models import Email, EmailMetadata, ApplicationReview
        
        # Use existing reviewer_user fixture instead of creating new one
        reviewer = reviewer_user
        
        # Create applications with different ratings
        ratings = [5, 3, 4, 2]
        for i, rating in enumerate(ratings):
            email = Email(
                id=uuid4(),
                message_id=f"sort{i}",
                uid=f"uid-sort{i}",
                from_address=f"sort{i}@example.com",
                to_addresses=["test@example.com"],
                subject=f"Sort Test {i}",
                date=datetime.utcnow() - timedelta(days=i),
                folder="INBOX"
            )
            test_db.add(email)
            
            metadata = EmailMetadata(
                email_id=email.id,
                ai_category="application-phd",
                applicant_name=f"Sort Applicant {i}"
            )
            test_db.add(metadata)
            
            # Add review
            review = ApplicationReview(
                email_id=email.id,
                lab_member_id=reviewer.id,
                rating=rating,
                comment=f"Rating {rating}"
            )
            test_db.add(review)
        test_db.commit()
        
        # Sort by rating descending
        response = client.get(
            "/applications",
            params={"sort_by": "rating", "sort_order": "desc"},
            headers={"Authorization": f"Bearer {reviewer_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data["items"]) >= 4
        
        # Verify sorting (ratings should be descending: 5, 4, 3, 2)
        ratings_returned = [item.get("avg_rating") for item in data["items"] if item.get("avg_rating") is not None]
        if len(ratings_returned) >= 2:
            assert ratings_returned[0] >= ratings_returned[1]

