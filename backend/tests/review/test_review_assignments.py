"""
Comprehensive tests for Review Assignments API

Tests cover:
- Creating assignments (single, bulk, with sharing)
- Listing assignments (as assignee)
- Batch CRUD (list, get, update, delete)
- Adding to existing batch
- Declining assignments
- Deleting individual assignments
- Preview bulk assignment
- Auto-completion on review submission
- Permission enforcement
- Duplicate handling
- Edge cases
"""
import pytest
from datetime import datetime, timedelta, timezone
from uuid import uuid4, UUID

from backend.core.database.models import (
    LabMember, Email, EmailMetadata,
    AssignmentBatch, AssignmentBatchShare, ApplicationReviewAssignment,
    ApplicationReview, Base,
)
from backend.api.routes.review_assignments import complete_assignments_on_review


# ============================================================================
# Additional fixtures
# ============================================================================

@pytest.fixture
def reviewer_user2(test_db):
    """Create second reviewer user."""
    user = LabMember(
        id=uuid4(),
        email="reviewer2@test.com",
        full_name="Reviewer Two",
        role="member",
        can_review=True,
        is_active=True,
        gsuite_id="reviewer2_gsuite_id",
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def reviewer_token2(reviewer_user2):
    from backend.api.review_auth import create_jwt_token
    return create_jwt_token(
        user_id=str(reviewer_user2.id),
        email=reviewer_user2.email,
        role=reviewer_user2.role,
    )


@pytest.fixture
def sample_application2(test_db):
    """Create second sample application."""
    email = Email(
        id=uuid4(),
        message_id="<test-app2@example.com>",
        uid="uid_app2",
        subject="Postdoc Application",
        from_name="Jane Smith",
        from_address="jane@university.edu",
        to_addresses=["lab@university.edu"],
        date=datetime.utcnow() - timedelta(days=3),
        body_text="Postdoc application...",
    )
    test_db.add(email)
    test_db.flush()
    metadata = EmailMetadata(
        email_id=email.id,
        ai_category="application-postdoc",
        applicant_name="Jane Smith",
        applicant_institution="University of Test",
        overall_recommendation_score=9,
        application_status="pending",
    )
    test_db.add(metadata)
    test_db.commit()
    test_db.refresh(email)
    test_db.refresh(metadata)
    return email, metadata


@pytest.fixture
def sample_batch(test_db, admin_user, reviewer_user, sample_application):
    """Create a batch with one assignment."""
    email, metadata = sample_application
    batch = AssignmentBatch(
        created_by=admin_user.id,
        notes="Test batch",
        deadline=datetime.now(timezone.utc) + timedelta(days=7),
    )
    test_db.add(batch)
    test_db.flush()

    assignment = ApplicationReviewAssignment(
        email_id=email.id,
        assigned_to=reviewer_user.id,
        batch_id=batch.id,
    )
    test_db.add(assignment)
    test_db.commit()
    test_db.refresh(batch)
    test_db.refresh(assignment)
    return batch, assignment


# ============================================================================
# POST /assignments — Create assignments
# ============================================================================

class TestCreateAssignments:
    def test_create_single_assignment(self, client, admin_token, reviewer_user, sample_application):
        email, _ = sample_application
        resp = client.post(
            "/assignments",
            json={
                "email_ids": [str(email.id)],
                "assigned_to": [str(reviewer_user.id)],
                "notes": "Please review",
                "deadline": (datetime.utcnow() + timedelta(days=7)).isoformat() + "Z",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["created"] == 1
        assert data["skipped_duplicates"] == 0
        assert data["batch_id"] is not None

    def test_create_bulk_assignments(self, client, admin_token, reviewer_user, reviewer_user2, sample_application, sample_application2):
        email1, _ = sample_application
        email2, _ = sample_application2
        resp = client.post(
            "/assignments",
            json={
                "email_ids": [str(email1.id), str(email2.id)],
                "assigned_to": [str(reviewer_user.id), str(reviewer_user2.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["created"] == 4  # 2 emails * 2 reviewers

    def test_create_with_sharing(self, client, admin_token, reviewer_user, reviewer_user2, sample_application, test_db):
        email, _ = sample_application
        resp = client.post(
            "/assignments",
            json={
                "email_ids": [str(email.id)],
                "assigned_to": [str(reviewer_user.id)],
                "share_with": [str(reviewer_user2.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        batch_id = UUID(resp.json()["batch_id"])

        # Verify share was created
        shares = test_db.query(AssignmentBatchShare).filter(
            AssignmentBatchShare.batch_id == batch_id,
        ).all()
        assert len(shares) == 1
        assert shares[0].shared_with == reviewer_user2.id

    def test_create_rejects_non_reviewer(self, client, admin_token, regular_user, sample_application):
        email, _ = sample_application
        resp = client.post(
            "/assignments",
            json={
                "email_ids": [str(email.id)],
                "assigned_to": [str(regular_user.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 400
        assert "review permission" in resp.json()["detail"]["message"]

    def test_create_skips_duplicates(self, client, admin_token, reviewer_user, sample_application, sample_batch):
        email, _ = sample_application
        # reviewer_user already assigned via sample_batch
        resp = client.post(
            "/assignments",
            json={
                "email_ids": [str(email.id)],
                "assigned_to": [str(reviewer_user.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        # All duplicates → 409
        assert resp.status_code == 409
        data = resp.json()["detail"]
        assert data["error"] == "all_duplicates"
        assert len(data["duplicates"]) == 1

    def test_create_partial_duplicates(self, client, admin_token, reviewer_user, reviewer_user2, sample_application, sample_batch):
        email, _ = sample_application
        # reviewer_user is already assigned, reviewer_user2 is not
        resp = client.post(
            "/assignments",
            json={
                "email_ids": [str(email.id)],
                "assigned_to": [str(reviewer_user.id), str(reviewer_user2.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["created"] == 1
        assert data["skipped_duplicates"] == 1

    def test_create_validates_max_100(self, client, admin_token, reviewer_user, sample_application):
        """Reject if cross-product exceeds 100."""
        email, _ = sample_application
        # Can't easily make 100+ IDs, but we can test the validation message
        fake_ids = [str(uuid4()) for _ in range(51)]
        resp = client.post(
            "/assignments",
            json={
                "email_ids": fake_ids,
                "assigned_to": [str(reviewer_user.id), str(uuid4())],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 400
        assert "Maximum 100" in resp.json()["detail"]["message"]

    def test_create_requires_auth(self, client, sample_application):
        email, _ = sample_application
        resp = client.post(
            "/assignments",
            json={
                "email_ids": [str(email.id)],
                "assigned_to": [str(uuid4())],
            },
        )
        assert resp.status_code in (401, 403)


# ============================================================================
# GET /assignments — List my assignments
# ============================================================================

class TestListMyAssignments:
    def test_list_pending_assignments(self, client, reviewer_token, reviewer_user, sample_batch):
        resp = client.get(
            "/assignments",
            headers={"Authorization": f"Bearer {reviewer_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["assignments"]) == 1
        assert data["assignments"][0]["status"] == "pending"
        assert data["summary"]["total_pending"] == 1

    def test_list_empty_for_unassigned_user(self, client, admin_token, sample_batch):
        """Admin created the batch but isn't assigned — should see nothing."""
        resp = client.get(
            "/assignments",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert len(resp.json()["assignments"]) == 0

    def test_filter_by_status(self, client, reviewer_token, reviewer_user, sample_batch, test_db):
        batch, assignment = sample_batch
        # Decline the assignment
        assignment.status = "declined"
        assignment.declined_at = datetime.now(timezone.utc)
        test_db.commit()

        # Should not show in default (pending) view
        resp = client.get("/assignments", headers={"Authorization": f"Bearer {reviewer_token}"})
        assert len(resp.json()["assignments"]) == 0

        # Should show with status=declined
        resp = client.get("/assignments?status=declined", headers={"Authorization": f"Bearer {reviewer_token}"})
        assert len(resp.json()["assignments"]) == 1

        # Should show with status=all
        resp = client.get("/assignments?status=all", headers={"Authorization": f"Bearer {reviewer_token}"})
        assert len(resp.json()["assignments"]) == 1

    def test_batch_stats_included(self, client, reviewer_token, sample_batch):
        resp = client.get("/assignments", headers={"Authorization": f"Bearer {reviewer_token}"})
        data = resp.json()
        a = data["assignments"][0]
        assert a["batch_total"] == 1
        assert a["batch_completed"] == 0

    def test_pagination(self, client, admin_token, reviewer_user, reviewer_token, test_db):
        """Create many assignments and test pagination."""
        batch = AssignmentBatch(created_by=admin_token)
        # Create 3 applications + assignments
        batch = AssignmentBatch(
            created_by=reviewer_user.id,  # self-assign for simplicity
            notes="Pagination test",
        )
        test_db.add(batch)
        test_db.flush()

        for i in range(3):
            email = Email(
                id=uuid4(), message_id=f"<page-test-{i}@test.com>",
                uid=f"puid_{i}", subject=f"App {i}",
                from_name=f"Person {i}", from_address=f"p{i}@test.com",
                to_addresses=["lab@test.com"], date=datetime.utcnow(),
            )
            test_db.add(email)
            test_db.flush()
            meta = EmailMetadata(email_id=email.id, ai_category="application-phd", applicant_name=f"Person {i}", application_status="pending")
            test_db.add(meta)
            test_db.flush()
            a = ApplicationReviewAssignment(email_id=email.id, assigned_to=reviewer_user.id, batch_id=batch.id)
            test_db.add(a)
        test_db.commit()

        resp = client.get("/assignments?page_size=2&status=all", headers={"Authorization": f"Bearer {reviewer_token}"})
        data = resp.json()
        assert data["pagination"]["total_items"] == 3
        assert data["pagination"]["total_pages"] == 2
        assert len(data["assignments"]) == 2


# ============================================================================
# GET /assignments/batches — List my batches
# ============================================================================

class TestListBatches:
    def test_list_owned_batches(self, client, admin_token, sample_batch):
        resp = client.get("/assignments/batches", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["batches"]) == 1
        assert data["batches"][0]["is_owner"] is True
        assert data["batches"][0]["stats"]["total"] == 1

    def test_list_shared_batches(self, client, reviewer_token2, reviewer_user2, sample_batch, test_db):
        batch, _ = sample_batch
        # Share with reviewer2
        share = AssignmentBatchShare(batch_id=batch.id, shared_with=reviewer_user2.id)
        test_db.add(share)
        test_db.commit()

        resp = client.get("/assignments/batches", headers={"Authorization": f"Bearer {reviewer_token2}"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["batches"]) == 1
        assert data["batches"][0]["is_owner"] is False

    def test_assignee_cannot_see_in_batch_list(self, client, reviewer_token, sample_batch):
        """Assignee should NOT see batch in /batches (that's for managers)."""
        resp = client.get("/assignments/batches", headers={"Authorization": f"Bearer {reviewer_token}"})
        assert resp.status_code == 200
        # reviewer_user is assigned but not creator/shared
        assert len(resp.json()["batches"]) == 0


# ============================================================================
# GET /assignments/batch/{id} — Batch detail
# ============================================================================

class TestGetBatchDetail:
    def test_owner_sees_all_assignments(self, client, admin_token, sample_batch):
        batch, _ = sample_batch
        resp = client.get(f"/assignments/batch/{batch.id}", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_edit"] is True
        assert len(data["assignments"]) == 1
        assert data["assignments"][0]["assignee_name"] is not None

    def test_assignee_sees_own_assignments_only(self, client, reviewer_token, reviewer_user2, sample_batch, test_db):
        batch, _ = sample_batch
        # Add a second assignment for reviewer2
        email2 = Email(
            id=uuid4(), message_id="<batch-detail-test@test.com>",
            uid="bd_uid", subject="Another App", from_name="X",
            from_address="x@test.com", to_addresses=["lab@test.com"],
            date=datetime.utcnow(),
        )
        test_db.add(email2)
        test_db.flush()
        meta2 = EmailMetadata(email_id=email2.id, ai_category="application-phd", applicant_name="X", application_status="pending")
        test_db.add(meta2)
        a2 = ApplicationReviewAssignment(email_id=email2.id, assigned_to=reviewer_user2.id, batch_id=batch.id)
        test_db.add(a2)
        test_db.commit()

        # reviewer sees only their assignment
        resp = client.get(f"/assignments/batch/{batch.id}", headers={"Authorization": f"Bearer {reviewer_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["can_edit"] is False
        assert len(data["assignments"]) == 1

    def test_unauthorized_user_gets_404(self, client, regular_token, sample_batch):
        batch, _ = sample_batch
        resp = client.get(f"/assignments/batch/{batch.id}", headers={"Authorization": f"Bearer {regular_token}"})
        assert resp.status_code == 404


# ============================================================================
# PATCH /assignments/batch/{id} — Update batch
# ============================================================================

class TestUpdateBatch:
    def test_owner_can_update(self, client, admin_token, sample_batch):
        batch, _ = sample_batch
        resp = client.patch(
            f"/assignments/batch/{batch.id}",
            json={"notes": "Updated notes", "deadline": (datetime.utcnow() + timedelta(days=14)).isoformat() + "Z"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200

    def test_shared_user_can_update(self, client, reviewer_token2, reviewer_user2, sample_batch, test_db):
        batch, _ = sample_batch
        share = AssignmentBatchShare(batch_id=batch.id, shared_with=reviewer_user2.id)
        test_db.add(share)
        test_db.commit()

        resp = client.patch(
            f"/assignments/batch/{batch.id}",
            json={"notes": "Shared user update"},
            headers={"Authorization": f"Bearer {reviewer_token2}"},
        )
        assert resp.status_code == 200

    def test_assignee_cannot_update(self, client, reviewer_token, sample_batch):
        batch, _ = sample_batch
        resp = client.patch(
            f"/assignments/batch/{batch.id}",
            json={"notes": "Assignee update"},
            headers={"Authorization": f"Bearer {reviewer_token}"},
        )
        assert resp.status_code == 403

    def test_update_shares(self, client, admin_token, reviewer_user2, sample_batch, test_db):
        batch, _ = sample_batch
        resp = client.patch(
            f"/assignments/batch/{batch.id}",
            json={"share_with": [str(reviewer_user2.id)]},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        shares = test_db.query(AssignmentBatchShare).filter(AssignmentBatchShare.batch_id == batch.id).all()
        assert len(shares) == 1

        # Remove all shares
        resp = client.patch(
            f"/assignments/batch/{batch.id}",
            json={"share_with": []},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        shares = test_db.query(AssignmentBatchShare).filter(AssignmentBatchShare.batch_id == batch.id).all()
        assert len(shares) == 0


# ============================================================================
# DELETE /assignments/batch/{id} — Delete batch
# ============================================================================

class TestDeleteBatch:
    def test_owner_can_delete(self, client, admin_token, sample_batch, test_db):
        batch, assignment = sample_batch
        batch_id = batch.id
        resp = client.delete(f"/assignments/batch/{batch_id}", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200

        # Verify cascade deleted assignments
        remaining = test_db.query(ApplicationReviewAssignment).filter(
            ApplicationReviewAssignment.batch_id == batch_id,
        ).all()
        assert len(remaining) == 0

    def test_assignee_cannot_delete_batch(self, client, reviewer_token, sample_batch):
        batch, _ = sample_batch
        resp = client.delete(f"/assignments/batch/{batch.id}", headers={"Authorization": f"Bearer {reviewer_token}"})
        assert resp.status_code == 403


# ============================================================================
# POST /assignments/batch/{id}/assignments — Add to batch
# ============================================================================

class TestAddToBatch:
    def test_add_new_assignments(self, client, admin_token, reviewer_user2, sample_batch, sample_application2):
        batch, _ = sample_batch
        email2, _ = sample_application2
        resp = client.post(
            f"/assignments/batch/{batch.id}/assignments",
            json={
                "email_ids": [str(email2.id)],
                "assigned_to": [str(reviewer_user2.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["created"] == 1

    def test_add_rejects_non_reviewer(self, client, admin_token, regular_user, sample_batch, sample_application2):
        batch, _ = sample_batch
        email2, _ = sample_application2
        resp = client.post(
            f"/assignments/batch/{batch.id}/assignments",
            json={
                "email_ids": [str(email2.id)],
                "assigned_to": [str(regular_user.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 400


# ============================================================================
# GET /assignments/application/{email_id} — Application assignments
# ============================================================================

class TestApplicationAssignments:
    def test_get_assignments_for_application(self, client, reviewer_token, sample_batch, sample_application):
        email, _ = sample_application
        resp = client.get(
            f"/assignments/application/{email.id}",
            headers={"Authorization": f"Bearer {reviewer_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["assignments"]) == 1
        assert data["assignments"][0]["status"] == "pending"

    def test_empty_for_unassigned_application(self, client, reviewer_token, sample_application2):
        email, _ = sample_application2
        resp = client.get(
            f"/assignments/application/{email.id}",
            headers={"Authorization": f"Bearer {reviewer_token}"},
        )
        assert resp.status_code == 200
        assert len(resp.json()["assignments"]) == 0


# ============================================================================
# PATCH /assignments/{id} — Decline assignment
# ============================================================================

class TestDeclineAssignment:
    def test_assignee_can_decline(self, client, reviewer_token, sample_batch, test_db):
        _, assignment = sample_batch
        resp = client.patch(
            f"/assignments/{assignment.id}",
            json={"status": "declined", "declined_reason": "On leave"},
            headers={"Authorization": f"Bearer {reviewer_token}"},
        )
        assert resp.status_code == 200

        test_db.refresh(assignment)
        assert assignment.status == "declined"
        assert assignment.declined_reason == "On leave"
        assert assignment.declined_at is not None

    def test_non_assignee_cannot_decline(self, client, admin_token, sample_batch):
        _, assignment = sample_batch
        resp = client.patch(
            f"/assignments/{assignment.id}",
            json={"status": "declined", "declined_reason": "Not my assignment"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 403

    def test_cannot_decline_completed(self, client, reviewer_token, sample_batch, test_db):
        _, assignment = sample_batch
        assignment.status = "completed"
        assignment.completed_at = datetime.now(timezone.utc)
        test_db.commit()

        resp = client.patch(
            f"/assignments/{assignment.id}",
            json={"status": "declined"},
            headers={"Authorization": f"Bearer {reviewer_token}"},
        )
        assert resp.status_code == 400


# ============================================================================
# DELETE /assignments/{id} — Remove individual assignment
# ============================================================================

class TestDeleteAssignment:
    def test_batch_owner_can_delete(self, client, admin_token, sample_batch, test_db):
        _, assignment = sample_batch
        aid = assignment.id
        resp = client.delete(f"/assignments/{aid}", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200
        assert test_db.query(ApplicationReviewAssignment).filter_by(id=aid).first() is None

    def test_assignee_cannot_delete(self, client, reviewer_token, sample_batch):
        _, assignment = sample_batch
        resp = client.delete(f"/assignments/{assignment.id}", headers={"Authorization": f"Bearer {reviewer_token}"})
        assert resp.status_code == 403

    def test_shared_user_can_delete(self, client, reviewer_token2, reviewer_user2, sample_batch, test_db):
        batch, assignment = sample_batch
        share = AssignmentBatchShare(batch_id=batch.id, shared_with=reviewer_user2.id)
        test_db.add(share)
        test_db.commit()

        resp = client.delete(f"/assignments/{assignment.id}", headers={"Authorization": f"Bearer {reviewer_token2}"})
        assert resp.status_code == 200


# ============================================================================
# POST /assignments/preview — Preview bulk assignment
# ============================================================================

class TestPreviewBulkAssignment:
    def test_preview_returns_applications(self, client, admin_token, reviewer_user, sample_application):
        email, _ = sample_application
        resp = client.post(
            "/assignments/preview",
            json={
                "date_from": (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "date_to": datetime.utcnow().strftime("%Y-%m-%d"),
                "assigned_to": [str(reviewer_user.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_applications"] >= 1
        assert data["new_to_assign"] >= 1

    def test_preview_detects_duplicates(self, client, admin_token, reviewer_user, sample_application, sample_batch):
        email, _ = sample_application
        resp = client.post(
            "/assignments/preview",
            json={
                "date_from": (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "date_to": datetime.utcnow().strftime("%Y-%m-%d"),
                "assigned_to": [str(reviewer_user.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["already_assigned"] >= 1

    def test_preview_with_category_filter(self, client, admin_token, reviewer_user, sample_application, sample_application2):
        resp = client.post(
            "/assignments/preview",
            json={
                "date_from": (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d"),
                "date_to": datetime.utcnow().strftime("%Y-%m-%d"),
                "assigned_to": [str(reviewer_user.id)],
                "category": "application-phd",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        # Should only include phd apps, not postdoc
        for app in data["applications"]:
            assert app["category"] == "application-phd"


# ============================================================================
# Auto-completion on review submission
# ============================================================================

class TestAutoCompletion:
    def test_complete_assignments_on_review(self, test_db, reviewer_user, sample_batch):
        _, assignment = sample_batch
        assert assignment.status == "pending"

        completed = complete_assignments_on_review(
            test_db, assignment.email_id, reviewer_user.id
        )
        test_db.commit()

        test_db.refresh(assignment)
        assert assignment.status == "completed"
        assert assignment.completed_at is not None
        assert len(completed) == 1

    def test_no_op_when_no_pending_assignments(self, test_db, reviewer_user, sample_batch):
        _, assignment = sample_batch
        assignment.status = "declined"
        test_db.commit()

        completed = complete_assignments_on_review(
            test_db, assignment.email_id, reviewer_user.id
        )
        assert len(completed) == 0
        test_db.refresh(assignment)
        assert assignment.status == "declined"  # Unchanged

    def test_no_op_for_wrong_reviewer(self, test_db, admin_user, sample_batch):
        _, assignment = sample_batch
        completed = complete_assignments_on_review(
            test_db, assignment.email_id, admin_user.id  # Not the assignee
        )
        assert len(completed) == 0
        test_db.refresh(assignment)
        assert assignment.status == "pending"


# ============================================================================
# Overdue detection
# ============================================================================

class TestOverdueDetection:
    def test_overdue_assignment_flagged(self, client, reviewer_token, reviewer_user, sample_application, test_db):
        email, _ = sample_application
        # Create batch with past deadline
        batch = AssignmentBatch(
            created_by=reviewer_user.id,
            deadline=datetime.now(timezone.utc) - timedelta(days=1),
        )
        test_db.add(batch)
        test_db.flush()
        a = ApplicationReviewAssignment(
            email_id=email.id, assigned_to=reviewer_user.id, batch_id=batch.id,
        )
        test_db.add(a)
        test_db.commit()

        resp = client.get("/assignments?status=all", headers={"Authorization": f"Bearer {reviewer_token}"})
        data = resp.json()
        overdue_items = [x for x in data["assignments"] if x["is_overdue"]]
        assert len(overdue_items) >= 1
        assert data["summary"]["total_overdue"] >= 1

    def test_non_overdue_when_no_deadline(self, client, reviewer_token, sample_batch):
        # sample_batch has a future deadline
        resp = client.get("/assignments", headers={"Authorization": f"Bearer {reviewer_token}"})
        data = resp.json()
        assert data["assignments"][0]["is_overdue"] is False


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_nonexistent_batch_returns_404(self, client, admin_token):
        fake_id = str(uuid4())
        resp = client.get(f"/assignments/batch/{fake_id}", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 404

    def test_nonexistent_assignment_returns_404(self, client, admin_token):
        fake_id = str(uuid4())
        resp = client.patch(
            f"/assignments/{fake_id}",
            json={"status": "declined"},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 404

    def test_invalid_email_id_rejected(self, client, admin_token, reviewer_user):
        fake_email = str(uuid4())
        resp = client.post(
            "/assignments",
            json={
                "email_ids": [fake_email],
                "assigned_to": [str(reviewer_user.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"]["message"]

    def test_self_assignment(self, client, reviewer_token, reviewer_user, sample_application):
        """Reviewer can assign applications to themselves."""
        email, _ = sample_application
        resp = client.post(
            "/assignments",
            json={
                "email_ids": [str(email.id)],
                "assigned_to": [str(reviewer_user.id)],
            },
            headers={"Authorization": f"Bearer {reviewer_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["created"] == 1

    def test_batch_stats_reflect_completed(self, client, admin_token, reviewer_user, reviewer_token, sample_batch, test_db):
        """Batch detail stats should update when an assignment is completed."""
        batch, assignment = sample_batch
        # Complete the assignment
        assignment.status = "completed"
        assignment.completed_at = datetime.now(timezone.utc)
        test_db.commit()

        resp = client.get(f"/assignments/batch/{batch.id}", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["stats"]["completed"] == 1
        assert data["stats"]["total"] == 1

    def test_batch_update_persists(self, client, admin_token, sample_batch, test_db):
        """After updating batch notes, getting batch detail reflects changes."""
        batch, _ = sample_batch
        new_notes = "Freshly updated notes"
        resp = client.patch(
            f"/assignments/batch/{batch.id}",
            json={"notes": new_notes},
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200

        # Verify via batch detail endpoint
        resp = client.get(f"/assignments/batch/{batch.id}", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200
        assert resp.json()["notes"] == new_notes

    def test_decline_then_reassign(self, client, admin_token, reviewer_token, reviewer_user, reviewer_user2, sample_batch, test_db):
        """After declining, the same app can be assigned to another reviewer."""
        batch, assignment = sample_batch
        email_id = assignment.email_id

        # Decline
        client.patch(
            f"/assignments/{assignment.id}",
            json={"status": "declined", "declined_reason": "Conflict of interest"},
            headers={"Authorization": f"Bearer {reviewer_token}"},
        )

        # Assign to reviewer2
        resp = client.post(
            f"/assignments/batch/{batch.id}/assignments",
            json={
                "email_ids": [str(email_id)],
                "assigned_to": [str(reviewer_user2.id)],
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert resp.json()["created"] == 1
