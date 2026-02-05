# Implementation Plan: Review Assignments Backend

## Overview

Implement the backend for the Review Assignments feature as specified in
`lab-application-review/docs/feature-request-review-assignments.md`.

## Self-Review Notes

**Verified against codebase:**
- `review_admin.py` already has `GET /admin/users` returning `can_review` field — no new reviewers endpoint needed. Frontend can use this.
- `build_application_query()` in `review_applications.py:88` takes named filter params. The `needs_review_by_me` pattern (bool query param → UUID internal param → `notin_` subquery) is the template for `assigned_to_me`.
- Review submission at `review_applications.py:1189-1260` commits twice (existing review path at line 1233, new review path at line 1259). Auto-completion hook must be called before each `db.commit()`.
- `review_stats.py` has `GET /stats/overview` — add assignment counts there.
- Migration chain: 001-006 exist; next is 007.
- Models use `datetime.utcnow` (naive) throughout. Spec says `datetime.now(timezone.utc)`. I'll use timezone-aware in new code but won't change existing patterns.

**Risks identified:**
- The review submission has TWO commit paths (line 1233 for update, line 1259 for create). Must call auto-completion before both.
- `build_application_query` returns a Query object. Adding EXISTS subquery filter is straightforward.

---

## Step 1: Database Models

**File:** `backend/core/database/models.py`

Add after `ApplicationShareToken` (line ~1006):
- `AssignmentBatch` model
- `AssignmentBatchShare` model
- `ApplicationReviewAssignment` model

Follow the spec exactly: FK behaviors (SET NULL, RESTRICT, CASCADE), unique constraints, indexes, `datetime.now(timezone.utc)` lambdas, relationships.

**File:** `backend/core/database/__init__.py`

Add new models to imports and `__all__`.

## Step 2: Alembic Migration

**File:** `alembic/versions/007_review_assignments.py`

Create tables in order (batches → shares → assignments) with all indexes and constraints per spec.

## Step 3: Pydantic Schemas

**File:** `backend/api/review_schemas.py`

Add schemas:
- `CreateAssignmentRequest` (email_ids, assigned_to, deadline, notes, share_with)
- `CreateAssignmentResponse` (batch_id, created, skipped_duplicates, duplicates)
- `AssignmentResponse` (for list items — includes derived fields)
- `BatchResponse` (with stats, shared_with, can_edit)
- `BatchDetailResponse` (batch + assignments list)
- `UpdateBatchRequest` (notes, deadline, share_with)
- `DeclineAssignmentRequest` (status, declined_reason)
- `AddToBatchRequest` (email_ids, assigned_to)
- `PreviewBulkAssignmentRequest` (date_from, date_to, assigned_to, category)
- `PreviewBulkAssignmentResponse`
- `ApplicationAssignmentResponse` (for per-application view)

## Step 4: Routes — New File

**File:** `backend/api/routes/review_assignments.py`

Router prefix: `/assignments`

Endpoints (in order of implementation):

1. **POST `/`** — Create batch + assignments
   - Auth: `get_current_user_hybrid`
   - Validate `can_review` for all assignees
   - Create batch, shares, cross-product assignments
   - Skip duplicates with detailed info
   - Single transaction
   - Max 100 assignments per request

2. **GET `/`** — My assignments (as assignee)
   - Auth: `get_current_user_hybrid`
   - Batch stats via subquery (avoid N+1)
   - Pagination, status filter, batch_id filter
   - Summary: total_pending, total_overdue

3. **GET `/batches`** — Batches I manage
   - Auth: `get_current_user_hybrid`
   - Union of owned + shared batches
   - Stats per batch via aggregate subquery
   - Pagination

4. **GET `/batch/{batch_id}`** — Batch detail
   - Permission: creator, shared user, or assignee
   - Assignees see only their own assignments
   - `can_edit` flag based on creator/shared

5. **PATCH `/batch/{batch_id}`** — Update batch
   - Permission: creator or shared users
   - Replace shares list
   - Update notes/deadline

6. **DELETE `/batch/{batch_id}`** — Delete batch
   - Permission: creator or shared users
   - CASCADE deletes assignments

7. **POST `/batch/{batch_id}/assignments`** — Add to batch
   - Permission: creator or shared users
   - Same validation as POST `/`

8. **GET `/application/{email_id}`** — Assignments for application
   - Auth: any authenticated user
   - Return all assignments with assignee/assigner names

9. **PATCH `/{id}`** — Decline assignment
   - Permission: assignee only
   - Set status, declined_reason, declined_at

10. **DELETE `/{id}`** — Remove single assignment
    - Permission: creator or shared users of parent batch

11. **POST `/preview`** — Preview bulk assignment
    - Query applications by date range + category
    - Check existing assignments for duplicates

### Helper Functions (in same file):

- `_check_batch_permission(db, batch, user, require_edit=False)` — centralized permission check
- `_get_batch_stats(db, batch_ids)` — efficient stats query
- `_build_assignment_response(assignment, batch, stats, ...)` — build response dict
- `complete_assignments_on_review(db, email_id, reviewer_id)` — auto-completion (no commit)

## Step 5: Auto-Completion Hook

**File:** `backend/api/routes/review_applications.py`

In `submit_review()` (~line 1189):
- Import `complete_assignments_on_review` from review_assignments
- Call before each `db.commit()` (lines 1233 and 1259)
- Wrap in try/except so assignment failure doesn't break review submission

## Step 6: `assigned_to_me` Filter

**File:** `backend/api/routes/review_applications.py`

In `build_application_query()` (~line 88):
- Add parameter: `assigned_to_me: Optional[UUID] = None`
- Add EXISTS subquery filter (not JOIN)

In `list_applications()` (~line 317):
- Add query param: `assigned_to_me: Optional[bool] = Query(None)`
- Pass `assigned_to_me=current_user.id if assigned_to_me else None`

## Step 7: Stats Update

**File:** `backend/api/routes/review_stats.py`

In `get_overview_stats()`:
- Add `my_pending_assignments` count
- Add `my_overdue_assignments` count (pending + deadline < now)

## Step 8: Register Router

**File:** `backend/api/main.py`

- Import `review_assignments`
- Add `app.include_router(review_assignments.router)` after collections router

## Step 9: Run Migration on Pi

```bash
~/git/services/mail-done-config/deploy/sync-and-deploy.sh deploy
```

---

## Implementation Order

1. Models + __init__ exports (Step 1)
2. Migration (Step 2)
3. Schemas (Step 3)
4. Routes file with all endpoints (Step 4)
5. Auto-completion hook (Step 5)
6. assigned_to_me filter (Step 6)
7. Stats update (Step 7)
8. Register router (Step 8)
9. Deploy + run migration (Step 9)
