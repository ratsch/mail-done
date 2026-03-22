"""
Tests for Property Listing API endpoints (/listings/*)

Covers all 29 route handlers in review_listings.py.
"""
import os
import pytest
from uuid import uuid4

from backend.core.database.models_property import PropertyListing

# Ensure CONFIG_DIR points to the project config directory
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
os.environ.setdefault("CONFIG_DIR", os.path.join(_PROJECT_ROOT, "config"))


def auth(token):
    return {"Authorization": f"Bearer {token}"}


# =============================================================================
# Auth / Permission Tests
# =============================================================================

class TestListingAuth:
    """Authentication and permission tests."""

    def test_no_auth_returns_401(self, client):
        response = client.get("/listings")
        assert response.status_code == 401

    def test_regular_user_returns_403(self, client, regular_token):
        response = client.get("/listings", headers=auth(regular_token))
        assert response.status_code == 403

    def test_reviewer_can_access(self, client, reviewer_token):
        response = client.get("/listings", headers=auth(reviewer_token))
        assert response.status_code == 200

    def test_admin_can_access(self, client, admin_token):
        response = client.get("/listings", headers=auth(admin_token))
        assert response.status_code == 200


# =============================================================================
# List & Filter (GET /listings)
# =============================================================================

class TestListListings:
    """Tests for GET /listings."""

    def test_empty_list(self, client, reviewer_token):
        response = client.get("/listings", headers=auth(reviewer_token))
        assert response.status_code == 200
        data = response.json()
        assert data["items"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["total_pages"] == 1

    def test_list_with_data(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", headers=auth(reviewer_token))
        data = response.json()
        # Default exclude_archived=True, so the archived listing (#4) is excluded
        assert data["total"] == 4

    def test_pagination(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", params={"limit": 2, "page": 1}, headers=auth(reviewer_token))
        data = response.json()
        assert len(data["items"]) == 2
        assert data["limit"] == 2
        assert data["total_pages"] >= 2

    def test_filter_by_plz(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", params={"plz": "7250"}, headers=auth(reviewer_token))
        data = response.json()
        assert all(item["plz"] == "7250" for item in data["items"])

    def test_filter_by_plz_comma_separated(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", params={"plz": "7250,8044"}, headers=auth(reviewer_token))
        data = response.json()
        assert all(item["plz"] in ("7250", "8044") for item in data["items"])

    def test_filter_by_min_price(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", params={"min_price": 1500000}, headers=auth(reviewer_token))
        data = response.json()
        assert all(item["price_chf"] >= 1500000 for item in data["items"])

    def test_filter_by_max_price(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", params={"max_price": 1000000}, headers=auth(reviewer_token))
        data = response.json()
        assert all(item["price_chf"] <= 1000000 for item in data["items"])

    def test_filter_by_listing_status(self, client, reviewer_token, multiple_listings):
        response = client.get(
            "/listings", params={"listing_status": "scored", "exclude_archived": False},
            headers=auth(reviewer_token),
        )
        data = response.json()
        assert all(item["listing_status"] == "scored" for item in data["items"])

    def test_filter_by_tier(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", params={"tier": 1}, headers=auth(reviewer_token))
        data = response.json()
        assert all(item["tier"] == 1 for item in data["items"])

    def test_filter_by_min_recommendation(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", params={"min_recommendation": 8}, headers=auth(reviewer_token))
        data = response.json()
        assert all(item["overall_recommendation"] >= 8 for item in data["items"])

    def test_filter_by_listing_source(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", params={"listing_source": "homegate"}, headers=auth(reviewer_token))
        data = response.json()
        assert all(item["listing_source"] == "homegate" for item in data["items"])

    def test_filter_by_scenario(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", params={"scenario": "A"}, headers=auth(reviewer_token))
        data = response.json()
        assert all(item["best_scenario"] == "A" for item in data["items"])

    def test_exclude_archived_default(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", headers=auth(reviewer_token))
        data = response.json()
        assert all(item["listing_status"] != "archived" for item in data["items"])

    def test_include_archived(self, client, reviewer_token, multiple_listings):
        response = client.get(
            "/listings", params={"exclude_archived": False}, headers=auth(reviewer_token),
        )
        data = response.json()
        statuses = [item["listing_status"] for item in data["items"]]
        assert "archived" in statuses

    def test_search(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings", params={"search": "Klosters"}, headers=auth(reviewer_token))
        data = response.json()
        assert data["total"] >= 1

    def test_sort_by_price_asc(self, client, reviewer_token, multiple_listings):
        response = client.get(
            "/listings", params={"sort_by": "price", "sort_order": "asc"},
            headers=auth(reviewer_token),
        )
        data = response.json()
        prices = [i["price_chf"] for i in data["items"] if i["price_chf"] is not None]
        assert prices == sorted(prices)

    def test_has_action_filter(self, client, reviewer_token, sample_listing, listing_with_action):
        response = client.get("/listings", params={"has_action": True}, headers=auth(reviewer_token))
        data = response.json()
        assert data["total"] >= 1

    def test_collection_filter(self, client, reviewer_token, sample_listing, sample_collection, collection_with_item):
        response = client.get(
            "/listings", params={"collection_id": str(sample_collection.id)},
            headers=auth(reviewer_token),
        )
        data = response.json()
        assert data["total"] == 1
        assert data["items"][0]["id"] == str(sample_listing.id)

    def test_response_includes_computed_fields(self, client, reviewer_token, sample_listing, listing_with_review):
        response = client.get("/listings", headers=auth(reviewer_token))
        data = response.json()
        item = data["items"][0]
        assert "avg_rating" in item
        assert "num_ratings" in item
        assert "my_rating" in item
        assert "latest_action" in item
        assert "latest_action_at" in item


# =============================================================================
# Detail View (GET /listings/{id})
# =============================================================================

class TestListingDetail:
    """Tests for GET /listings/{listing_id}."""

    def test_get_detail(self, client, reviewer_token, sample_listing):
        response = client.get(f"/listings/{sample_listing.id}", headers=auth(reviewer_token))
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_listing.id)
        assert data["address"] == "Talgasse 9, 7250 Klosters"
        assert data["price_chf"] == 1200000
        # Detail-only fields present
        assert "description" in data
        assert "sources" in data
        assert "reviews" in data
        assert "actions" in data
        assert "documents" in data
        assert "due_diligence_total" in data

    def test_detail_not_found(self, client, reviewer_token):
        response = client.get(f"/listings/{uuid4()}", headers=auth(reviewer_token))
        assert response.status_code == 404

    def test_detail_invalid_uuid(self, client, reviewer_token):
        response = client.get("/listings/not-a-uuid", headers=auth(reviewer_token))
        assert response.status_code == 404

    def test_detail_includes_reviews(self, client, reviewer_token, sample_listing, listing_with_review):
        response = client.get(f"/listings/{sample_listing.id}", headers=auth(reviewer_token))
        data = response.json()
        assert data["num_ratings"] == 1
        assert data["avg_rating"] == 4.0
        assert len(data["reviews"]) == 1
        assert data["reviews"][0]["rating"] == 4
        assert data["my_rating"] == 4  # reviewer_user's review

    def test_detail_includes_actions(self, client, admin_token, sample_listing, listing_with_action):
        response = client.get(f"/listings/{sample_listing.id}", headers=auth(admin_token))
        data = response.json()
        assert len(data["actions"]) == 1
        assert data["actions"][0]["action"] == "interested"
        assert data["latest_action"] == "interested"


# =============================================================================
# Reviews (POST/DELETE /listings/{id}/review)
# =============================================================================

class TestListingReviews:
    """Tests for review endpoints."""

    def test_submit_review(self, client, reviewer_token, sample_listing):
        response = client.post(
            f"/listings/{sample_listing.id}/review",
            json={"rating": 5, "comment": "Excellent"},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["avg_rating"] == 5.0
        assert data["num_ratings"] == 1

    def test_update_review_upsert(self, client, reviewer_token, sample_listing, listing_with_review):
        response = client.post(
            f"/listings/{sample_listing.id}/review",
            json={"rating": 2, "comment": "Changed my mind"},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["avg_rating"] == 2.0  # Updated, not duplicated
        assert data["num_ratings"] == 1

    def test_review_invalid_rating(self, client, reviewer_token, sample_listing):
        response = client.post(
            f"/listings/{sample_listing.id}/review",
            json={"rating": 6},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 422

    def test_delete_review(self, client, reviewer_token, sample_listing, listing_with_review):
        response = client.delete(
            f"/listings/{sample_listing.id}/review", headers=auth(reviewer_token),
        )
        assert response.status_code == 200

    def test_delete_review_not_found(self, client, reviewer_token, sample_listing):
        response = client.delete(
            f"/listings/{sample_listing.id}/review", headers=auth(reviewer_token),
        )
        assert response.status_code == 404


# =============================================================================
# Actions (POST/GET /listings/{id}/actions)
# =============================================================================

class TestListingActions:
    """Tests for action endpoints."""

    def test_record_action(self, client, admin_token, sample_listing):
        response = client.post(
            f"/listings/{sample_listing.id}/actions",
            json={"action": "interested", "notes": "Worth investigating"},
            headers=auth(admin_token),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["action"] == "interested"
        assert data["resulting_status"] == "under_evaluation"

    def test_action_updates_listing_status(self, client, admin_token, sample_listing):
        client.post(
            f"/listings/{sample_listing.id}/actions",
            json={"action": "request_viewing"},
            headers=auth(admin_token),
        )
        detail = client.get(f"/listings/{sample_listing.id}", headers=auth(admin_token)).json()
        assert detail["listing_status"] == "viewing_scheduled"

    def test_action_no_regression(self, client, admin_token, sample_listing):
        """Status shouldn't go backwards (viewing_scheduled → needs_info)."""
        # First: advance to viewing_scheduled
        client.post(
            f"/listings/{sample_listing.id}/actions",
            json={"action": "request_viewing"},
            headers=auth(admin_token),
        )
        # Then: request_info should not regress to needs_info
        response = client.post(
            f"/listings/{sample_listing.id}/actions",
            json={"action": "request_info"},
            headers=auth(admin_token),
        )
        data = response.json()
        assert data["resulting_status"] == "viewing_scheduled"  # No regression

    def test_terminal_action_always_applies(self, client, admin_token, sample_listing):
        """Archive should always work regardless of current status."""
        client.post(
            f"/listings/{sample_listing.id}/actions",
            json={"action": "request_viewing"},
            headers=auth(admin_token),
        )
        response = client.post(
            f"/listings/{sample_listing.id}/actions",
            json={"action": "archive"},
            headers=auth(admin_token),
        )
        data = response.json()
        assert data["resulting_status"] == "archived"

    def test_invalid_action_rejected(self, client, admin_token, sample_listing):
        response = client.post(
            f"/listings/{sample_listing.id}/actions",
            json={"action": "invalid_action"},
            headers=auth(admin_token),
        )
        assert response.status_code == 422

    def test_reviewer_cannot_record_action(self, client, reviewer_token, sample_listing):
        """Actions require admin role."""
        response = client.post(
            f"/listings/{sample_listing.id}/actions",
            json={"action": "interested"},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 403

    def test_get_action_history(self, client, admin_token, sample_listing, listing_with_action):
        response = client.get(
            f"/listings/{sample_listing.id}/actions", headers=auth(admin_token),
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["action"] == "interested"

    def test_action_history_multiple(self, client, admin_token, sample_listing):
        """Multiple actions are all recorded."""
        client.post(f"/listings/{sample_listing.id}/actions", json={"action": "interested"}, headers=auth(admin_token))
        client.post(f"/listings/{sample_listing.id}/actions", json={"action": "request_viewing"}, headers=auth(admin_token))
        response = client.get(f"/listings/{sample_listing.id}/actions", headers=auth(admin_token))
        data = response.json()
        assert len(data) == 2
        actions = {d["action"] for d in data}
        assert actions == {"interested", "request_viewing"}


# =============================================================================
# Manual Edit (PATCH /listings/{id})
# =============================================================================

class TestManualEdit:
    """Tests for PATCH /listings/{listing_id}."""

    def test_update_fields(self, client, reviewer_token, sample_listing):
        response = client.patch(
            f"/listings/{sample_listing.id}",
            json={"price_chf": 1150000, "agent_name": "Max Muster"},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 200
        detail = client.get(f"/listings/{sample_listing.id}", headers=auth(reviewer_token)).json()
        assert detail["price_chf"] == 1150000
        assert detail["agent_name"] == "Max Muster"

    def test_price_per_sqm_recalculated(self, client, reviewer_token, sample_listing):
        client.patch(
            f"/listings/{sample_listing.id}",
            json={"price_chf": 2400000},
            headers=auth(reviewer_token),
        )
        detail = client.get(f"/listings/{sample_listing.id}", headers=auth(reviewer_token)).json()
        assert detail["price_per_sqm"] == 20000  # 2400000 / 120

    def test_empty_update_rejected(self, client, reviewer_token, sample_listing):
        response = client.patch(
            f"/listings/{sample_listing.id}",
            json={},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 400


# =============================================================================
# Enrichment (POST /listings/{id}/enrich, PATCH /listings/{id}/houzy)
# =============================================================================

class TestEnrichment:
    """Tests for enrichment endpoints."""

    def test_trigger_enrichment(self, client, admin_token, sample_listing):
        response = client.post(
            f"/listings/{sample_listing.id}/enrich", headers=auth(admin_token),
        )
        assert response.status_code == 202
        assert "listing_id" in response.json()

    def test_update_houzy_params(self, client, reviewer_token, sample_listing):
        response = client.patch(
            f"/listings/{sample_listing.id}/houzy",
            json={"zustand": 3.5, "ausbaustandard": 4.0},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 200
        detail = client.get(f"/listings/{sample_listing.id}", headers=auth(reviewer_token)).json()
        assert detail["zustand_rating"] == 3.5
        assert detail["zustand_confirmed"] is True
        assert detail["ausbaustandard_rating"] == 4.0
        assert detail["ausbaustandard_confirmed"] is True

    def test_houzy_invalid_range(self, client, reviewer_token, sample_listing):
        response = client.patch(
            f"/listings/{sample_listing.id}/houzy",
            json={"zustand": 6.0, "ausbaustandard": 0.5},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 422


# =============================================================================
# Due Diligence
# =============================================================================

class TestDueDiligence:
    """Tests for due diligence endpoints."""

    def test_generate_checklist(self, client, admin_token, sample_listing):
        response = client.post(
            f"/listings/{sample_listing.id}/due-diligence",
            headers=auth(admin_token),
        )
        assert response.status_code == 201, f"Expected 201, got {response.status_code}: {response.json()}"
        data = response.json()
        assert data["items_created"] > 0

    def test_generate_checklist_idempotent(self, client, admin_token, sample_listing):
        client.post(f"/listings/{sample_listing.id}/due-diligence", headers=auth(admin_token))
        response = client.post(f"/listings/{sample_listing.id}/due-diligence", headers=auth(admin_token))
        assert response.status_code == 409

    def test_get_checklist(self, client, admin_token, reviewer_token, sample_listing):
        client.post(f"/listings/{sample_listing.id}/due-diligence", headers=auth(admin_token))
        response = client.get(f"/listings/{sample_listing.id}/due-diligence", headers=auth(reviewer_token))
        assert response.status_code == 200
        data = response.json()
        assert len(data) > 0
        assert "category" in data[0]
        assert "question" in data[0]
        assert data[0]["status"] == "open"

    def test_update_checklist_item(self, client, admin_token, reviewer_token, sample_listing):
        client.post(f"/listings/{sample_listing.id}/due-diligence", headers=auth(admin_token))
        items = client.get(f"/listings/{sample_listing.id}/due-diligence", headers=auth(reviewer_token)).json()
        item_id = items[0]["id"]

        response = client.patch(
            f"/listings/{sample_listing.id}/due-diligence/{item_id}",
            json={"status": "answered", "answer": "All clear", "source": "agent"},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "answered"
        assert data["answer"] == "All clear"
        assert data["date_answered"] is not None


# =============================================================================
# Documents
# =============================================================================

class TestDocuments:
    """Tests for document endpoints."""

    def test_add_document(self, client, reviewer_token, sample_listing):
        response = client.post(
            f"/listings/{sample_listing.id}/documents",
            json={"document_type": "verkaufsbrochure", "filename": "brochure.pdf", "source": "agent"},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 201
        data = response.json()
        assert data["document_type"] == "verkaufsbrochure"
        assert data["filename"] == "brochure.pdf"

    def test_list_documents(self, client, reviewer_token, sample_listing):
        client.post(
            f"/listings/{sample_listing.id}/documents",
            json={"document_type": "grundbuch", "filename": "grundbuch.pdf"},
            headers=auth(reviewer_token),
        )
        response = client.get(f"/listings/{sample_listing.id}/documents", headers=auth(reviewer_token))
        assert response.status_code == 200
        assert len(response.json()) == 1


# =============================================================================
# Private Notes (PUT /listings/{id}/private-notes)
# =============================================================================

class TestPrivateNotes:
    """Tests for private notes endpoint."""

    def test_create_private_notes(self, client, reviewer_token, sample_listing):
        response = client.put(
            f"/listings/{sample_listing.id}/private-notes",
            json={"notes": "Check the roof condition"},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 200

    def test_private_notes_in_detail(self, client, reviewer_token, sample_listing):
        client.put(
            f"/listings/{sample_listing.id}/private-notes",
            json={"notes": "My secret notes"},
            headers=auth(reviewer_token),
        )
        detail = client.get(f"/listings/{sample_listing.id}", headers=auth(reviewer_token)).json()
        assert detail["my_private_notes"] == "My secret notes"

    def test_update_private_notes_upsert(self, client, reviewer_token, sample_listing):
        client.put(f"/listings/{sample_listing.id}/private-notes", json={"notes": "v1"}, headers=auth(reviewer_token))
        client.put(f"/listings/{sample_listing.id}/private-notes", json={"notes": "v2"}, headers=auth(reviewer_token))
        detail = client.get(f"/listings/{sample_listing.id}", headers=auth(reviewer_token)).json()
        assert detail["my_private_notes"] == "v2"

    def test_private_notes_per_user(self, client, reviewer_token, admin_token, sample_listing):
        client.put(f"/listings/{sample_listing.id}/private-notes", json={"notes": "reviewer note"}, headers=auth(reviewer_token))
        client.put(f"/listings/{sample_listing.id}/private-notes", json={"notes": "admin note"}, headers=auth(admin_token))
        reviewer_detail = client.get(f"/listings/{sample_listing.id}", headers=auth(reviewer_token)).json()
        admin_detail = client.get(f"/listings/{sample_listing.id}", headers=auth(admin_token)).json()
        assert reviewer_detail["my_private_notes"] == "reviewer note"
        assert admin_detail["my_private_notes"] == "admin note"


# =============================================================================
# Collections
# =============================================================================

class TestCollections:
    """Tests for collection endpoints."""

    def test_list_collections_empty(self, client, reviewer_token):
        response = client.get("/listings/collections", headers=auth(reviewer_token))
        assert response.status_code == 200
        assert response.json() == []

    def test_create_collection(self, client, reviewer_token):
        response = client.post(
            "/listings/collections",
            json={"name": "Visit Saturday", "description": "Weekend viewing list"},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Visit Saturday"
        assert data["item_count"] == 0

    def test_create_duplicate_collection(self, client, reviewer_token, sample_collection):
        response = client.post(
            "/listings/collections",
            json={"name": "Klosters Shortlist"},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 409

    def test_get_collection_detail(self, client, reviewer_token, sample_collection, collection_with_item, sample_listing):
        response = client.get(
            f"/listings/collections/{sample_collection.id}",
            headers=auth(reviewer_token),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Klosters Shortlist"
        assert data["item_count"] == 1
        assert data["items"][0]["listing_id"] == str(sample_listing.id)

    def test_add_to_collection(self, client, reviewer_token, sample_collection, sample_listing):
        response = client.post(
            f"/listings/collections/{sample_collection.id}/items",
            json={"listing_id": str(sample_listing.id)},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 201

    def test_add_duplicate_to_collection(self, client, reviewer_token, sample_collection, sample_listing, collection_with_item):
        response = client.post(
            f"/listings/collections/{sample_collection.id}/items",
            json={"listing_id": str(sample_listing.id)},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 409

    def test_remove_from_collection(self, client, reviewer_token, collection_with_item):
        response = client.delete(
            f"/listings/collections/{collection_with_item.collection_id}/items/{collection_with_item.id}",
            headers=auth(reviewer_token),
        )
        assert response.status_code == 200


# =============================================================================
# Compare (GET /listings/compare)
# =============================================================================

class TestCompare:
    """Tests for compare endpoint."""

    def test_compare_two_listings(self, client, reviewer_token, multiple_listings):
        ids = f"{multiple_listings[0].id},{multiple_listings[1].id}"
        response = client.get("/listings/compare", params={"ids": ids}, headers=auth(reviewer_token))
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_compare_too_few(self, client, reviewer_token, multiple_listings):
        response = client.get(
            "/listings/compare", params={"ids": str(multiple_listings[0].id)},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 400

    def test_compare_too_many(self, client, reviewer_token, multiple_listings):
        ids = ",".join(str(l.id) for l in multiple_listings[:4])
        response = client.get("/listings/compare", params={"ids": ids}, headers=auth(reviewer_token))
        assert response.status_code == 400


# =============================================================================
# Map (GET /listings/map)
# =============================================================================

class TestMap:
    """Tests for map endpoint."""

    def test_get_map_data(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings/map", headers=auth(reviewer_token))
        assert response.status_code == 200
        data = response.json()
        # Only geocoded, non-archived listings
        assert all(item["latitude"] is not None for item in data)
        assert all(item["longitude"] is not None for item in data)

    def test_map_excludes_archived(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings/map", headers=auth(reviewer_token))
        data = response.json()
        assert all(item["listing_status"] != "archived" for item in data)


# =============================================================================
# Stats (GET /listings/stats)
# =============================================================================

class TestStats:
    """Tests for stats endpoint."""

    def test_get_stats(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings/stats", headers=auth(reviewer_token))
        assert response.status_code == 200
        data = response.json()
        assert data["total_listings"] == 5
        assert "by_status" in data
        assert "by_scenario" in data
        assert "by_source" in data
        assert "by_tier" in data
        assert data["by_status"]["scored"] == 1
        assert data["by_status"]["archived"] == 1

    def test_stats_empty(self, client, reviewer_token):
        response = client.get("/listings/stats", headers=auth(reviewer_token))
        assert response.status_code == 200
        assert response.json()["total_listings"] == 0


# =============================================================================
# Tags (GET /listings/tags)
# =============================================================================

class TestTags:
    """Tests for tags endpoint."""

    def test_get_tags(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings/tags", headers=auth(reviewer_token))
        assert response.status_code == 200
        data = response.json()
        assert "quiet" in data
        assert "urban" in data

    def test_tags_empty(self, client, reviewer_token):
        response = client.get("/listings/tags", headers=auth(reviewer_token))
        assert response.status_code == 200
        assert response.json() == []


# =============================================================================
# Export (GET /listings/export)
# =============================================================================

class TestExport:
    """Tests for export endpoint."""

    def test_export_csv(self, client, admin_token, multiple_listings):
        response = client.get("/listings/export", params={"format": "csv"}, headers=auth(admin_token))
        assert response.status_code == 200
        assert "text/csv" in response.headers["content-type"]
        content = response.content.decode("utf-8")
        lines = content.strip().split("\n")
        assert len(lines) == 6  # header + 5 listings

    def test_export_requires_admin(self, client, reviewer_token, multiple_listings):
        response = client.get("/listings/export", headers=auth(reviewer_token))
        assert response.status_code == 403


# =============================================================================
# Share Tokens
# =============================================================================

class TestShareTokens:
    """Tests for share token endpoints."""

    def test_create_share_token(self, client, reviewer_token, sample_listing):
        response = client.post(
            f"/listings/{sample_listing.id}/share-token",
            json={"expires_in_hours": 24, "can_view_reviews": True},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 201
        data = response.json()
        assert "token" in data
        assert "share_url" in data
        assert data["permissions"]["can_view_reviews"] is True

    def test_list_share_tokens(self, client, reviewer_token, sample_listing, sample_share_token):
        response = client.get(
            f"/listings/{sample_listing.id}/share-tokens", headers=auth(reviewer_token),
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["is_revoked"] is False

    def test_revoke_share_token(self, client, reviewer_token, sample_listing, sample_share_token):
        st, _ = sample_share_token
        response = client.delete(
            f"/listings/{sample_listing.id}/share-tokens/{st.id}",
            headers=auth(reviewer_token),
        )
        assert response.status_code == 200

    def test_shared_listing_public_access(self, client, sample_listing, sample_share_token):
        _, raw_token = sample_share_token
        response = client.get(f"/listings/shared/{raw_token}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == str(sample_listing.id)
        assert data["my_private_notes"] is None  # Never exposed
        assert "shared_at" in data
        assert "shared_by" in data

    def test_shared_listing_revoked(self, client, test_db, sample_listing, sample_share_token):
        st, raw_token = sample_share_token
        st.is_revoked = True
        test_db.commit()
        response = client.get(f"/listings/shared/{raw_token}")
        assert response.status_code == 410

    def test_shared_listing_invalid_token(self, client):
        response = client.get("/listings/shared/bogus_token_value")
        assert response.status_code == 404


# =============================================================================
# Info Requests (POST /listings/{id}/request-info)
# =============================================================================

class TestInfoRequests:
    """Tests for info request endpoint."""

    def test_request_info(self, client, admin_token, sample_listing):
        response = client.post(
            f"/listings/{sample_listing.id}/request-info",
            json={"message_template": "Please send documentation"},
            headers=auth(admin_token),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_request_info_updates_status(self, client, admin_token, sample_listing):
        # sample_listing starts as "scored"
        client.post(
            f"/listings/{sample_listing.id}/request-info",
            json={},
            headers=auth(admin_token),
        )
        detail = client.get(f"/listings/{sample_listing.id}", headers=auth(admin_token)).json()
        # scored (idx 3) < needs_info (idx 2) — wait, needs_info is idx 2 which is LOWER
        # Actually: needs_info is at index 2, scored is at index 3
        # So needs_info < scored — regression prevented, status stays "scored"
        assert detail["listing_status"] == "scored"

    def test_request_info_from_new(self, client, admin_token, test_db):
        """From 'new' status, request_info should transition to needs_info."""
        listing = PropertyListing(id=uuid4(), listing_status="new", plz="8000")
        test_db.add(listing)
        test_db.commit()
        client.post(f"/listings/{listing.id}/request-info", json={}, headers=auth(admin_token))
        detail = client.get(f"/listings/{listing.id}", headers=auth(admin_token)).json()
        assert detail["listing_status"] == "needs_info"

    def test_request_info_requires_admin(self, client, reviewer_token, sample_listing):
        response = client.post(
            f"/listings/{sample_listing.id}/request-info",
            json={},
            headers=auth(reviewer_token),
        )
        assert response.status_code == 403


# =============================================================================
# Integration: Full Workflow
# =============================================================================

class TestWorkflow:
    """End-to-end workflow tests."""

    def test_full_listing_workflow(self, client, admin_token, reviewer_token, test_db):
        """New listing → review → interested → request viewing → make offer."""
        # Create listing directly in DB
        listing = PropertyListing(
            id=uuid4(), address="Seestrasse 1", plz="8002",
            municipality="Zürich", price_chf=3000000, price_known=True,
            listing_status="new", tier=1,
        )
        test_db.add(listing)
        test_db.commit()
        lid = str(listing.id)

        # Reviewer submits review
        r = client.post(f"/listings/{lid}/review", json={"rating": 5, "comment": "Dream house"}, headers=auth(reviewer_token))
        assert r.status_code == 200

        # Admin marks interested
        r = client.post(f"/listings/{lid}/actions", json={"action": "interested"}, headers=auth(admin_token))
        assert r.json()["resulting_status"] == "under_evaluation"

        # Admin requests viewing
        r = client.post(f"/listings/{lid}/actions", json={"action": "request_viewing"}, headers=auth(admin_token))
        assert r.json()["resulting_status"] == "viewing_scheduled"

        # Admin makes offer
        r = client.post(f"/listings/{lid}/actions", json={"action": "make_offer", "notes": "CHF 2.8M"}, headers=auth(admin_token))
        assert r.json()["resulting_status"] == "offer_made"

        # Verify full state
        detail = client.get(f"/listings/{lid}", headers=auth(reviewer_token)).json()
        assert detail["listing_status"] == "offer_made"
        assert detail["num_ratings"] == 1
        assert detail["avg_rating"] == 5.0
        assert len(detail["actions"]) == 3

    def test_private_notes_isolation(self, client, admin_token, reviewer_token, test_db):
        """Each user's private notes are separate."""
        listing = PropertyListing(id=uuid4(), plz="8000", listing_status="new")
        test_db.add(listing)
        test_db.commit()
        lid = str(listing.id)

        client.put(f"/listings/{lid}/private-notes", json={"notes": "Admin note"}, headers=auth(admin_token))
        client.put(f"/listings/{lid}/private-notes", json={"notes": "Reviewer note"}, headers=auth(reviewer_token))

        admin_view = client.get(f"/listings/{lid}", headers=auth(admin_token)).json()
        reviewer_view = client.get(f"/listings/{lid}", headers=auth(reviewer_token)).json()

        assert admin_view["my_private_notes"] == "Admin note"
        assert reviewer_view["my_private_notes"] == "Reviewer note"
