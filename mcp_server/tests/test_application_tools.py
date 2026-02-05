"""
Unit tests for MCP Application Review tools.

Tests both the API client methods and server tool definitions.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import os

# Set required environment variables before importing
os.environ.setdefault("MCP_API_KEY", "test-key")
os.environ.setdefault("BACKEND_API_KEY", "test-backend-key")
os.environ.setdefault("EMAIL_API_URL", "http://localhost:8000")


class TestApplicationAPIClient:
    """Test EmailAPIClient application methods."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock API client."""
        with patch.dict(os.environ, {
            "MCP_API_KEY": "test-key",
            "BACKEND_API_KEY": "test-backend-key",
        }):
            from mcp_server.api_client import EmailAPIClient
            client = EmailAPIClient()
            client._request = AsyncMock()
            return client

    @pytest.mark.asyncio
    async def test_list_applications_basic(self, mock_client):
        """Test list_applications with no filters."""
        mock_client._request.return_value = {
            "total": 2,
            "page": 1,
            "page_size": 20,
            "applications": [
                {
                    "email_id": "uuid-1",
                    "applicant_name": "Alice Smith",
                    "applicant_institution": "MIT",
                    "date": "2024-01-15T10:00:00Z",
                    "category": "application-phd",
                    "scientific_excellence_score": 8,
                    "research_fit_score": 7,
                    "overall_recommendation_score": 8,
                },
                {
                    "email_id": "uuid-2",
                    "applicant_name": "Bob Jones",
                    "applicant_institution": "Stanford",
                    "date": "2024-01-14T10:00:00Z",
                    "category": "application-postdoc",
                    "scientific_excellence_score": 9,
                    "research_fit_score": 8,
                    "overall_recommendation_score": 9,
                }
            ]
        }

        result = await mock_client.list_applications()

        assert result["total"] == 2
        assert len(result["applications"]) == 2
        assert result["applications"][0]["applicant_name"] == "Alice Smith"
        assert result["applications"][1]["applicant_name"] == "Bob Jones"

    @pytest.mark.asyncio
    async def test_list_applications_with_filters(self, mock_client):
        """Test list_applications with various filters."""
        mock_client._request.return_value = {
            "total": 1,
            "page": 1,
            "page_size": 20,
            "applications": [
                {
                    "email_id": "uuid-1",
                    "applicant_name": "Alice Smith",
                    "category": "application-phd",
                    "overall_recommendation_score": 8,
                }
            ]
        }

        result = await mock_client.list_applications(
            category="application-phd",
            min_recommendation_score=7,
            search_name="Alice",
            profile_tags=["ml-experience", "genomics"],
            highest_degree=["Masters", "PhD"],
            sort_by="overall_recommendation_score",
            sort_order="desc"
        )

        # Verify the request was made with correct params
        call_args = mock_client._request.call_args
        params = call_args.kwargs["params"]

        assert params["category"] == "application-phd"
        assert params["min_recommendation_score"] == 7
        assert params["search_name"] == "Alice"
        assert params["profile_tags"] == "ml-experience,genomics"
        assert params["highest_degree"] == "Masters,PhD"
        assert params["sort_by"] == "overall_recommendation_score"
        assert params["sort_order"] == "desc"

    @pytest.mark.asyncio
    async def test_list_applications_date_filters(self, mock_client):
        """Test list_applications with date range filters."""
        mock_client._request.return_value = {
            "total": 0,
            "applications": []
        }

        await mock_client.list_applications(
            received_after="2024-01-01",
            received_before="2024-06-30"
        )

        call_args = mock_client._request.call_args
        params = call_args.kwargs["params"]

        assert params["received_after"] == "2024-01-01"
        assert params["received_before"] == "2024-06-30"

    @pytest.mark.asyncio
    async def test_list_applications_pagination(self, mock_client):
        """Test list_applications pagination."""
        mock_client._request.return_value = {
            "total": 100,
            "page": 3,
            "page_size": 10,
            "applications": []
        }

        result = await mock_client.list_applications(page=3, page_size=10)

        call_args = mock_client._request.call_args
        params = call_args.kwargs["params"]

        assert params["page"] == 3
        assert params["page_size"] == 10

    @pytest.mark.asyncio
    async def test_get_application_details(self, mock_client):
        """Test get_application_details returns full info."""
        mock_client._request.return_value = {
            "email_id": "uuid-1",
            "date": "2024-01-15T10:00:00Z",
            "subject": "PhD Application - Alice Smith",
            "applicant_name": "Alice Smith",
            "applicant_email": "alice@mit.edu",
            "applicant_institution": "MIT",
            "nationality": "USA",
            "highest_degree": "Masters",
            "category": "application-phd",
            "scientific_excellence_score": 8,
            "scientific_excellence_reason": "Strong publication record",
            "research_fit_score": 7,
            "research_fit_reason": "Good alignment with ML research",
            "overall_recommendation_score": 8,
            "recommendation_reason": "Excellent candidate overall",
            "coding_experience_score": 9,
            "coding_experience_evidence": "Multiple GitHub projects",
            "summary": "Strong ML researcher with publications",
            "key_strengths": ["Strong ML background", "Publications"],
            "concerns": ["Limited biology experience"],
            "github_account": "https://github.com/alicesmith",
            "folder_path": "folder-id-123",
            "email_text_link": "https://drive.google.com/...",
            "reviews": [
                {"rating": 4, "comment": "Good candidate"}
            ],
            "avg_rating": 4.0,
            "num_ratings": 1,
        }

        result = await mock_client.get_application_details("uuid-1")

        # Verify transformed structure
        assert result["email_id"] == "uuid-1"
        assert result["applicant"]["name"] == "Alice Smith"
        assert result["applicant"]["institution"] == "MIT"
        assert result["applicant"]["online_profiles"]["github"] == "https://github.com/alicesmith"
        assert result["scores"]["scientific_excellence"]["score"] == 8
        assert result["scores"]["scientific_excellence"]["reason"] == "Strong publication record"
        assert result["technical_experience"]["coding"]["score"] == 9
        assert result["ai_evaluation"]["summary"] == "Strong ML researcher with publications"
        assert result["google_drive"]["folder_id"] == "folder-id-123"
        assert result["reviews"]["avg_rating"] == 4.0

    @pytest.mark.asyncio
    async def test_get_application_details_error(self, mock_client):
        """Test get_application_details handles errors."""
        mock_client._request.return_value = {
            "error": "Application not found"
        }

        result = await mock_client.get_application_details("nonexistent-uuid")

        assert "error" in result
        assert result["error"] == "Application not found"

    @pytest.mark.asyncio
    async def test_get_application_available_tags(self, mock_client):
        """Test get_application_available_tags returns tags."""
        mock_client._request.return_value = {
            "tags": [
                {"name": "ml-experience", "count": 45},
                {"name": "genomics", "count": 32},
                {"name": "clinical", "count": 18},
            ]
        }

        result = await mock_client.get_application_available_tags()

        assert "tags" in result
        assert len(result["tags"]) == 3
        assert result["tags"][0]["name"] == "ml-experience"

    @pytest.mark.asyncio
    async def test_get_application_collections(self, mock_client):
        """Test get_application_collections returns collections."""
        mock_client._request.return_value = {
            "collections": [
                {"id": "coll-1", "name": "Fall 2024 PhD", "count": 25},
                {"id": "coll-2", "name": "Postdoc Applications", "count": 12},
            ]
        }

        result = await mock_client.get_application_collections()

        assert "collections" in result
        assert len(result["collections"]) == 2
        assert result["collections"][0]["name"] == "Fall 2024 PhD"


class TestApplicationTransformMethods:
    """Test the transform methods for application data."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock API client."""
        with patch.dict(os.environ, {
            "MCP_API_KEY": "test-key",
            "BACKEND_API_KEY": "test-backend-key",
        }):
            from mcp_server.api_client import EmailAPIClient
            return EmailAPIClient()

    def test_transform_application_list_item(self, mock_client):
        """Test _transform_application_list_item formats correctly."""
        app = {
            "email_id": "uuid-1",
            "applicant_name": "Alice Smith",
            "applicant_institution": "MIT",
            "date": "2024-01-15",
            "category": "application-phd",
            "scientific_excellence_score": 8,
            "research_fit_score": 7,
            "overall_recommendation_score": 8,
            "relevance_score": 6,
            "coding_experience_score": 9,
            "medical_data_experience_score": 3,
            "omics_genomics_experience_score": 5,
            "sequence_analysis_experience_score": 4,
            "image_analysis_experience_score": 6,
            "avg_rating": 4.0,
            "num_ratings": 2,
            "my_rating": 4,
            "status": "reviewed",
            "decision": "shortlist",
            "profile_tags": ["ml-experience"],
            "highest_degree": "Masters",
            "application_source": "direct",
            "folder_path": "folder-123",
            "email_text_link": "https://drive.google.com/..."
        }

        result = mock_client._transform_application_list_item(app)

        assert result["email_id"] == "uuid-1"
        assert result["applicant_name"] == "Alice Smith"
        assert result["scores"]["scientific_excellence"] == 8
        assert result["scores"]["research_fit"] == 7
        assert result["technical_scores"]["coding"] == 9
        assert result["review_summary"]["avg_rating"] == 4.0
        assert result["status"] == "reviewed"
        assert result["decision"] == "shortlist"

    def test_transform_application_detail(self, mock_client):
        """Test _transform_application_detail formats full details correctly."""
        app = {
            "email_id": "uuid-1",
            "date": "2024-01-15",
            "subject": "PhD Application",
            "message_id": "<msg@example.com>",
            "applicant_name": "Alice Smith",
            "applicant_email": "alice@mit.edu",
            "applicant_institution": "MIT",
            "nationality": "USA",
            "highest_degree": "Masters",
            "current_situation": "PhD student",
            "recent_thesis_title": "ML for Biology",
            "recommendation_source": "Website",
            "github_account": "https://github.com/alice",
            "linkedin_account": "https://linkedin.com/in/alice",
            "google_scholar_account": "https://scholar.google.com/...",
            "category": "application-phd",
            "subcategory": "ml",
            "confidence": 0.95,
            "application_source": "direct",
            "scientific_excellence_score": 8,
            "scientific_excellence_reason": "Strong publication record",
            "research_fit_score": 7,
            "research_fit_reason": "Good fit with ML focus",
            "overall_recommendation_score": 8,
            "recommendation_reason": "Excellent candidate",
            "relevance_score": 6,
            "relevance_reason": "Relevant background",
            "coding_experience_score": 9,
            "coding_experience_evidence": "GitHub projects",
            "omics_genomics_experience_score": 5,
            "omics_genomics_experience_evidence": "Some coursework",
            "medical_data_experience_score": 3,
            "medical_data_experience_evidence": "Limited",
            "sequence_analysis_experience_score": 4,
            "sequence_analysis_experience_evidence": "Basic experience",
            "image_analysis_experience_score": 7,
            "image_analysis_experience_evidence": "CV projects",
            "summary": "Strong ML researcher",
            "reasoning": "Good fit for the lab",
            "key_strengths": ["ML expertise", "Publications"],
            "concerns": ["Limited bio background"],
            "next_steps": "Schedule interview",
            "additional_notes": "Good communication",
            "profile_tags": ["ml-experience"],
            "is_mass_email": False,
            "no_research_background": False,
            "irrelevant_field": False,
            "insufficient_materials": False,
            "is_cold_email": False,
            "is_not_application": False,
            "is_not_application_reason": None,
            "folder_path": "folder-123",
            "email_text_link": "https://drive.google.com/email",
            "llm_response_link": "https://drive.google.com/llm",
            "attachments_list": ["CV.pdf", "Publications.pdf"],
            "consolidated_attachments": ["All_Materials.pdf"],
            "reference_letter_attachments": ["Ref1.pdf"],
            "avg_rating": 4.0,
            "num_ratings": 2,
            "reviews": [{"rating": 4, "comment": "Good"}],
            "decision": "shortlist",
            "needs_reply": True,
            "reply_deadline": "2024-02-01",
            "reply_suggestion": "Request interview",
            "action_items": ["Schedule call"],
            "suggested_folder": "Interviews",
            "suggested_labels": ["priority"],
            "answer_options": ["Accept", "Reject"],
            "should_request_additional_info": False,
            "missing_information_items": [],
            "potential_recommendation_score": 9
        }

        result = mock_client._transform_application_detail(app)

        # Check structure
        assert "applicant" in result
        assert "classification" in result
        assert "scores" in result
        assert "technical_experience" in result
        assert "ai_evaluation" in result
        assert "google_drive" in result
        assert "reviews" in result
        assert "status" in result
        assert "additional_info" in result

        # Check nested values
        assert result["applicant"]["name"] == "Alice Smith"
        assert result["applicant"]["online_profiles"]["github"] == "https://github.com/alice"
        assert result["scores"]["scientific_excellence"]["score"] == 8
        assert result["scores"]["scientific_excellence"]["reason"] == "Strong publication record"
        assert result["technical_experience"]["coding"]["score"] == 9
        assert result["ai_evaluation"]["key_strengths"] == ["ML expertise", "Publications"]
        assert result["google_drive"]["attachments"] == ["CV.pdf", "Publications.pdf"]
        assert result["reviews"]["decision"] == "shortlist"


class TestServerToolDefinitions:
    """Test that server tool definitions are correct."""

    def test_application_tools_defined(self):
        """Test that all application tools are defined in server."""
        # Can't easily import server due to mcp dependency, so check file content
        import pathlib
        server_path = pathlib.Path(__file__).parent.parent / "server.py"
        content = server_path.read_text()

        # Check tool definitions exist
        assert 'name="list_applications"' in content
        assert 'name="get_application_details"' in content
        assert 'name="get_application_tags"' in content
        assert 'name="get_application_collections"' in content

    def test_application_handlers_defined(self):
        """Test that all application handlers are defined in server."""
        import pathlib
        server_path = pathlib.Path(__file__).parent.parent / "server.py"
        content = server_path.read_text()

        # Check handlers exist
        assert 'elif name == "list_applications":' in content
        assert 'elif name == "get_application_details":' in content
        assert 'elif name == "get_application_tags":' in content
        assert 'elif name == "get_application_collections":' in content

    def test_list_applications_schema_has_filters(self):
        """Test list_applications tool has all filter properties."""
        import pathlib
        server_path = pathlib.Path(__file__).parent.parent / "server.py"
        content = server_path.read_text()

        # Check key filter properties are in the schema
        expected_properties = [
            '"category"',
            '"min_recommendation_score"',
            '"min_excellence_score"',
            '"min_research_fit_score"',
            '"search_name"',
            '"received_after"',
            '"received_before"',
            '"application_status"',
            '"has_decision"',
            '"profile_tags"',
            '"highest_degree"',
            '"sort_by"',
            '"sort_order"',
            '"page"',
            '"page_size"',
        ]

        for prop in expected_properties:
            assert prop in content, f"Missing property {prop} in list_applications schema"


class TestApplicationToolIntegration:
    """Integration-style tests for application tools (with mocked backend)."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock API client."""
        with patch.dict(os.environ, {
            "MCP_API_KEY": "test-key",
            "BACKEND_API_KEY": "test-backend-key",
        }):
            from mcp_server.api_client import EmailAPIClient
            client = EmailAPIClient()
            client._request = AsyncMock()
            return client

    @pytest.mark.asyncio
    async def test_list_applications_calls_correct_endpoint(self, mock_client):
        """Test list_applications calls the correct API endpoint."""
        mock_client._request.return_value = {
            "total": 0,
            "applications": []
        }

        await mock_client.list_applications(category="application-phd")

        # Verify endpoint called
        call_args = mock_client._request.call_args
        assert call_args[0][0] == "GET"  # method
        assert call_args[0][1] == "/applications"  # endpoint

    @pytest.mark.asyncio
    async def test_get_application_details_calls_correct_endpoint(self, mock_client):
        """Test get_application_details calls the correct API endpoint."""
        mock_client._request.return_value = {
            "email_id": "test-uuid",
            "applicant_name": "Test User"
        }

        await mock_client.get_application_details("test-uuid")

        # Verify endpoint called
        call_args = mock_client._request.call_args
        assert call_args[0][0] == "GET"  # method
        assert call_args[0][1] == "/applications/test-uuid"  # endpoint

    @pytest.mark.asyncio
    async def test_get_application_tags_calls_correct_endpoint(self, mock_client):
        """Test get_application_available_tags calls the correct API endpoint."""
        mock_client._request.return_value = {"tags": []}

        await mock_client.get_application_available_tags()

        call_args = mock_client._request.call_args
        assert call_args[0][0] == "GET"
        assert call_args[0][1] == "/applications/available-tags"

    @pytest.mark.asyncio
    async def test_get_application_collections_calls_correct_endpoint(self, mock_client):
        """Test get_application_collections calls the correct API endpoint."""
        mock_client._request.return_value = {"collections": []}

        await mock_client.get_application_collections()

        call_args = mock_client._request.call_args
        assert call_args[0][0] == "GET"
        assert call_args[0][1] == "/collections"
