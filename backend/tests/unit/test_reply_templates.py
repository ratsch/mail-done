"""
Unit Tests for Reply Templates

Tests template-based reply generation.
"""
import pytest
from backend.core.replies.templates import ReplyTemplates


class TestReplyTemplates:
    """Test suite for ReplyTemplates"""
    
    @pytest.fixture
    def templates(self):
        """Create templates instance"""
        return ReplyTemplates()
    
    # =========================================================================
    # Template Retrieval Tests
    # =========================================================================
    
    def test_get_phd_accept_template(self, templates):
        """Test retrieving PhD acceptance template"""
        template = templates.get_template(
            category="application-phd",
            decision="accept",
            tone="enthusiastic"
        )
        
        assert template is not None
    
    def test_get_phd_decline_template(self, templates):
        """Test retrieving PhD decline template"""
        template = templates.get_template(
            category="application-phd",
            decision="decline",
            tone="polite"
        )
        
        assert template is not None
    
    def test_get_invalid_template(self, templates):
        """Test that invalid template returns None"""
        template = templates.get_template(
            category="invalid-category",
            decision="accept",
            tone="friendly"
        )
        
        assert template is None
    
    def test_fallback_to_first_variant(self, templates):
        """Test fallback when tone doesn't exist"""
        template = templates.get_template(
            category="application-phd",
            decision="accept",
            tone="invalid-tone"
        )
        
        # Should fall back to first variant
        assert template is not None
    
    # =========================================================================
    # Subject Line Tests
    # =========================================================================
    
    def test_subject_line_generation(self, templates):
        """Test subject line generation"""
        subject = templates.get_subject(
            category="application-phd",
            decision="accept",
            original_subject="PhD Application - Machine Learning"
        )
        
        assert subject == "Re: PhD Application - Machine Learning"
    
    # =========================================================================
    # Variable Substitution Tests
    # =========================================================================
    
    def test_generate_phd_accept_with_context(self, templates):
        """Test generating PhD acceptance with context"""
        body = templates.generate_from_template(
            category="application-phd",
            decision="accept",
            tone="enthusiastic",
            context={
                'applicant_name': 'Jane Smith',
                'research_area': 'machine learning',
                'specific_detail': 'your work on deep learning for genomics'
            }
        )
        
        assert body is not None
        assert 'Jane Smith' in body
        assert 'machine learning' in body
        assert 'deep learning for genomics' in body
    
    def test_generate_with_missing_variables(self, templates):
        """Test generation with missing variables (safe_substitute)"""
        body = templates.generate_from_template(
            category="application-phd",
            decision="accept",
            tone="enthusiastic",
            context={
                'applicant_name': 'John Doe'
                # Missing research_area and specific_detail
            }
        )
        
        assert body is not None
        assert 'John Doe' in body
        # Should have placeholder text for missing vars
        assert body is not None
    
    def test_generate_invitation_accept(self, templates):
        """Test invitation acceptance template"""
        body = templates.generate_from_template(
            category="invitation-speaking",
            decision="accept",
            tone="confirm",
            context={
                'organizer_name': 'Dr. Smith',
                'event_name': 'ICML 2026',
                'event_date': 'July 15, 2026',
                'topic': 'deep learning for genomics',
                'duration': '45 minutes'
            }
        )
        
        assert body is not None
        assert 'Dr. Smith' in body
        assert 'ICML 2026' in body
        assert 'July 15, 2026' in body
    
    def test_generate_review_decline(self, templates):
        """Test review decline template"""
        body = templates.generate_from_template(
            category="review-peer-journal",
            decision="decline",
            tone="capacity",
            context={
                'editor_name': 'Dr. Johnson',
                'manuscript_id': 'BIOINF-2025-123',
                'journal_name': 'Bioinformatics',
                'alternative_reviewers': 'Dr. Lee, Dr. Brown'
            }
        )
        
        assert body is not None
        assert 'Dr. Johnson' in body
        assert 'BIOINF-2025-123' in body
        assert 'overcommitted' in body or 'capacity' in body.lower()
    
    # =========================================================================
    # Available Templates Tests
    # =========================================================================
    
    def test_list_available_templates(self, templates):
        """Test listing all available templates"""
        available = templates.list_available_templates()
        
        assert len(available) > 0
        assert 'application-phd-accept' in available
        assert 'invitation-speaking-decline' in available
    
    def test_get_available_tones(self, templates):
        """Test getting available tones for a template"""
        tones = templates.get_available_tones(
            category="application-phd",
            decision="accept"
        )
        
        assert len(tones) > 0
        assert 'enthusiastic' in tones
        assert 'cautious' in tones
    
    # =========================================================================
    # Edge Case Tests
    # =========================================================================
    
    def test_empty_context(self, templates):
        """Test generation with empty context"""
        body = templates.generate_from_template(
            category="general",
            decision="acknowledge",
            tone="received",
            context={}
        )
        
        # Should still generate something with placeholders
        assert body is not None
    
    def test_special_characters_in_context(self, templates):
        """Test handling special characters"""
        body = templates.generate_from_template(
            category="work-colleague",
            decision="acknowledge",
            tone="confirm",
            context={
                'colleague_name': "O'Brien",
                'topic': "DNA & RNA analysis",
                'response': "I'll review it"
            }
        )
        
        assert body is not None
        assert "O'Brien" in body

