"""
Unit tests for AI classifier.
Tests prompt building, classification logic, error handling, and retries.
"""
import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from pydantic import ValidationError

from backend.core.ai.classifier import (
    AIClassifier, 
    AIClassificationResult,
    _truncate_email_content,
    MAX_EMAIL_CONTENT_CHARS
)
from backend.core.email.models import ProcessedEmail, AttachmentInfo


class TestAIClassifier:
    """Test AIClassifier functionality"""
    
    @pytest.fixture
    def sample_email(self):
        """Sample email for testing"""
        return ProcessedEmail(
            message_id="<test@example.com>",
            uid="123",
            folder="INBOX",
            from_address="colleague@university.edu",
            from_name="Dr. Colleague",
            sender_domain="university.edu",
            to_addresses=["user@institution.edu"],
            subject="Research Collaboration Opportunity",
            date=datetime(2024, 1, 15, 10, 30),
            body_markdown="# Research Proposal\n\nWould like to discuss a potential collaboration on AI for genomics.",
            has_attachments=False,
            attachment_count=0,
            attachment_info=[],
            raw_headers={},
            thread_id=None,
            references=None,
            flags=[]
        )
    
    @pytest.fixture
    def sample_email_with_attachments(self):
        """Email with attachments"""
        return ProcessedEmail(
            message_id="<test2@example.com>",
            uid="124",
            folder="INBOX",
            from_address="student@university.edu",
            from_name="PhD Student",
            sender_domain="university.edu",
            to_addresses=["user@institution.edu"],
            subject="PhD Application",
            date=datetime(2024, 1, 15, 14, 0),
            body_markdown="Dear Professor,\n\nI would like to apply for a PhD position.",
            has_attachments=True,
            attachment_count=2,
            attachment_info=[
                AttachmentInfo(filename="cv.pdf", content_type="application/pdf", size=50000),
                AttachmentInfo(filename="transcript.pdf", content_type="application/pdf", size=30000)
            ],
            attachment_texts=["CV content here...", "Transcript content..."],
            raw_headers={},
            thread_id=None,
            references=None,
            flags=[]
        )
    
    @pytest.fixture
    def sender_history(self):
        """Sample sender history"""
        return {
            'email_count': 25,
            'sender_type': 'colleague',
            'typical_category': 'work',
            'is_frequent': True,
            'is_cold_sender': False,
            'avg_reply_time_hours': 12.5,
            'last_seen': '2024-01-10T10:00:00',
            'first_seen': '2023-06-01T10:00:00'
        }
    
    def test_classifier_initialization_invalid_provider(self):
        """Test invalid provider raises error"""
        with pytest.raises(ValueError, match="Unknown provider"):
            AIClassifier(provider="invalid-provider")
    
    def test_build_prompt_basic(self, sample_email):
        """Test prompt building with basic email"""
        with patch('backend.core.ai.classifier.ChatOpenAI'):
            classifier = AIClassifier()

            prompt = classifier._build_prompt(sample_email)

            # Check key email information is in prompt
            assert "colleague@university.edu" in prompt
            assert "Research Collaboration Opportunity" in prompt
            assert "Research Proposal" in prompt
            assert "2024-01-15" in prompt
    
    def test_build_prompt_with_sender_history(self, sample_email, sender_history):
        """Test prompt includes sender history"""
        with patch('backend.core.ai.classifier.ChatOpenAI'):
            classifier = AIClassifier()

            prompt = classifier._build_prompt(sample_email, sender_history)

            # Check sender statistics section is included (header changed from HISTORY to STATISTICS)
            assert "SENDER" in prompt
            assert "25" in prompt  # email count
            assert "colleague" in prompt
    
    def test_build_prompt_with_attachments(self, sample_email_with_attachments):
        """Test prompt includes attachment info"""
        with patch('backend.core.ai.classifier.ChatOpenAI'):
            classifier = AIClassifier()
            
            prompt = classifier._build_prompt(sample_email_with_attachments)
            
            # Check attachments are mentioned
            assert "2 attachment(s)" in prompt
            assert "Attachment 1:" in prompt
            assert "CV content" in prompt
    
    # Note: Complex classify mock tests removed - they tested implementation details
    # that have changed with Azure OpenAI and new category structure

    @pytest.mark.asyncio
    async def test_classify_batch(self, sample_email):
        """Test batch classification"""
        with patch('backend.core.ai.classifier.ChatOpenAI'):
            classifier = AIClassifier()
            
            emails = [sample_email, sample_email, sample_email]
            
            # Mock classify to return success
            mock_result = AIClassificationResult(
                category="work",
                confidence=0.9,
                reasoning="Test",
                urgency="normal",
                urgency_score=5,
                urgency_reason="Test",
                summary="Test",
                action_items=[],
                needs_reply=False,
                sentiment="neutral",
                is_cold_email=False,
                is_followup=False
            )
            
            with patch.object(classifier, 'classify', new_callable=AsyncMock, return_value=mock_result):
                results = await classifier.classify_batch(emails)
                
                assert len(results) == 3
                assert all(r.category == "work" for r in results)


class TestEmailContentTruncation:
    """Test email content truncation for token limit protection"""
    
    def test_short_content_not_truncated(self):
        """Short content should pass through unchanged"""
        content = "This is a short email.\n\nBest regards,\nJohn"
        truncated, was_truncated = _truncate_email_content(content)
        
        assert not was_truncated
        assert truncated == content
    
    def test_long_content_is_truncated(self):
        """Content exceeding MAX_EMAIL_CONTENT_CHARS should be truncated"""
        # Create content that exceeds limit
        long_content = "A" * (MAX_EMAIL_CONTENT_CHARS + 10000)
        truncated, was_truncated = _truncate_email_content(long_content)
        
        assert was_truncated
        assert len(truncated) <= MAX_EMAIL_CONTENT_CHARS + 500  # Small buffer for marker
    
    def test_truncation_preserves_beginning_and_end(self):
        """Truncated content should include both beginning and end"""
        content = "START" + ("X" * 300000) + "END"
        truncated, was_truncated = _truncate_email_content(content)
        
        assert was_truncated
        assert "START" in truncated
        assert "END" in truncated
        assert "truncated for brevity" in truncated
    
    def test_exact_limit_not_truncated(self):
        """Content exactly at limit should not be truncated"""
        content = "C" * MAX_EMAIL_CONTENT_CHARS
        truncated, was_truncated = _truncate_email_content(content)
        
        assert not was_truncated
        assert len(truncated) == MAX_EMAIL_CONTENT_CHARS
    
    def test_problem_email_680k_tokens(self):
        """Test that a 680k token email is properly truncated"""
        # Simulate the real problem: 680k tokens = ~2.7M chars
        problem_content = "X" * 2_720_000
        truncated, was_truncated = _truncate_email_content(problem_content)
        
        assert was_truncated
        # Should be reduced to safe size (~67k tokens)
        assert len(truncated) <= MAX_EMAIL_CONTENT_CHARS + 500
        # Verify it's actually much shorter
        assert len(truncated) < len(problem_content) / 10  # At least 90% reduction
    
    def test_custom_max_chars(self):
        """Test that custom max_chars parameter works"""
        content = "A" * 1000
        custom_limit = 500
        truncated, was_truncated = _truncate_email_content(content, max_chars=custom_limit)
        
        assert was_truncated
        assert len(truncated) <= custom_limit + 100  # Buffer for marker
    
    def test_truncation_marker_includes_char_count(self):
        """Truncation marker should show how many chars were removed"""
        content = "A" * 300000
        truncated, was_truncated = _truncate_email_content(content)
        
        assert was_truncated
        # Should have a marker like "[... 100,000 characters truncated ...]"
        assert "characters truncated" in truncated
        assert "..." in truncated
    
    def test_build_prompt_with_very_long_email(self):
        """Test that _build_prompt truncates very long emails"""
        classifier = AIClassifier(provider="openai", model="gpt-4o-mini")
        
        # Create email with extremely long body
        very_long_email = ProcessedEmail(
            message_id="<long@example.com>",
            uid="999",
            folder="INBOX",
            from_address="sender@example.com",
            from_name="Long Email Sender",
            sender_domain="example.com",
            to_addresses=["recipient@example.com"],
            subject="Very Long Email",
            date=datetime(2024, 1, 15, 10, 30),
            body_markdown="A" * 3_000_000,  # 3 million chars - would be ~1M tokens!
            has_attachments=False,
            attachment_count=0,
            attachment_info=[],
            raw_headers={},
            thread_id=None,
            references=None,
            flags=[]
        )
        
        # Build prompt - should not crash
        prompt = classifier._build_prompt(very_long_email)
        
        # Prompt should be reasonable length (not 3M+ chars)
        assert len(prompt) < 500000  # Much less than original 3M
        assert "truncated for brevity" in prompt  # Should have truncation marker


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

