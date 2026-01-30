"""
Test email models (Pydantic validation).
Ensures models match inbox-zero patterns.
"""
import pytest
from datetime import datetime
from backend.core.email.models import (
    Email,
    ProcessedEmail,
    EmailCategory,
    SenderType,
    ReplyStatus,
    AppleMailColor,
    EmailAction,
    ActionResult,
    ClassificationResult,
    AttachmentInfo,
)
from backend.core.email.annotations import EmailMetadata, EmailNote
from pydantic import ValidationError


class TestEmailModels:
    """Test Pydantic email models"""
    
    def test_email_model_creation(self):
        """Test creating Email model"""
        email = Email(
            uid="123",
            message_id="<test@example.com>",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            date=datetime.now(),
            body_text="Test body"
        )
        
        assert email.uid == "123"
        assert email.subject == "Test"
        assert len(email.to_addresses) == 1
    
    def test_processed_email_model(self):
        """Test ProcessedEmail model"""
        processed = ProcessedEmail(
            uid="456",
            message_id="<test@example.com>",
            subject="Test",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            date=datetime.now(),
            body_markdown="**Test** body",
            sender_domain="example.com"
        )
        
        assert processed.uid == "456"
        assert processed.body_markdown == "**Test** body"
        assert processed.sender_domain == "example.com"
        assert processed.has_attachments == False
    
    def test_email_category_enum(self):
        """Test EmailCategory enum values"""
        assert EmailCategory.PERSONAL == "personal"
        assert EmailCategory.WORK == "work"
        assert EmailCategory.RECEIPT == "receipt"
        assert EmailCategory.NEWSLETTER == "newsletter"
    
    def test_email_action_validation(self):
        """Test EmailAction validation"""
        # Valid move action
        action = EmailAction(type="move", folder="Archive")
        assert action.folder == "Archive"
        assert action.is_valid == True
        
        # Invalid: move without folder
        with pytest.raises(ValueError, match="folder is required"):
            EmailAction(type="move")
        
        # Valid forward
        action = EmailAction(type="forward", forward_to="test@example.com")
        assert action.forward_to == "test@example.com"
        assert action.is_valid == True
        
        # Invalid forward without to
        with pytest.raises(ValueError, match="forward_to is required"):
            EmailAction(type="forward")
        
        # Valid color action
        action = EmailAction(type="color", color=AppleMailColor.RED)
        assert action.type == "color"
        assert action.is_valid == True
    
    def test_classification_result(self):
        """Test classification result with all required fields"""
        result = ClassificationResult(
            category=EmailCategory.PERSONAL,
            sender_type=SenderType.KNOWN,
            reply_status=ReplyStatus.NO_ACTION,
            is_urgent=False,
            confidence=0.95,
            reasoning="Personal greeting detected",
            action=EmailAction(type="keep")
        )
        assert result.category == EmailCategory.PERSONAL
        assert result.confidence == 0.95
        assert result.reasoning == "Personal greeting detected"
        
        # Edge: low confidence
        low_conf = ClassificationResult(
            category=EmailCategory.WORK,
            sender_type=SenderType.NEW,
            reply_status=ReplyStatus.NEEDS_REPLY,
            is_urgent=False,
            confidence=0.1,
            reasoning="Uncertain",
            action=EmailAction(type="keep")
        )
        assert low_conf.confidence == 0.1
    
    def test_classification_result_model(self):
        """Test ClassificationResult model"""
        result = ClassificationResult(
            category=EmailCategory.WORK,
            sender_type=SenderType.KNOWN,
            reply_status=ReplyStatus.NEEDS_REPLY,
            is_urgent=True,
            confidence=0.95,
            reasoning="Email from colleague about urgent deadline",
            action=EmailAction(type="color", color=AppleMailColor.RED)
        )
        
        assert result.category == EmailCategory.WORK
        assert result.is_urgent == True
        assert result.confidence == 0.95
        assert result.action.type == "color"
    
    def test_apple_mail_color_enum(self):
        """Test Apple Mail color codes"""
        assert AppleMailColor.RED == 1
        assert AppleMailColor.ORANGE == 2
        assert AppleMailColor.YELLOW == 3
        assert AppleMailColor.GREEN == 4
        assert AppleMailColor.BLUE == 5
        assert AppleMailColor.PURPLE == 6
        assert AppleMailColor.GRAY == 7
    
    def test_action_result_model(self):
        """Test ActionResult model"""
        result = ActionResult(
            success=True,
            dry_run=True,
            description="Would move email to Archive",
            reversible=True
        )
        
        assert result.success == True
        assert result.dry_run == True
        assert result.reversible == True

    def test_attachment_info(self):
        attach = AttachmentInfo(
            filename="doc.pdf",
            content_type="application/pdf",
            size=1024,
            extracted_text="Sample text"
        )
        assert attach.filename == "doc.pdf"
        assert attach.size == 1024
        
        # Empty text
        empty = AttachmentInfo(
            filename="empty.txt",
            content_type="text/plain",
            size=0,
            extracted_text=""
        )
        assert empty.extracted_text == ""

    def test_email_metadata(self):
        metadata = EmailMetadata(
            email_uid="123",
            ai_summary="Test summary",
            ai_sentiment="positive",
            ai_action_items=["Task 1", "Task 2"],
            classification_reasoning="Based on content",
            classification_confidence=0.85,
            user_notes=["Note 1"],
            user_tags=["tag1", "tag2"],
            related_project="ProjectX",
            priority_level=3,
            contains_pii=True,
            contains_sensitive=False
        )
        
        assert metadata.email_uid == "123"
        assert len(metadata.ai_action_items) == 2
        assert metadata.priority_level == 3
        assert metadata.contains_pii is True
        
        # Invalid priority
        with pytest.raises(ValidationError):
            EmailMetadata(email_uid="123", priority_level=6)
        
        # Invalid confidence
        with pytest.raises(ValidationError):
            EmailMetadata(email_uid="123", classification_confidence=1.5)

    def test_email_note(self):
        note = EmailNote(
            email_uid="456",
            note_type="user_note",
            content="Detailed note",
            created_by="user",
            metadata={"key": "value"},
            private=True,
            visible_in_mail=False
        )
        
        assert note.email_uid == "456"
        assert note.private is True
        assert note.metadata["key"] == "value"
        
        # Required fields
        with pytest.raises(ValidationError):
            EmailNote(email_uid="456")  # Missing note_type and content

