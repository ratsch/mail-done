"""
Test email annotations and metadata system.
"""
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
from backend.core.email.annotations import (
    AnnotationManager,
    EmailMetadata,
    EmailNote,
    IMAPKeywords
)


class TestEmailNote:
    """Test EmailNote model"""
    
    def test_email_note_creation(self):
        """Test creating an email note"""
        note = EmailNote(
            email_uid="123",
            note_type="user_note",
            content="This needs follow-up"
        )
        
        assert note.email_uid == "123"
        assert note.note_type == "user_note"
        assert note.content == "This needs follow-up"
        assert note.created_by == "system"
        assert note.private == False
        assert note.visible_in_mail == False
    
    def test_email_note_with_metadata(self):
        """Test note with additional metadata"""
        note = EmailNote(
            email_uid="456",
            note_type="ai_summary",
            content="Meeting request from colleague",
            created_by="ai",
            metadata={"confidence": 0.95, "model": "gpt-4"},
            private=True
        )
        
        assert note.metadata["confidence"] == 0.95
        assert note.private == True


class TestEmailMetadata:
    """Test EmailMetadata model"""
    
    def test_email_metadata_creation(self):
        """Test creating email metadata"""
        metadata = EmailMetadata(
            email_uid="789",
            ai_summary="Project update email",
            ai_sentiment="positive",
            ai_action_items=["Review document", "Schedule meeting"],
            classification_reasoning="Work email with deadline",
            classification_confidence=0.92
        )
        
        assert metadata.email_uid == "789"
        assert metadata.ai_summary == "Project update email"
        assert metadata.ai_sentiment == "positive"
        assert len(metadata.ai_action_items) == 2
        assert metadata.classification_confidence == 0.92
    
    def test_email_metadata_defaults(self):
        """Test metadata with default values"""
        metadata = EmailMetadata(email_uid="001")
        
        assert metadata.email_uid == "001"
        assert metadata.ai_summary is None
        assert metadata.ai_action_items == []
        assert metadata.user_notes == []
        assert metadata.user_tags == []
        assert metadata.contains_pii == False
        assert metadata.contains_sensitive == False
    
    def test_priority_validation(self):
        """Test priority level validation (1-5)"""
        metadata = EmailMetadata(email_uid="002", priority_level=5)
        assert metadata.priority_level == 5
        
        # Test invalid priority
        with pytest.raises(ValueError):
            EmailMetadata(email_uid="003", priority_level=10)
    
    def test_pii_tracking(self):
        """Test PII tracking"""
        metadata = EmailMetadata(
            email_uid="004",
            contains_pii=True,
            contains_sensitive=True
        )
        
        assert metadata.contains_pii == True
        assert metadata.contains_sensitive == True


class TestAnnotationManager:
    """Test AnnotationManager functionality"""
    
    def test_initialization(self):
        """Test annotation manager initialization"""
        manager = AnnotationManager()
        
        assert manager.imap is None
        assert manager.database is None
        assert manager.pending_metadata == {}
    
    def test_initialization_with_imap(self, mock_imap_client):
        """Test initialization with IMAP monitor"""
        manager = AnnotationManager(imap_monitor=mock_imap_client)
        
        assert manager.imap == mock_imap_client
    
    def test_add_simple_note_without_imap(self):
        """Test adding simple note without IMAP connection"""
        manager = AnnotationManager()
        
        result = manager.add_simple_note("123", "Important")
        
        assert result == False  # Should fail without IMAP
    
    def test_add_simple_note_with_imap(self, mock_imap_client):
        """Test adding simple note with IMAP"""
        manager = AnnotationManager(imap_monitor=mock_imap_client)
        mock_imap_client.add_custom_flag = Mock(return_value=True)
        
        result = manager.add_simple_note("123", "Follow Up")
        
        assert result == True
        mock_imap_client.add_custom_flag.assert_called_with("123", "Follow_Up")
    
    def test_add_detailed_metadata_to_memory(self):
        """Test adding detailed metadata (no database)"""
        manager = AnnotationManager()
        
        metadata = EmailMetadata(
            email_uid="456",
            ai_summary="Test summary",
            classification_confidence=0.88
        )
        
        result = manager.add_detailed_metadata(metadata)
        
        assert result == True
        assert "456" in manager.pending_metadata
        assert manager.pending_metadata["456"].ai_summary == "Test summary"
    
    def test_add_ai_generated_note(self):
        """Test adding AI-generated annotations"""
        manager = AnnotationManager()
        
        result = manager.add_ai_generated_note(
            email_uid="789",
            summary="Meeting invitation",
            action_items=["Accept invite", "Add to calendar"],
            classification_reason="Work email with calendar attachment",
            entities={"people": ["John Doe"], "orgs": ["ACME Corp"]}
        )
        
        assert result == True
        metadata = manager.get_metadata("789")
        assert metadata is not None
        assert metadata.ai_summary == "Meeting invitation"
        assert len(metadata.ai_action_items) == 2
        assert "John Doe" in metadata.ai_entities["people"]
    
    def test_add_user_note(self):
        """Test adding user note"""
        manager = AnnotationManager()
        
        result = manager.add_user_note(
            email_uid="101",
            note="Need to follow up by Friday",
            visible_in_mail=False,
            tags=["urgent", "follow_up"]
        )
        
        assert result == True
        metadata = manager.get_metadata("101")
        assert "Need to follow up by Friday" in metadata.user_notes
        assert "urgent" in metadata.user_tags
        assert "follow_up" in metadata.user_tags
    
    def test_mark_contains_pii(self):
        """Test marking email as containing PII"""
        manager = AnnotationManager()
        
        result = manager.mark_contains_pii(
            email_uid="202",
            pii_types=["ssn", "address"]
        )
        
        assert result == True
        metadata = manager.get_metadata("202")
        assert metadata.contains_pii == True
    
    def test_set_project_context(self, mock_imap_client):
        """Test setting project context"""
        manager = AnnotationManager(imap_monitor=mock_imap_client)
        mock_imap_client.add_custom_flag = Mock(return_value=True)
        
        deadline = datetime.now() + timedelta(days=14)
        
        result = manager.set_project_context(
            email_uid="303",
            project="AIResearch",
            priority=4,
            deadline=deadline
        )
        
        assert result == True
        metadata = manager.get_metadata("303")
        assert metadata.related_project == "AIResearch"
        assert metadata.priority_level == 4
        assert metadata.deadline == deadline
        
        # Should also add IMAP flag
        mock_imap_client.add_custom_flag.assert_called()
    
    def test_get_metadata_not_found(self):
        """Test getting metadata for non-existent email"""
        manager = AnnotationManager()
        
        result = manager.get_metadata("nonexistent")
        
        assert result is None
    
    def test_export_metadata_for_database(self):
        """Test exporting metadata for database migration"""
        manager = AnnotationManager()
        
        # Add some metadata
        manager.add_ai_generated_note("001", summary="Summary 1")
        manager.add_ai_generated_note("002", summary="Summary 2")
        manager.add_ai_generated_note("003", summary="Summary 3")
        
        exported = manager.export_metadata_for_database()
        
        assert len(exported) == 3
        assert all(isinstance(m, EmailMetadata) for m in exported)
    
    def test_clear_pending(self):
        """Test clearing pending metadata"""
        manager = AnnotationManager()
        
        manager.add_ai_generated_note("001", summary="Test")
        manager.add_ai_generated_note("002", summary="Test")
        
        assert len(manager.pending_metadata) == 2
        
        manager.clear_pending()
        
        assert len(manager.pending_metadata) == 0


class TestIMAPKeywords:
    """Test IMAP keywords helper"""
    
    def test_predefined_keywords(self):
        """Test predefined IMAP keywords"""
        assert IMAPKeywords.IMPORTANT == "$Important"
        assert IMAPKeywords.FOLLOW_UP == "$FollowUp"
        assert IMAPKeywords.REVIEWED == "$Reviewed"
        assert IMAPKeywords.WAITING_REPLY == "$WaitingReply"
        assert IMAPKeywords.DELEGATED == "$Delegated"
    
    def test_project_keyword_generation(self):
        """Test generating project keywords"""
        keyword = IMAPKeywords.project("AI Research")
        
        assert keyword == "$Project_AI_Research"
    
    def test_project_keyword_sanitization(self):
        """Test project keyword with special characters"""
        keyword = IMAPKeywords.project("Project-Name With Spaces")
        
        # Should replace spaces and dashes with underscores
        assert "_" in keyword
        assert " " not in keyword
        assert "-" not in keyword
    
    def test_priority_keyword_generation(self):
        """Test generating priority keywords"""
        assert IMAPKeywords.priority(1) == "$Priority1"
        assert IMAPKeywords.priority(5) == "$Priority5"

