"""
Test email preprocessing system for forwarded emails and special cases.
"""
import pytest
from datetime import datetime
from backend.core.email.preprocessor import (
    PreprocessingRule,
    EmailPreprocessor
)
from backend.core.email.models import ProcessedEmail


class TestPreprocessingRule:
    """Test preprocessing rules"""
    
    def test_match_from_pattern(self):
        """Test matching on from address"""
        rule = PreprocessingRule(
            name="Test",
            description="Test rule",
            match_from="user@research-lab.org"
        )
        
        email = ProcessedEmail(
            uid="1",
            message_id="<test@example.com>",
            subject="Test",
            from_address="user@research-lab.org",
            to_addresses=["auto@research-lab.org"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="research-lab.org",
            attachment_info=[],
            raw_headers={}
        )
        
        assert rule.matches(email, {}) == True
    
    def test_match_from_and_to(self):
        """Test matching on both from and to (AND operation)"""
        rule = PreprocessingRule(
            name="Auto-forward",
            description="Auto-forwarded emails",
            match_from="user@research-lab.org",
            match_to="user.auto@research-lab.org"
        )
        
        # Both match
        email = ProcessedEmail(
            uid="1",
            message_id="<test@example.com>",
            subject="Test",
            from_address="user@research-lab.org",
            to_addresses=["user.auto@research-lab.org"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="research-lab.org",
            attachment_info=[],
            raw_headers={}
        )
        
        assert rule.matches(email, {}) == True
        
        # Different to address - should not match
        email.to_addresses = ["other@example.com"]
        assert rule.matches(email, {}) == False
    
    def test_extract_original_from(self):
        """Test extracting original sender from headers"""
        rule = PreprocessingRule(
            name="Unwrap forwarded",
            description="Extract original sender",
            match_from="forwarder@example.com",
            extract_original_from="X-Original-Sender"
        )
        
        email = ProcessedEmail(
            uid="1",
            message_id="<test@example.com>",
            subject="Fwd: Original Subject",
            from_address="forwarder@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="example.com",
            attachment_info=[],
            raw_headers={}
        )
        
        # Headers contain original sender
        headers = {
            'X-Original-Sender': 'original@sender.com',
            'Subject': 'Fwd: Original Subject'
        }
        
        # Apply preprocessing
        processed = rule.apply(email, headers)
        
        # Should have extracted original sender
        assert processed.from_address == 'original@sender.com'
        assert processed.sender_domain == 'sender.com'
    
    def test_extract_original_subject(self):
        """Test extracting original subject"""
        rule = PreprocessingRule(
            name="Unwrap subject",
            description="Extract original subject",
            match_from="forwarder@example.com",
            extract_original_subject="X-Original-Subject"
        )
        
        email = ProcessedEmail(
            uid="1",
            message_id="<test@example.com>",
            subject="Fwd: Something",
            from_address="forwarder@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="example.com",
            attachment_info=[],
            raw_headers={}
        )
        
        headers = {
            'X-Original-Subject': 'Original Question'
        }
        
        processed = rule.apply(email, headers)
        
        assert processed.subject == 'Original Question'
    
    def test_remove_prefix(self):
        """Test removing subject prefix"""
        rule = PreprocessingRule(
            name="Clean Fwd",
            description="Remove Fwd: prefix",
            match_subject="^Fwd:",
            remove_prefix="^Fwd:\\s*"
        )
        
        email = ProcessedEmail(
            uid="1",
            message_id="<test@example.com>",
            subject="Fwd: Actual Subject",
            from_address="sender@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="example.com",
            attachment_info=[],
            raw_headers={}
        )
        
        processed = rule.apply(email, {})
        
        assert processed.subject == "Actual Subject"
    
    def test_header_mappings(self):
        """Test custom header mappings"""
        rule = PreprocessingRule(
            name="Custom mapping",
            description="Map custom headers",
            match_from=".*",
            header_mappings={
                'from': 'Resent-From',
                'subject': 'Resent-Subject'
            }
        )
        
        email = ProcessedEmail(
            uid="1",
            message_id="<test@example.com>",
            subject="Forwarded subject",
            from_address="forwarder@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="example.com",
            attachment_info=[],
            raw_headers={}
        )
        
        headers = {
            'Resent-From': 'real@sender.com',
            'Resent-Subject': 'Real Subject'
        }
        
        processed = rule.apply(email, headers)
        
        assert processed.from_address == 'real@sender.com'
        assert processed.subject == 'Real Subject'
        assert processed.sender_domain == 'sender.com'


class TestEmailPreprocessor:
    """Test complete preprocessor"""
    
    def test_preprocessor_initialization(self):
        """Test creating preprocessor"""
        preprocessor = EmailPreprocessor([])
        
        assert len(preprocessor.rules) == 0
    
    def test_preprocess_with_matching_rule(self):
        """Test preprocessing with matching rule"""
        rule = PreprocessingRule(
            name="Auto-forward unwrap",
            description="Unwrap auto-forwarded emails",
            match_from="user@research-lab.org",
            match_to="user.auto@research-lab.org",
            extract_original_from="X-Original-Sender"
        )
        
        preprocessor = EmailPreprocessor([rule])
        
        email = ProcessedEmail(
            uid="1",
            message_id="<test@example.com>",
            subject="Fwd: Meeting Request",
            from_address="user@research-lab.org",
            to_addresses=["user.auto@research-lab.org"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="research-lab.org",
            attachment_info=[],
            raw_headers={}
        )
        
        headers = {
            'X-Original-Sender': 'colleague@institution.edu',
            'From': 'user@research-lab.org',
            'To': 'user.auto@research-lab.org'
        }
        
        # Preprocess
        result = preprocessor.preprocess(email, headers)
        
        # Should have unwrapped to show real sender
        assert result.from_address == 'colleague@institution.edu'
        assert result.sender_domain == 'institution.edu'
    
    def test_preprocess_no_matching_rule(self):
        """Test preprocessing when no rules match"""
        rule = PreprocessingRule(
            name="Specific rule",
            description="Test",
            match_from="specific@example.com"
        )
        
        preprocessor = EmailPreprocessor([rule])
        
        email = ProcessedEmail(
            uid="1",
            message_id="<test@example.com>",
            subject="Test",
            from_address="other@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="example.com",
            attachment_info=[],
            raw_headers={}
        )
        
        result = preprocessor.preprocess(email, {})
        
        # Should be unchanged
        assert result.from_address == "other@example.com"
    
    def test_multiple_preprocessing_rules(self):
        """Test applying multiple preprocessing rules in sequence"""
        rule1 = PreprocessingRule(
            name="Extract sender",
            description="Extract original sender",
            match_from=".*",
            extract_original_from="X-Original-Sender",
            priority=1
        )
        
        rule2 = PreprocessingRule(
            name="Clean subject",
            description="Remove Fwd: prefix",
            match_subject="^Fwd:",
            remove_prefix="^Fwd:\\s*",
            priority=2
        )
        
        preprocessor = EmailPreprocessor([rule1, rule2])
        
        email = ProcessedEmail(
            uid="1",
            message_id="<test@example.com>",
            subject="Fwd: Important",
            from_address="forwarder@example.com",
            to_addresses=["me@example.com"],
            date=datetime.now(),
            body_markdown="Content",
            sender_domain="example.com",
            attachment_info=[],
            raw_headers={}
        )
        
        headers = {
            'X-Original-Sender': 'real@sender.com'
        }
        
        result = preprocessor.preprocess(email, headers)
        
        # Both transformations should apply
        assert result.from_address == 'real@sender.com'
        assert result.subject == 'Important'  # Fwd: removed


class TestForwardedEmailUseCase:
    """Test the specific use case: auto-forwarded emails"""
    
    def test_user_auto_forward_scenario(self):
        """
        Test your specific scenario:
        Emails forwarded from user@research-lab.org to user.auto@research-lab.org
        """
        rule = PreprocessingRule(
            name="Auto-forwarded from main account",
            description="Unwrap auto-forwarded emails",
            match_from="^user@research-lab.org$",
            match_to="user.auto@research-lab.org",
            extract_original_from="X-Original-Sender",
            extract_original_to="X-Original-To",
            extract_original_subject="X-Original-Subject"
        )
        
        # Forwarded email (as it appears in auto inbox)
        forwarded_email = ProcessedEmail(
            uid="123",
            message_id="<fwd@research-lab.org>",
            subject="Fwd: Research Paper Review",
            from_address="user@research-lab.org",  # Forwarder
            to_addresses=["user.auto@research-lab.org"],  # Auto inbox
            date=datetime.now(),
            body_markdown="Original email content",
            sender_domain="research-lab.org",
            attachment_info=[],
            raw_headers={}
        )
        
        # Original headers from the actual sender
        headers = {
            'X-Original-Sender': 'colleague@institution.edu',
            'X-Original-To': 'user@research-lab.org',
            'X-Original-Subject': 'Research Paper Review',
            'From': 'user@research-lab.org',
            'To': 'user.auto@research-lab.org'
        }
        
        # Apply preprocessing
        processed = rule.apply(forwarded_email, headers)
        
        # Should show original sender, not forwarder
        assert processed.from_address == 'colleague@institution.edu'
        assert processed.sender_domain == 'institution.edu'
        
        # Should show original recipient
        assert 'user@research-lab.org' in processed.to_addresses
        
        # Should show original subject
        assert processed.subject == 'Research Paper Review'
        
        # Now classification will work on the REAL email, not the forwarding wrapper!

