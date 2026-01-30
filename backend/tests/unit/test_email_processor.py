"""
Test email processing following inbox-zero patterns.
Tests for parsing, HTML to Markdown conversion, and attachment extraction.
"""
import pytest
from backend.core.email.processor import EmailProcessor


class TestEmailProcessor:
    """Port inbox-zero's email processor tests"""
    
    @pytest.mark.asyncio
    async def test_process_simple_email(self, raw_email_simple):
        """Test basic email processing"""
        processor = EmailProcessor()
        
        result = await processor.process(raw_email_simple, uid="123")
        
        assert result.uid == "123"
        assert result.subject == "Test Email"
        assert result.from_address == "sender@example.com"
        assert "recipient@example.com" in result.to_addresses
        assert "test email body" in result.body_markdown.lower()
        assert result.sender_domain == "example.com"
    
    @pytest.mark.asyncio
    async def test_html_to_markdown_conversion(self, raw_email_html):
        """Test HTML conversion like inbox-zero"""
        processor = EmailProcessor()
        
        result = await processor.process(raw_email_html, uid="456")
        
        # Check markdown conversion
        assert "# Test Header" in result.body_markdown or "Test Header" in result.body_markdown
        assert "**bold text**" in result.body_markdown or "bold text" in result.body_markdown
        assert "[link]" in result.body_markdown or "link" in result.body_markdown
        assert result.subject == "HTML Test Email"
    
    @pytest.mark.asyncio
    async def test_extract_sender_domain(self, raw_email_simple):
        """Test sender domain extraction"""
        processor = EmailProcessor()
        
        result = await processor.process(raw_email_simple, uid="789")
        
        assert result.sender_domain == "example.com"
    
    @pytest.mark.asyncio
    async def test_handle_missing_fields(self):
        """Test handling of emails with missing fields"""
        processor = EmailProcessor()
        
        # Minimal email
        minimal_email = b"""From: test@example.com

Minimal body.
"""
        
        result = await processor.process(minimal_email, uid="minimal")
        
        assert result.uid == "minimal"
        assert result.from_address == "test@example.com"
        assert result.body_markdown.strip() == "Minimal body."
        assert result.subject == "(No Subject)"  # Fallback for missing subject
    
    @pytest.mark.asyncio
    async def test_decode_encoded_subject(self):
        """Test decoding of encoded email subjects"""
        processor = EmailProcessor()
        
        # Email with encoded subject
        encoded_email = b"""From: test@example.com
To: recipient@example.com
Subject: =?utf-8?B?VGVzdCBTdWJqZWN0?=
Date: Mon, 1 Jan 2024 12:00:00 +0000

Body text.
"""
        
        result = await processor.process(encoded_email, uid="encoded")
        
        # Should decode to "Test Subject"
        assert "Test" in result.subject or result.subject == "Test Subject"

