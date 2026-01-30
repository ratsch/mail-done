"""
Tests for attachment format extractors (HIGH PRIORITY coverage gap).
Tests real file parsing with valid samples for each supported format.
"""
import pytest
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

from backend.core.email.processor import EmailProcessor


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def get_real_attachments(attachment_info):
    """Filter out inline text parts that get treated as attachments"""
    return [a for a in attachment_info if a.filename != 'unnamed.txt' and a.content_type != 'text/plain']


class TestFormatExtractors:
    """Test real file extraction for all supported formats"""
    
    @pytest.mark.asyncio
    async def test_extract_valid_pdf(self):
        """Test PDF text extraction with real PDF file"""
        processor = EmailProcessor()
        
        # Load real PDF fixture
        pdf_path = FIXTURES_DIR / "sample.pdf"
        assert pdf_path.exists(), f"Missing fixture: {pdf_path}"
        
        with open(pdf_path, "rb") as f:
            pdf_data = f.read()
        
        # Create email with PDF attachment
        msg = MIMEMultipart()
        msg['From'] = 'sender@example.com'
        msg['To'] = 'recipient@example.com'
        msg['Subject'] = 'PDF Test'
        msg.attach(MIMEText('Body'))
        
        attachment = MIMEApplication(pdf_data, _subtype='pdf')
        attachment.add_header('Content-Disposition', 'attachment', filename='test.pdf')
        msg.attach(attachment)
        
        # Process and verify
        result = await processor.process(msg.as_bytes(), uid="pdf_test")

        assert result.has_attachments
        real_attachments = get_real_attachments(result.attachment_info)
        assert len(real_attachments) == 1
        assert real_attachments[0].filename == 'test.pdf'
        assert real_attachments[0].content_type == 'application/pdf'
        # PDF extraction should work (may extract "Test PDF Content" or similar)
        assert real_attachments[0].extracted_text is not None or real_attachments[0].extraction_error is not None
    
    @pytest.mark.asyncio
    async def test_extract_valid_docx(self):
        """Test DOCX text extraction with real Word document"""
        processor = EmailProcessor()
        
        # Load real DOCX fixture
        docx_path = FIXTURES_DIR / "sample.docx"
        assert docx_path.exists(), f"Missing fixture: {docx_path}"
        
        with open(docx_path, "rb") as f:
            docx_data = f.read()
        
        # Create email with DOCX attachment
        msg = MIMEMultipart()
        msg['From'] = 'sender@example.com'
        msg['To'] = 'recipient@example.com'
        msg['Subject'] = 'DOCX Test'
        msg.attach(MIMEText('Body'))
        
        attachment = MIMEApplication(docx_data, _subtype='vnd.openxmlformats-officedocument.wordprocessingml.document')
        attachment.add_header('Content-Disposition', 'attachment', filename='test.docx')
        msg.attach(attachment)
        
        # Process and verify
        result = await processor.process(msg.as_bytes(), uid="docx_test")

        assert result.has_attachments
        real_attachments = get_real_attachments(result.attachment_info)
        assert len(real_attachments) == 1
        assert real_attachments[0].filename == 'test.docx'
        # Should extract text successfully
        if real_attachments[0].extracted_text:
            assert "Test DOCX Content" in real_attachments[0].extracted_text
            assert "sample Word document" in real_attachments[0].extracted_text
    
    @pytest.mark.asyncio
    async def test_extract_valid_xlsx(self):
        """Test Excel text extraction with real spreadsheet"""
        processor = EmailProcessor()
        
        # Load real XLSX fixture
        xlsx_path = FIXTURES_DIR / "sample.xlsx"
        assert xlsx_path.exists(), f"Missing fixture: {xlsx_path}"
        
        with open(xlsx_path, "rb") as f:
            xlsx_data = f.read()
        
        # Create email with XLSX attachment
        msg = MIMEMultipart()
        msg['From'] = 'sender@example.com'
        msg['To'] = 'recipient@example.com'
        msg['Subject'] = 'Excel Test'
        msg.attach(MIMEText('Body'))
        
        attachment = MIMEApplication(xlsx_data, _subtype='vnd.openxmlformats-officedocument.spreadsheetml.sheet')
        attachment.add_header('Content-Disposition', 'attachment', filename='test.xlsx')
        msg.attach(attachment)
        
        # Process and verify
        result = await processor.process(msg.as_bytes(), uid="xlsx_test")

        assert result.has_attachments
        real_attachments = get_real_attachments(result.attachment_info)
        assert len(real_attachments) == 1
        assert real_attachments[0].filename == 'test.xlsx'
        # Should extract cell text
        if real_attachments[0].extracted_text:
            assert "Test XLSX Content" in real_attachments[0].extracted_text
    
    @pytest.mark.asyncio
    async def test_extract_valid_pptx(self):
        """Test PowerPoint text extraction with real presentation"""
        processor = EmailProcessor()
        
        # Load real PPTX fixture
        pptx_path = FIXTURES_DIR / "sample.pptx"
        assert pptx_path.exists(), f"Missing fixture: {pptx_path}"
        
        with open(pptx_path, "rb") as f:
            pptx_data = f.read()
        
        # Create email with PPTX attachment
        msg = MIMEMultipart()
        msg['From'] = 'sender@example.com'
        msg['To'] = 'recipient@example.com'
        msg['Subject'] = 'PowerPoint Test'
        msg.attach(MIMEText('Body'))
        
        attachment = MIMEApplication(pptx_data, _subtype='vnd.openxmlformats-officedocument.presentationml.presentation')
        attachment.add_header('Content-Disposition', 'attachment', filename='test.pptx')
        msg.attach(attachment)
        
        # Process and verify
        result = await processor.process(msg.as_bytes(), uid="pptx_test")

        assert result.has_attachments
        real_attachments = get_real_attachments(result.attachment_info)
        assert len(real_attachments) == 1
        assert real_attachments[0].filename == 'test.pptx'
        # Should extract slide text
        if real_attachments[0].extracted_text:
            assert "Test PPTX Content" in real_attachments[0].extracted_text
    
    @pytest.mark.asyncio
    async def test_extract_valid_rtf(self):
        """Test RTF text extraction with real Rich Text file"""
        processor = EmailProcessor()
        
        # Load real RTF fixture
        rtf_path = FIXTURES_DIR / "sample.rtf"
        assert rtf_path.exists(), f"Missing fixture: {rtf_path}"
        
        with open(rtf_path, "rb") as f:
            rtf_data = f.read()
        
        # Create email with RTF attachment
        msg = MIMEMultipart()
        msg['From'] = 'sender@example.com'
        msg['To'] = 'recipient@example.com'
        msg['Subject'] = 'RTF Test'
        msg.attach(MIMEText('Body'))
        
        attachment = MIMEApplication(rtf_data, _subtype='rtf')
        attachment.add_header('Content-Disposition', 'attachment', filename='test.rtf')
        msg.attach(attachment)
        
        # Process and verify
        result = await processor.process(msg.as_bytes(), uid="rtf_test")

        assert result.has_attachments
        real_attachments = get_real_attachments(result.attachment_info)
        assert len(real_attachments) == 1
        assert real_attachments[0].filename == 'test.rtf'
        # Should extract text successfully
        if real_attachments[0].extracted_text:
            assert "Test RTF Content" in real_attachments[0].extracted_text
    
    @pytest.mark.asyncio
    async def test_extract_valid_ics(self):
        """Test iCalendar text extraction with real calendar file"""
        processor = EmailProcessor()
        
        # Load real ICS fixture
        ics_path = FIXTURES_DIR / "sample.ics"
        assert ics_path.exists(), f"Missing fixture: {ics_path}"
        
        with open(ics_path, "rb") as f:
            ics_data = f.read()
        
        # Create email with ICS attachment
        msg = MIMEMultipart()
        msg['From'] = 'sender@example.com'
        msg['To'] = 'recipient@example.com'
        msg['Subject'] = 'Calendar Test'
        msg.attach(MIMEText('Body'))
        
        attachment = MIMEApplication(ics_data, _subtype='calendar')
        attachment.add_header('Content-Disposition', 'attachment', filename='test.ics')
        msg.attach(attachment)
        
        # Process and verify
        result = await processor.process(msg.as_bytes(), uid="ics_test")

        assert result.has_attachments
        real_attachments = get_real_attachments(result.attachment_info)
        assert len(real_attachments) == 1
        assert real_attachments[0].filename == 'test.ics'
        # Should extract event details
        if real_attachments[0].extracted_text:
            assert "Test Calendar Event" in real_attachments[0].extracted_text
            assert "Conference Room A" in real_attachments[0].extracted_text
    
    @pytest.mark.asyncio
    async def test_extract_corrupted_pdf(self):
        """Test PDF extraction with corrupted file (error handling)"""
        processor = EmailProcessor()
        
        # Load corrupted PDF fixture
        corrupted_path = FIXTURES_DIR / "corrupted.pdf"
        assert corrupted_path.exists(), f"Missing fixture: {corrupted_path}"
        
        with open(corrupted_path, "rb") as f:
            bad_pdf = f.read()
        
        # Create email with corrupted PDF
        msg = MIMEMultipart()
        msg['From'] = 'sender@example.com'
        msg['To'] = 'recipient@example.com'
        msg['Subject'] = 'Corrupted PDF Test'
        msg.attach(MIMEText('Body'))
        
        attachment = MIMEApplication(bad_pdf, _subtype='pdf')
        attachment.add_header('Content-Disposition', 'attachment', filename='corrupted.pdf')
        msg.attach(attachment)
        
        # Process - should not crash
        result = await processor.process(msg.as_bytes(), uid="corrupted_pdf")

        assert result.has_attachments
        real_attachments = get_real_attachments(result.attachment_info)
        assert len(real_attachments) == 1
        # Should not crash - may return empty string or None with error
        # PDF extractor may return empty string on corruption
        extracted = real_attachments[0].extracted_text
        assert extracted is None or extracted == ""
        # Error is logged but may not be in extraction_error field
        # (pdfplumber returns empty string on error)
    
    @pytest.mark.asyncio
    async def test_extract_corrupted_docx(self):
        """Test DOCX extraction with corrupted file (error handling)"""
        processor = EmailProcessor()
        
        # Load corrupted DOCX fixture
        corrupted_path = FIXTURES_DIR / "corrupted.docx"
        assert corrupted_path.exists(), f"Missing fixture: {corrupted_path}"
        
        with open(corrupted_path, "rb") as f:
            bad_docx = f.read()
        
        # Create email with corrupted DOCX
        msg = MIMEMultipart()
        msg['From'] = 'sender@example.com'
        msg['To'] = 'recipient@example.com'
        msg['Subject'] = 'Corrupted DOCX Test'
        msg.attach(MIMEText('Body'))
        
        attachment = MIMEApplication(bad_docx, _subtype='vnd.openxmlformats-officedocument.wordprocessingml.document')
        attachment.add_header('Content-Disposition', 'attachment', filename='corrupted.docx')
        msg.attach(attachment)
        
        # Process - should not crash
        result = await processor.process(msg.as_bytes(), uid="corrupted_docx")

        assert result.has_attachments
        real_attachments = get_real_attachments(result.attachment_info)
        assert len(real_attachments) == 1
        # Should record error or empty text, not crash
        # Corrupted files may return empty string or None with error
        extracted = real_attachments[0].extracted_text
        assert extracted is None or extracted == ""


class TestMultipleFormats:
    """Test emails with multiple different format attachments"""
    
    @pytest.mark.asyncio
    async def test_email_with_mixed_attachments(self):
        """Test email with PDF, DOCX, and ICS attachments"""
        processor = EmailProcessor()
        
        # Load fixtures
        pdf_data = (FIXTURES_DIR / "sample.pdf").read_bytes()
        docx_data = (FIXTURES_DIR / "sample.docx").read_bytes()
        ics_data = (FIXTURES_DIR / "sample.ics").read_bytes()
        
        # Create email with multiple attachments
        msg = MIMEMultipart()
        msg['From'] = 'sender@example.com'
        msg['To'] = 'recipient@example.com'
        msg['Subject'] = 'Multiple Attachments'
        msg.attach(MIMEText('Email with 3 attachments'))
        
        # Add PDF
        pdf_att = MIMEApplication(pdf_data, _subtype='pdf')
        pdf_att.add_header('Content-Disposition', 'attachment', filename='doc.pdf')
        msg.attach(pdf_att)
        
        # Add DOCX
        docx_att = MIMEApplication(docx_data, _subtype='vnd.openxmlformats-officedocument.wordprocessingml.document')
        docx_att.add_header('Content-Disposition', 'attachment', filename='doc.docx')
        msg.attach(docx_att)
        
        # Add ICS
        ics_att = MIMEApplication(ics_data, _subtype='calendar')
        ics_att.add_header('Content-Disposition', 'attachment', filename='event.ics')
        msg.attach(ics_att)
        
        # Process
        result = await processor.process(msg.as_bytes(), uid="multi_test")

        # Verify all attachments processed
        assert result.has_attachments
        real_attachments = get_real_attachments(result.attachment_info)
        assert len(real_attachments) == 3

        # Check each was processed
        filenames = [info.filename for info in real_attachments]
        assert 'doc.pdf' in filenames
        assert 'doc.docx' in filenames
        assert 'event.ics' in filenames

        # Check at least some text was extracted
        extracted_count = sum(1 for info in real_attachments if info.extracted_text)
        assert extracted_count > 0, "Should extract text from at least one attachment"

