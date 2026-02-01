"""
Content Analyzer for Documents

Detects image content and text layers in documents to determine OCR needs.
"""

import io
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ContentAnalysis:
    """Results of content analysis for a document."""
    has_images: bool = False           # File contains image content
    has_native_text: bool = False      # File has extractable text layer
    is_image_only: bool = False        # File is all images, no native text
    ocr_recommended: bool = False      # OCR would likely help
    page_count: Optional[int] = None   # Number of pages (if applicable)
    image_page_count: int = 0          # Pages that are primarily images
    text_page_count: int = 0           # Pages with native text
    is_scanned_with_ocr: bool = False  # Scanned PDF with existing OCR overlay


def analyze_pdf(pdf_bytes: bytes) -> ContentAnalysis:
    """
    Analyze a PDF to detect image content and text layers.

    Uses pdfplumber to check each page for:
    - Extractable text (native text layer)
    - Images (scanned content)
    """
    import pdfplumber

    result = ContentAnalysis()

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            result.page_count = len(pdf.pages)

            for page in pdf.pages:
                # Check for text
                text = page.extract_text() or ""
                has_text = len(text.strip()) > 50  # More than trivial text

                # Check for images
                images = page.images or []
                has_images = len(images) > 0

                # Also check for large images that might be scanned pages
                page_area = page.width * page.height
                large_images = [
                    img for img in images
                    if (img.get('width', 0) * img.get('height', 0)) > (page_area * 0.5)
                ]
                is_scanned_page = len(large_images) > 0

                if has_images or is_scanned_page:
                    result.has_images = True
                    result.image_page_count += 1

                if has_text:
                    result.has_native_text = True
                    result.text_page_count += 1

            # Determine if image-only (no native text despite having images)
            result.is_image_only = result.has_images and not result.has_native_text

            # Detect scanned PDF with OCR overlay:
            # - Has images on most pages (scanned)
            # - Also has text (OCR was applied)
            if result.has_images and result.has_native_text and result.page_count:
                image_coverage = result.image_page_count / result.page_count
                if image_coverage > 0.5:  # More than half pages have large images
                    result.is_scanned_with_ocr = True

            # OCR recommended if document has image content and we haven't run OCR:
            # - Image-only (scanned without any OCR) -> definitely needs OCR
            # - Scanned with existing OCR overlay -> recommend re-OCR with our pipeline
            # - Has images but low text coverage -> partial OCR, needs more
            # The ocr_recommended flag is initialized here, but the actual decision
            # also considers ocr_applied (set separately). Final logic:
            # ocr_recommended AND NOT ocr_applied = needs OCR
            if result.is_image_only:
                result.ocr_recommended = True
            elif result.is_scanned_with_ocr:
                # Has existing OCR but we haven't run ours - recommend re-OCR
                result.ocr_recommended = True
            elif result.has_images and result.page_count:
                text_coverage = result.text_page_count / result.page_count
                if text_coverage < 0.5:  # Less than half pages have text
                    result.ocr_recommended = True

    except Exception as e:
        logger.warning(f"PDF analysis failed: {e}")
        # On error, assume it might need OCR
        result.ocr_recommended = True

    return result


def analyze_image(image_bytes: bytes, mime_type: str) -> ContentAnalysis:
    """
    Analyze an image file.

    Images are always image-only and OCR recommended.
    """
    return ContentAnalysis(
        has_images=True,
        has_native_text=False,
        is_image_only=True,
        ocr_recommended=True,
        page_count=1,
        image_page_count=1,
        text_page_count=0,
    )


def analyze_office_document(doc_bytes: bytes, mime_type: str) -> ContentAnalysis:
    """
    Analyze Office documents (docx, xlsx, pptx).

    These typically have native text, may contain embedded images.
    """
    result = ContentAnalysis(
        has_native_text=True,  # Office docs always have native text
        is_image_only=False,
        ocr_recommended=False,
    )

    # Could add image detection for embedded images later
    # For now, assume office docs don't need OCR

    return result


def analyze_text_file(text_bytes: bytes) -> ContentAnalysis:
    """
    Analyze plain text files.

    Text files have native text, no images.
    """
    return ContentAnalysis(
        has_images=False,
        has_native_text=True,
        is_image_only=False,
        ocr_recommended=False,
    )


def analyze_content(
    file_bytes: bytes,
    mime_type: Optional[str],
    filename: Optional[str] = None,
) -> ContentAnalysis:
    """
    Analyze file content to detect image/text properties.

    Args:
        file_bytes: File content
        mime_type: MIME type of the file
        filename: Optional filename for extension-based detection

    Returns:
        ContentAnalysis with detected properties
    """
    if not mime_type and filename:
        import mimetypes
        mime_type, _ = mimetypes.guess_type(filename)

    mime_type = mime_type or ""

    # PDF
    if mime_type == "application/pdf" or (filename and filename.lower().endswith(".pdf")):
        return analyze_pdf(file_bytes)

    # Images
    if mime_type.startswith("image/"):
        return analyze_image(file_bytes, mime_type)

    # Office documents
    office_types = [
        "application/vnd.openxmlformats-officedocument",
        "application/msword",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
    ]
    if any(mime_type.startswith(t) for t in office_types):
        return analyze_office_document(file_bytes, mime_type)

    # Plain text
    if mime_type.startswith("text/"):
        return analyze_text_file(file_bytes)

    # Unknown - assume text-based, no OCR needed
    return ContentAnalysis(
        has_native_text=True,
        is_image_only=False,
        ocr_recommended=False,
    )
