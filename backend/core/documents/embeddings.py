"""
Document Embedding Service

Generates vector embeddings for documents using the same model as emails
(text-embedding-3-large, 3072 dimensions) to enable unified semantic search.

Supports:
- Full document embeddings
- Page-level embeddings for multi-page documents
- Chunking for very long pages
- Metadata-only embeddings for documents without extractable text
"""

import logging
from dataclasses import dataclass
from typing import List, Optional
from uuid import UUID

from backend.core.documents.models import Document, DocumentEmbedding, ExtractionStatus
from backend.core.documents.repository import DocumentRepository

logger = logging.getLogger(__name__)

# Constants matching email embedding configuration
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
EMBEDDING_DIMENSIONS = 3072
MAX_TOKENS_PER_CHUNK = 8000  # Safe limit for embedding model
CHARS_PER_TOKEN_ESTIMATE = 4  # Rough estimate
MAX_CHARS_PER_CHUNK = MAX_TOKENS_PER_CHUNK * CHARS_PER_TOKEN_ESTIMATE


@dataclass
class PageContent:
    """Content for a single page of a document."""
    page_number: int
    text: str
    start_offset: int = 0
    end_offset: int = 0


@dataclass
class SheetContent:
    """Content for a single sheet of a spreadsheet."""
    sheet_name: str
    sheet_index: int
    text: str


@dataclass
class SectionContent:
    """Content for a section of text/markdown."""
    section_index: int
    title: Optional[str]  # Header text for markdown, None for plain text
    level: Optional[int]  # Header level (1-6) for markdown
    text: str


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings_created: int
    pages_processed: int
    chunks_created: int
    model: str


class DocumentEmbeddingService:
    """
    Service for generating document embeddings.

    Uses the same embedding model as email embeddings to enable
    unified semantic search across both corpora.
    """

    def __init__(
        self,
        repository: DocumentRepository,
        embedding_client=None,
    ):
        """
        Initialize the embedding service.

        Args:
            repository: DocumentRepository for storing embeddings
            embedding_client: OpenAI client for generating embeddings.
                              If None, will be lazy-loaded.
        """
        self.repository = repository
        self._client = embedding_client
        self.model = DEFAULT_EMBEDDING_MODEL

    @property
    def client(self):
        """Lazy load the OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI()
            except ImportError:
                logger.error("OpenAI client not available")
                raise RuntimeError("OpenAI client required for embeddings")
        return self._client

    def prepare_document_for_embedding(self, document: Document) -> str:
        """
        Build embedding text from document content and metadata.

        Even documents without extractable text get embeddings from:
        - Filename (often semantic: "Q4_Financial_Report.pdf")
        - Path/location (semantic context: "/finance/invoices/2025/")
        - Title and summary if available
        - Document type and tags

        Args:
            document: Document to prepare

        Returns:
            Text suitable for embedding generation
        """
        parts = []

        # Filename is always available and often meaningful
        if document.original_filename:
            parts.append(f"Filename: {document.original_filename}")

        # Title if extracted
        if document.title:
            parts.append(f"Title: {document.title}")

        # Document type classification
        if document.document_type:
            parts.append(f"Type: {document.document_type}")

        # AI-generated tags
        if document.ai_tags:
            parts.append(f"Tags: {', '.join(document.ai_tags)}")

        # Category
        if document.ai_category:
            parts.append(f"Category: {document.ai_category}")

        # Summary (one-line description)
        if document.summary:
            parts.append(f"Summary: {document.summary}")

        # Main content (extracted text)
        if document.extracted_text:
            # Truncate if too long
            text = document.extracted_text
            if len(text) > MAX_CHARS_PER_CHUNK:
                text = text[:MAX_CHARS_PER_CHUNK] + "..."
            parts.append(text)

        # Get primary origin path for context
        if document.origins:
            primary = next(
                (o for o in document.origins if o.is_primary),
                document.origins[0] if document.origins else None
            )
            if primary and primary.origin_path:
                parts.append(f"Location: {primary.origin_path}")

        return "\n\n".join(parts)

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector (3072 dimensions)
        """
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        response = self.client.embeddings.create(
            model=self.model,
            input=text,
        )
        return response.data[0].embedding

    async def generate_document_embedding(
        self,
        document: Document,
    ) -> EmbeddingResult:
        """
        Generate embedding for a document.

        Uses stored extraction structure if available:
        - For PDFs with pages: generates per-page embeddings
        - For XLSX with sheets: generates per-sheet embeddings
        - For others: generates whole-document or chunked embeddings

        Args:
            document: Document to embed

        Returns:
            EmbeddingResult with counts
        """
        # Check for stored extraction structure (pages/sheets)
        structure = await self.repository.get_extraction_structure(document.id)

        if structure:
            if "pages" in structure and structure["pages"]:
                # Convert to PageContent objects
                pages = [
                    PageContent(
                        page_number=p["page"],
                        text=p["text"],
                    )
                    for p in structure["pages"]
                    if p.get("text")
                ]
                if pages:
                    logger.info(f"Generating per-page embeddings for {len(pages)} pages")
                    return await self.generate_page_embeddings(document, pages)

            elif "sheets" in structure and structure["sheets"]:
                # Convert to SheetContent objects
                sheets = [
                    SheetContent(
                        sheet_name=s.get("sheet", f"Sheet {s.get('sheet_index', 0) + 1}"),
                        sheet_index=s.get("sheet_index", 0),
                        text=s["text"],
                    )
                    for s in structure["sheets"]
                    if s.get("text")
                ]
                if sheets:
                    logger.info(f"Generating per-sheet embeddings for {len(sheets)} sheets")
                    return await self.generate_sheet_embeddings(document, sheets)

            elif "sections" in structure and structure["sections"]:
                # Convert to SectionContent objects (for text/markdown)
                sections = [
                    SectionContent(
                        section_index=s.get("section", 0),
                        title=s.get("title"),
                        level=s.get("level"),
                        text=s["text"],
                    )
                    for s in structure["sections"]
                    if s.get("text")
                ]
                if sections:
                    logger.info(f"Generating per-section embeddings for {len(sections)} sections")
                    return await self.generate_section_embeddings(document, sections)

        # Fall back to standard document embedding
        # Delete any existing embeddings (for re-embedding)
        await self.repository.delete_embeddings(document.id)

        # Prepare text for embedding
        text = self.prepare_document_for_embedding(document)

        if not text or not text.strip():
            logger.warning(f"No content to embed for document {document.id}")
            return EmbeddingResult(
                embeddings_created=0,
                pages_processed=0,
                chunks_created=0,
                model=self.model,
            )

        # For simple documents, create a single embedding
        if len(text) <= MAX_CHARS_PER_CHUNK:
            embedding = await self.generate_embedding(text)
            await self.repository.add_embedding(
                document_id=document.id,
                embedding=embedding,
                page_number=None,  # Whole document
                chunk_index=0,
                chunk_text=text,
                model=self.model,
            )
            return EmbeddingResult(
                embeddings_created=1,
                pages_processed=1,
                chunks_created=1,
                model=self.model,
            )

        # For long documents, chunk the text
        chunks = self._chunk_text(text, MAX_CHARS_PER_CHUNK)
        embeddings_created = 0

        for i, chunk in enumerate(chunks):
            embedding = await self.generate_embedding(chunk)
            await self.repository.add_embedding(
                document_id=document.id,
                embedding=embedding,
                page_number=None,  # No page info for simple chunking
                chunk_index=i,
                chunk_start=i * MAX_CHARS_PER_CHUNK,
                chunk_end=min((i + 1) * MAX_CHARS_PER_CHUNK, len(text)),
                chunk_text=chunk,
                model=self.model,
            )
            embeddings_created += 1

        return EmbeddingResult(
            embeddings_created=embeddings_created,
            pages_processed=1,
            chunks_created=embeddings_created,
            model=self.model,
        )

    async def generate_page_embeddings(
        self,
        document: Document,
        pages: List[PageContent],
    ) -> EmbeddingResult:
        """
        Generate per-page embeddings for multi-page documents.

        Args:
            document: Document being embedded
            pages: List of page contents

        Returns:
            EmbeddingResult with counts
        """
        # Delete existing embeddings
        await self.repository.delete_embeddings(document.id)

        embeddings_created = 0
        total_chunks = 0

        for page in pages:
            if not page.text or not page.text.strip():
                continue

            # Add document context to each page
            page_text = self._prepare_page_text(document, page)

            if len(page_text) <= MAX_CHARS_PER_CHUNK:
                # Single embedding for page
                embedding = await self.generate_embedding(page_text)
                await self.repository.add_embedding(
                    document_id=document.id,
                    embedding=embedding,
                    page_number=page.page_number,
                    chunk_index=0,
                    chunk_text=page_text,
                    model=self.model,
                )
                embeddings_created += 1
                total_chunks += 1
            else:
                # Chunk long pages
                chunks = self._chunk_text(page_text, MAX_CHARS_PER_CHUNK)
                for i, chunk in enumerate(chunks):
                    embedding = await self.generate_embedding(chunk)
                    await self.repository.add_embedding(
                        document_id=document.id,
                        embedding=embedding,
                        page_number=page.page_number,
                        chunk_index=i,
                        chunk_text=chunk,
                        model=self.model,
                    )
                    embeddings_created += 1
                    total_chunks += 1

        return EmbeddingResult(
            embeddings_created=embeddings_created,
            pages_processed=len(pages),
            chunks_created=total_chunks,
            model=self.model,
        )

    async def generate_sheet_embeddings(
        self,
        document: Document,
        sheets: List[SheetContent],
    ) -> EmbeddingResult:
        """
        Generate per-sheet embeddings for multi-sheet spreadsheets.

        Args:
            document: Document being embedded
            sheets: List of sheet contents

        Returns:
            EmbeddingResult with counts
        """
        # Delete existing embeddings
        await self.repository.delete_embeddings(document.id)

        embeddings_created = 0
        total_chunks = 0

        for sheet in sheets:
            if not sheet.text or not sheet.text.strip():
                continue

            # Add document context to each sheet
            sheet_text = self._prepare_sheet_text(document, sheet)

            if len(sheet_text) <= MAX_CHARS_PER_CHUNK:
                # Single embedding for sheet
                embedding = await self.generate_embedding(sheet_text)
                await self.repository.add_embedding(
                    document_id=document.id,
                    embedding=embedding,
                    page_number=sheet.sheet_index + 1,  # 1-indexed for consistency
                    chunk_index=0,
                    chunk_text=sheet_text,
                    model=self.model,
                )
                embeddings_created += 1
                total_chunks += 1
            else:
                # Chunk long sheets
                chunks = self._chunk_text(sheet_text, MAX_CHARS_PER_CHUNK)
                for i, chunk in enumerate(chunks):
                    embedding = await self.generate_embedding(chunk)
                    await self.repository.add_embedding(
                        document_id=document.id,
                        embedding=embedding,
                        page_number=sheet.sheet_index + 1,
                        chunk_index=i,
                        chunk_text=chunk,
                        model=self.model,
                    )
                    embeddings_created += 1
                    total_chunks += 1

        return EmbeddingResult(
            embeddings_created=embeddings_created,
            pages_processed=len(sheets),  # "pages" here means sheets
            chunks_created=total_chunks,
            model=self.model,
        )

    def _prepare_page_text(self, document: Document, page: PageContent) -> str:
        """Prepare page text with document context."""
        parts = []

        # Add document context
        if document.title:
            parts.append(f"Document: {document.title}")
        elif document.original_filename:
            parts.append(f"Document: {document.original_filename}")

        parts.append(f"Page {page.page_number}")

        # Add page content
        parts.append(page.text)

        return "\n\n".join(parts)

    def _prepare_sheet_text(self, document: Document, sheet: SheetContent) -> str:
        """Prepare sheet text with document context."""
        parts = []

        # Add document context
        if document.title:
            parts.append(f"Document: {document.title}")
        elif document.original_filename:
            parts.append(f"Document: {document.original_filename}")

        parts.append(f"Sheet: {sheet.sheet_name}")

        # Add sheet content
        parts.append(sheet.text)

        return "\n\n".join(parts)

    async def generate_section_embeddings(
        self,
        document: Document,
        sections: List[SectionContent],
    ) -> EmbeddingResult:
        """
        Generate per-section embeddings for text/markdown documents.

        Args:
            document: Document being embedded
            sections: List of section contents

        Returns:
            EmbeddingResult with counts
        """
        # Delete existing embeddings
        await self.repository.delete_embeddings(document.id)

        embeddings_created = 0
        total_chunks = 0

        for section in sections:
            if not section.text or not section.text.strip():
                continue

            # Add document context to each section
            section_text = self._prepare_section_text(document, section)

            if len(section_text) <= MAX_CHARS_PER_CHUNK:
                # Single embedding for section
                embedding = await self.generate_embedding(section_text)
                await self.repository.add_embedding(
                    document_id=document.id,
                    embedding=embedding,
                    page_number=section.section_index + 1,  # 1-indexed
                    chunk_index=0,
                    chunk_text=section_text,
                    model=self.model,
                )
                embeddings_created += 1
                total_chunks += 1
            else:
                # Chunk long sections
                chunks = self._chunk_text(section_text, MAX_CHARS_PER_CHUNK)
                for i, chunk in enumerate(chunks):
                    embedding = await self.generate_embedding(chunk)
                    await self.repository.add_embedding(
                        document_id=document.id,
                        embedding=embedding,
                        page_number=section.section_index + 1,
                        chunk_index=i,
                        chunk_text=chunk,
                        model=self.model,
                    )
                    embeddings_created += 1
                    total_chunks += 1

        return EmbeddingResult(
            embeddings_created=embeddings_created,
            pages_processed=len(sections),
            chunks_created=total_chunks,
            model=self.model,
        )

    def _prepare_section_text(self, document: Document, section: SectionContent) -> str:
        """Prepare section text with document context."""
        parts = []

        # Add document context
        if document.title:
            parts.append(f"Document: {document.title}")
        elif document.original_filename:
            parts.append(f"Document: {document.original_filename}")

        # Add section header if available
        if section.title:
            parts.append(f"Section: {section.title}")
        else:
            parts.append(f"Section {section.section_index + 1}")

        # Add section content
        parts.append(section.text)

        return "\n\n".join(parts)

    def _chunk_text(self, text: str, max_chars: int) -> List[str]:
        """
        Split text into chunks at word boundaries.

        Args:
            text: Text to chunk
            max_chars: Maximum characters per chunk

        Returns:
            List of text chunks
        """
        if len(text) <= max_chars:
            return [text]

        chunks = []
        current_pos = 0

        while current_pos < len(text):
            # Find end of chunk
            end_pos = current_pos + max_chars

            if end_pos >= len(text):
                chunks.append(text[current_pos:])
                break

            # Try to break at paragraph
            para_break = text.rfind('\n\n', current_pos, end_pos)
            if para_break > current_pos + max_chars // 2:
                end_pos = para_break + 2
            else:
                # Try to break at sentence
                sentence_break = max(
                    text.rfind('. ', current_pos, end_pos),
                    text.rfind('! ', current_pos, end_pos),
                    text.rfind('? ', current_pos, end_pos),
                )
                if sentence_break > current_pos + max_chars // 2:
                    end_pos = sentence_break + 2
                else:
                    # Break at word boundary
                    word_break = text.rfind(' ', current_pos, end_pos)
                    if word_break > current_pos:
                        end_pos = word_break + 1

            chunks.append(text[current_pos:end_pos])
            current_pos = end_pos

        return chunks

    async def embed_pending_documents(
        self,
        limit: int = 10,
    ) -> int:
        """
        Generate embeddings for documents that need them.

        Processes documents with completed extraction that don't have embeddings.

        Args:
            limit: Maximum documents to process

        Returns:
            Number of documents processed
        """
        # Get pending embedding tasks from queue
        tasks = await self.repository.get_pending_tasks(
            task_type="generate_embedding",
            limit=limit,
        )

        processed = 0
        for task in tasks:
            # Claim the task
            claimed = await self.repository.mark_task_processing(
                task_id=task.id,
                worker_id="embedding_service",
            )
            if not claimed:
                continue

            try:
                document = await self.repository.get_by_id(task.document_id)
                if not document:
                    await self.repository.mark_task_failed(task.id, "Document not found")
                    continue

                await self.generate_document_embedding(document)
                await self.repository.mark_task_completed(task.id)
                processed += 1
                logger.info(f"Generated embeddings for document {document.id}")

            except Exception as e:
                logger.error(f"Failed to embed document {task.document_id}: {e}")
                await self.repository.mark_task_failed(task.id, str(e))

        return processed

    async def regenerate_pending_embeddings(
        self,
        limit: int = 10,
    ) -> int:
        """
        Regenerate embeddings for documents whose text has changed.

        Processes "regenerate_embedding" tasks, which delete existing
        embeddings before generating new ones.

        Args:
            limit: Maximum documents to process

        Returns:
            Number of documents processed
        """
        # Get pending regeneration tasks from queue
        tasks = await self.repository.get_pending_tasks(
            task_type="regenerate_embedding",
            limit=limit,
        )

        processed = 0
        for task in tasks:
            # Claim the task
            claimed = await self.repository.mark_task_processing(
                task_id=task.id,
                worker_id="embedding_service",
            )
            if not claimed:
                continue

            try:
                document = await self.repository.get_by_id(task.document_id)
                if not document:
                    await self.repository.mark_task_failed(task.id, "Document not found")
                    continue

                # Delete existing embeddings first
                deleted_count = await self.repository.delete_embeddings(document.id)
                if deleted_count > 0:
                    logger.info(f"Deleted {deleted_count} old embeddings for document {document.id}")

                # Generate new embeddings
                result = await self.generate_document_embedding(document)
                await self.repository.mark_task_completed(task.id)
                processed += 1
                logger.info(f"Regenerated {result.embeddings_created} embeddings for document {document.id}")

            except Exception as e:
                logger.error(f"Failed to regenerate embeddings for document {task.document_id}: {e}")
                await self.repository.mark_task_failed(task.id, str(e))

        return processed

    async def process_all_pending(
        self,
        limit: int = 10,
    ) -> dict:
        """
        Process all pending embedding tasks (new and regeneration).

        Args:
            limit: Maximum documents to process per type

        Returns:
            Dict with counts: {"generated": N, "regenerated": M}
        """
        generated = await self.embed_pending_documents(limit=limit)
        regenerated = await self.regenerate_pending_embeddings(limit=limit)
        return {"generated": generated, "regenerated": regenerated}
