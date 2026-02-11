"""
Folder Scanner

Scans folders on local or remote hosts to discover and register documents.
Supports:
- Recursive/non-recursive scanning
- Extension filtering
- Exclusion patterns
- Hidden file skipping
- Max file size limits
- Incremental scanning (skip unchanged files)
"""

import fnmatch
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Dict, Any
from uuid import UUID

from backend.core.documents.config import FolderScanConfig, HostConfig
from backend.core.documents.models import Document, ExtractionStatus
from backend.core.documents.processor import DocumentProcessor
from backend.core.documents.repository import DocumentRepository

logger = logging.getLogger(__name__)

# Extensions that support OCR sidecar files
OCR_SIDECAR_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}


def _load_ocr_sidecar(file_path: Path) -> tuple[Optional[dict], Optional[float]]:
    """Check for .ocr.json sidecar file and load it.

    Returns:
        Tuple of (sidecar_data, sidecar_mtime) or (None, None) if not found.
    """
    sidecar_path = file_path.parent / (file_path.name + ".ocr.json")
    if not sidecar_path.exists():
        return None, None
    try:
        sidecar_mtime = sidecar_path.stat().st_mtime
        with open(sidecar_path) as f:
            data = json.load(f)
        logger.debug(f"Loaded OCR sidecar: {sidecar_path} (mtime={sidecar_mtime})")
        return data, sidecar_mtime
    except (json.JSONDecodeError, IOError, OSError) as e:
        logger.warning(f"Failed to load OCR sidecar {sidecar_path}: {e}")
        return None, None


def _get_stored_sidecar_mtime(document: "Document") -> Optional[float]:
    """Get the stored OCR sidecar mtime from a document's extraction_structure."""
    structure = document.extraction_structure
    if not structure or not isinstance(structure, dict):
        return None
    return structure.get("ocr_sidecar_mtime")


def _ocr_quality_to_float(quality: str) -> float:
    """Map OCR quality string to numeric score."""
    return {"high": 0.95, "medium": 0.70, "low": 0.40}.get(
        str(quality).lower(), 0.70
    )


@dataclass
class FileInfo:
    """Information about a discovered file."""
    path: Path
    size: int
    mtime: float
    extension: str

    @property
    def mtime_datetime(self) -> datetime:
        """Return mtime as datetime."""
        return datetime.fromtimestamp(self.mtime)


@dataclass
class ScanResult:
    """Result of a folder scan operation."""
    total_files_found: int = 0
    files_processed: int = 0
    new_documents: int = 0
    updated_documents: int = 0
    duplicate_documents: int = 0
    reindexed: int = 0
    texts_extracted: int = 0
    ocr_sidecars_applied: int = 0
    embeddings_generated: int = 0
    skipped_unchanged: int = 0
    skipped_too_large: int = 0
    skipped_excluded: int = 0
    errors: int = 0
    error_details: list = field(default_factory=list)
    scan_duration_seconds: float = 0.0

    def add_error(self, path: str, error: str):
        """Add an error to the result."""
        self.errors += 1
        self.error_details.append({"path": path, "error": error})

    def progress_summary(self, total_to_process: int, elapsed_seconds: float) -> str:
        """Return a compact progress line."""
        remaining = total_to_process - self.files_processed
        rate = self.files_processed / elapsed_seconds if elapsed_seconds > 0 else 0
        eta = remaining / rate if rate > 0 else 0
        return (
            f"[{self.files_processed}/{total_to_process}] "
            f"{self.new_documents} new, {self.updated_documents} updated, "
            f"{self.ocr_sidecars_applied} OCR sidecars, {self.errors} errors "
            f"({rate:.1f} files/s, ~{eta:.0f}s remaining)"
        )

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Scan complete:",
            f"  Files found:          {self.total_files_found}",
            f"  Processed:            {self.files_processed}",
            f"  New documents:        {self.new_documents}",
            f"  Updated documents:    {self.updated_documents}",
            f"  Already known:        {self.duplicate_documents}",
            f"  OCR sidecars applied: {self.ocr_sidecars_applied}",
            f"  Texts extracted:      {self.texts_extracted}",
            f"  Embeddings generated: {self.embeddings_generated}",
        ]
        if self.reindexed > 0:
            lines.append(f"  Reindexed:            {self.reindexed}")
        lines.extend([
            f"  Skipped (unchanged):  {self.skipped_unchanged}",
            f"  Skipped (too large):  {self.skipped_too_large}",
            f"  Skipped (excluded):   {self.skipped_excluded}",
            f"  Errors:               {self.errors}",
            f"  Duration:             {self.scan_duration_seconds:.1f}s",
        ])
        return "\n".join(lines)


class ScanCache:
    """
    Cache for tracking file state between scans.

    Stores file path -> (mtime, size, checksum) mappings to enable
    incremental scans that skip unchanged files.
    """

    def __init__(self, cache_path: Optional[Path] = None):
        """
        Initialize scan cache.

        Args:
            cache_path: Path to cache file. If None, uses in-memory cache only.
        """
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._dirty = False

        if cache_path and cache_path.exists():
            self._load()

    def _load(self):
        """Load cache from disk."""
        try:
            with open(self.cache_path) as f:
                self._cache = json.load(f)
            logger.debug(f"Loaded scan cache with {len(self._cache)} entries")
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load scan cache: {e}")
            self._cache = {}

    def save(self):
        """Save cache to disk if dirty."""
        if not self._dirty or not self.cache_path:
            return

        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f)
            self._dirty = False
            logger.debug(f"Saved scan cache with {len(self._cache)} entries")
        except IOError as e:
            logger.error(f"Failed to save scan cache: {e}")

    def should_process(self, file_info: FileInfo) -> bool:
        """
        Check if file should be processed (has changed since last scan).

        Args:
            file_info: Information about the file

        Returns:
            True if file should be processed, False if unchanged
        """
        key = str(file_info.path)
        cached = self._cache.get(key)

        if not cached:
            return True

        # Check if mtime or size changed
        if cached.get("mtime") != file_info.mtime:
            return True
        if cached.get("size") != file_info.size:
            return True

        return False

    def mark_processed(self, file_info: FileInfo, checksum: str):
        """
        Mark file as processed with its checksum.

        Args:
            file_info: Information about the file
            checksum: SHA-256 checksum of file content
        """
        key = str(file_info.path)
        self._cache[key] = {
            "mtime": file_info.mtime,
            "size": file_info.size,
            "checksum": checksum,
            "scanned_at": datetime.utcnow().isoformat(),
        }
        self._dirty = True

    def get_checksum(self, path: Path) -> Optional[str]:
        """Get cached checksum for a file path."""
        cached = self._cache.get(str(path))
        return cached.get("checksum") if cached else None

    def get_all_paths(self) -> set[str]:
        """Get all cached file paths."""
        return set(self._cache.keys())

    def remove(self, path: str):
        """Remove a path from the cache."""
        if path in self._cache:
            del self._cache[path]
            self._dirty = True

    def clear(self):
        """Clear all cached entries."""
        self._cache.clear()
        self._dirty = True


class FolderScanner:
    """
    Scanner for discovering and registering documents from folders.
    """

    def __init__(
        self,
        processor: DocumentProcessor,
        cache: Optional[ScanCache] = None,
        embedding_service: Optional["DocumentEmbeddingService"] = None,
    ):
        """
        Initialize folder scanner.

        Args:
            processor: DocumentProcessor for registering documents
            cache: Optional ScanCache for incremental scanning
            embedding_service: Optional embedding service for generating embeddings inline
        """
        self.processor = processor
        self.cache = cache
        self.embedding_service = embedding_service

    async def _apply_ocr_sidecar(
        self,
        document: Document,
        sidecar: dict,
        sidecar_mtime: float,
        result: "ScanResult",
        filename: str,
    ) -> bool:
        """
        Apply OCR sidecar data to a document.

        Updates content analysis, provides OCR text, sets ocr_applied flag,
        stores sidecar mtime for change detection, updates metadata,
        and generates embeddings.

        Returns:
            True if sidecar was successfully applied.
        """
        repo = self.processor.repository
        doc_id = document.id

        # 1. Build OCR text from pages (validate early, before modifying DB)
        pages = sidecar.get("pages", [])
        if not pages:
            logger.warning(f"OCR sidecar for {filename} has no pages")
            return False

        full_text = "\n\n".join(p.get("text", "") for p in pages)
        if not full_text.strip():
            logger.warning(f"OCR sidecar for {filename} has empty text")
            return False

        ocr_structure = [
            {"page": p.get("page", i + 1), "text": p.get("text", "")}
            for i, p in enumerate(pages)
        ]

        ocr_method = sidecar.get("ocr_method", "external_ocr")
        ocr_quality = _ocr_quality_to_float(sidecar.get("ocr_quality", "medium"))

        # 2. Content analysis from sidecar
        await repo.update_content_analysis(
            document_id=doc_id,
            has_images=sidecar.get("has_images"),
            has_native_text=sidecar.get("has_native_text"),
            is_image_only=sidecar.get("is_image_only"),
            is_scanned_with_ocr=sidecar.get("is_scanned_with_ocr"),
            ocr_recommended=False,  # OCR is being applied now
            text_source="ocr",
        )

        # 3. Provide OCR text (handles quality comparison, encryption, status)
        # Note: this also queues a regenerate_embedding task as a side effect
        was_updated, embeddings_queued = await self.processor.provide_ocr_text(
            document_id=doc_id,
            ocr_text=full_text,
            ocr_method=ocr_method,
            ocr_quality=ocr_quality,
            ocr_structure=ocr_structure,
            force=True,
        )

        if not was_updated:
            logger.debug(f"OCR sidecar for {filename}: text not updated (quality check)")
            return False

        # 4. Set ocr_applied flag + pipeline version + store sidecar mtime
        #    provide_ocr_text does NOT set ocr_applied — we must do it here.
        #    Use dict() copy so SQLAlchemy detects the JSON mutation on plain Column(JSON).
        doc_updated = await repo.get_by_id(doc_id)
        if doc_updated:
            doc_updated.ocr_applied = True
            version = sidecar.get("ocr_version") or "v1.0"
            doc_updated.ocr_pipeline_version = version
            doc_updated.extraction_version = version
            # Store sidecar mtime in extraction_structure for change detection
            structure = dict(doc_updated.extraction_structure or {})
            structure["ocr_sidecar_mtime"] = sidecar_mtime
            doc_updated.extraction_structure = structure

        # 5. Metadata
        from datetime import date as date_type
        doc_date = None
        raw_date = sidecar.get("document_date")
        if raw_date:
            try:
                doc_date = date_type.fromisoformat(raw_date)
            except (ValueError, TypeError):
                pass

        await repo.update_document_metadata(
            document_id=doc_id,
            title=sidecar.get("document_title"),
            summary=sidecar.get("summary"),
            document_date=doc_date,
            document_type=sidecar.get("document_type"),
            language=sidecar.get("language"),
            page_count=sidecar.get("num_pages"),
        )

        # 6. Generate embeddings inline (if service available)
        #    provide_ocr_text queued a regenerate_embedding task; if we generate
        #    inline, cancel that task to avoid redundant double-generation.
        if self.embedding_service:
            try:
                doc_for_embed = await repo.get_by_id(doc_id)
                if doc_for_embed:
                    embed_result = await self.embedding_service.generate_document_embedding(
                        doc_for_embed,
                        single_embedding=self.embedding_service.single_embedding,
                    )
                    result.embeddings_generated += embed_result.embeddings_created
                    logger.info(f"Generated {embed_result.embeddings_created} embeddings for {filename} (from OCR sidecar)")
                    # Cancel the queued regenerate_embedding task since we just did it inline
                    await repo.cancel_pending_tasks(doc_id, "regenerate_embedding")
            except Exception as e:
                logger.warning(f"Embedding generation failed for {filename}: {e}")
                # Leave the queued task — background processor will retry

        result.ocr_sidecars_applied += 1
        logger.info(f"Applied OCR sidecar for {filename}: {len(pages)} pages, quality={ocr_quality}")
        return True

    def discover_files(
        self,
        config: FolderScanConfig,
    ) -> Iterator[FileInfo]:
        """
        Discover files matching scan configuration.

        Args:
            config: Scan configuration

        Yields:
            FileInfo for each matching file
        """
        base_path = Path(config.base_path)
        if not base_path.exists():
            logger.error(f"Base path does not exist: {base_path}")
            return

        extensions = set(ext.lower().lstrip('.') for ext in config.effective_extensions)
        exclude_patterns = config.effective_exclude_patterns
        max_size = config.effective_max_file_size_bytes

        def should_exclude(path: Path) -> bool:
            """Check if path matches any exclusion pattern."""
            path_str = str(path)
            for pattern in exclude_patterns:
                if fnmatch.fnmatch(path_str, pattern):
                    return True
                if fnmatch.fnmatch(path.name, pattern):
                    return True
            return False

        def is_hidden(path: Path) -> bool:
            """Check if path or any parent is hidden."""
            for part in path.parts:
                if part.startswith('.') and part not in ('.', '..'):
                    return True
            return False

        def scan_directory(dir_path: Path, depth: int = 0) -> Iterator[FileInfo]:
            """Recursively scan a directory."""
            try:
                entries = list(dir_path.iterdir())
            except PermissionError:
                logger.warning(f"Permission denied: {dir_path}")
                return
            except OSError as e:
                logger.warning(f"Error reading directory {dir_path}: {e}")
                return

            for entry in entries:
                try:
                    # Skip hidden files/directories if configured
                    if config.skip_hidden and is_hidden(entry):
                        continue

                    # Skip symlinks if not following
                    if entry.is_symlink() and not config.follow_symlinks:
                        continue

                    # Handle directories
                    if entry.is_dir():
                        if config.recursive and not should_exclude(entry):
                            yield from scan_directory(entry, depth + 1)
                        continue

                    # Handle files
                    if not entry.is_file():
                        continue

                    # Check exclusion patterns
                    if should_exclude(entry):
                        continue

                    # Check extension
                    ext = entry.suffix.lower().lstrip('.')
                    if ext not in extensions:
                        continue

                    # Get file stats
                    try:
                        stat = entry.stat()
                    except OSError as e:
                        logger.warning(f"Cannot stat file {entry}: {e}")
                        continue

                    # Check file size
                    if stat.st_size > max_size:
                        continue

                    yield FileInfo(
                        path=entry,
                        size=stat.st_size,
                        mtime=stat.st_mtime,
                        extension=ext,
                    )

                except Exception as e:
                    logger.warning(f"Error processing {entry}: {e}")

        yield from scan_directory(base_path)

    async def scan(
        self,
        config: FolderScanConfig,
        limit: Optional[int] = None,
        dry_run: bool = False,
        extract_text: bool = True,
        progress_callback: Optional[callable] = None,
        reindex: bool = False,
        commit_interval: int = 50,
        skip_ocr_sidecars: bool = False,
        progress_interval: int = 20,
    ) -> ScanResult:
        """
        Scan a folder and register discovered documents.

        Args:
            config: Scan configuration
            limit: Maximum number of files to process
            dry_run: If True, don't actually register documents
            extract_text: If True, extract text from documents immediately (default: True)
            progress_callback: Optional callback(file_info, result) for progress
            reindex: If True, re-extract text and regenerate embeddings for existing documents
            commit_interval: Commit to database every N files (default: 50). Set to 0 to disable.
            skip_ocr_sidecars: If True, ignore .ocr.json sidecar files (default: False)
            progress_interval: Log progress summary every N processed files (default: 20)

        Returns:
            ScanResult with scan statistics
        """
        import time
        start_time = time.time()

        result = ScanResult()
        processed_count = 0
        files_since_commit = 0
        reindexed_document_ids = set()  # Track already reindexed documents to avoid duplicates

        # Collect files to process (enables total count for progress reporting)
        logger.info("Discovering files...")
        all_files = []
        for file_info in self.discover_files(config):
            result.total_files_found += 1
            if limit and len(all_files) >= limit:
                continue  # keep counting total but stop collecting

            if not reindex and self.cache and not self.cache.should_process(file_info):
                result.skipped_unchanged += 1
                continue

            if file_info.size > config.effective_max_file_size_bytes:
                result.skipped_too_large += 1
                continue

            all_files.append(file_info)

        total_to_process = len(all_files)
        logger.info(
            f"Found {result.total_files_found} files, {total_to_process} to process "
            f"({result.skipped_unchanged} unchanged, {result.skipped_too_large} too large)"
        )

        for file_info in all_files:
            # Progress callback
            if progress_callback:
                progress_callback(file_info, result)

            # Dry run - just count
            if dry_run:
                result.files_processed += 1
                processed_count += 1
                continue

            # Register document (handles both new files and content changes)
            try:
                # Calculate checksum first to detect changes
                checksum = self.processor.calculate_checksum(str(file_info.path))
                mime_type = self.processor.get_mime_type(str(file_info.path))

                document, is_new, orphaned_id = await self.processor.handle_file_change(
                    origin_type="folder",
                    origin_host=config.host.name,
                    file_path=str(file_info.path),
                    new_checksum=checksum,
                    file_size=file_info.size,
                    mime_type=mime_type,
                    file_modified_at=file_info.mtime_datetime,
                )

                if orphaned_id:
                    logger.info(f"Document {orphaned_id} orphaned after file change")

                if is_new:
                    result.new_documents += 1

                    # Check for OCR sidecar file
                    sidecar_applied = False
                    if not skip_ocr_sidecars and file_info.extension in OCR_SIDECAR_EXTENSIONS:
                        sidecar, sidecar_mtime = _load_ocr_sidecar(file_info.path)
                        if sidecar:
                            try:
                                sidecar_applied = await self._apply_ocr_sidecar(
                                    document, sidecar, sidecar_mtime, result, file_info.path.name,
                                )
                                if sidecar_applied:
                                    # Cancel the extract_text task queued by handle_file_change —
                                    # OCR sidecar already provided the text, no need for native extraction.
                                    await self.processor.repository.cancel_pending_tasks(
                                        document.id, "extract_text"
                                    )
                            except Exception as e:
                                logger.warning(f"OCR sidecar failed for {file_info.path.name}: {e}")
                                try:
                                    self.processor.repository.db.rollback()
                                except Exception:
                                    pass

                    # Fall back to normal extraction if no sidecar or sidecar failed
                    if not sidecar_applied and extract_text:
                        try:
                            with open(file_info.path, 'rb') as f:
                                file_content = f.read()

                            # Analyze content for OCR detection
                            from backend.core.documents.content_analyzer import analyze_content
                            content_analysis = analyze_content(
                                file_content,
                                mime_type,
                                filename=file_info.path.name,
                            )

                            # Update content analysis fields
                            await self.processor.repository.update_content_analysis(
                                document_id=document.id,
                                has_images=content_analysis.has_images,
                                has_native_text=content_analysis.has_native_text,
                                is_image_only=content_analysis.is_image_only,
                                is_scanned_with_ocr=content_analysis.is_scanned_with_ocr,
                                ocr_recommended=content_analysis.ocr_recommended,
                                text_source='native' if content_analysis.has_native_text else 'none',
                            )

                            # Use structured extraction for multi-page/multi-sheet documents
                            extraction_result = await self.processor.extract_with_structure(
                                document=document,
                                file_content=file_content,
                            )

                            if extraction_result.text:
                                # Update document with extracted text
                                await self.processor.repository.update_extraction(
                                    document_id=document.id,
                                    extracted_text=extraction_result.text,
                                    extraction_status=ExtractionStatus.COMPLETED,
                                    extraction_method=extraction_result.method,
                                    extraction_quality=extraction_result.quality_score,
                                    page_count=extraction_result.page_count,
                                )
                                result.texts_extracted += 1

                                # Store structured data for embedding generation
                                if extraction_result.has_structure:
                                    await self.processor.repository.store_extraction_structure(
                                        document_id=document.id,
                                        pages=extraction_result.pages,
                                        sheets=extraction_result.sheets,
                                        sections=extraction_result.sections,
                                    )
                                    if extraction_result.pages:
                                        logger.debug(f"Extracted {len(extraction_result.pages)} pages from {file_info.path.name}")
                                    elif extraction_result.sheets:
                                        logger.debug(f"Extracted {len(extraction_result.sheets)} sheets from {file_info.path.name}")
                                    elif extraction_result.sections:
                                        logger.debug(f"Extracted {len(extraction_result.sections)} sections from {file_info.path.name}")
                                else:
                                    logger.debug(f"Extracted {len(extraction_result.text)} chars from {file_info.path.name}")

                                # Generate embedding inline if service provided
                                if self.embedding_service:
                                    try:
                                        # Reload document to get updated extraction data
                                        doc_for_embed = await self.processor.repository.get_by_id(document.id)
                                        if doc_for_embed:
                                            embed_result = await self.embedding_service.generate_document_embedding(
                                                doc_for_embed,
                                                single_embedding=self.embedding_service.single_embedding,
                                            )
                                            result.embeddings_generated += embed_result.embeddings_created
                                            logger.info(f"Generated {embed_result.embeddings_created} embeddings for {file_info.path.name}")
                                    except Exception as e:
                                        logger.warning(f"Embedding generation failed for {file_info.path.name}: {e}")
                            else:
                                # No text extracted (binary file, unsupported format, etc.)
                                await self.processor.repository.update_extraction(
                                    document_id=document.id,
                                    extracted_text=None,
                                    extraction_status=ExtractionStatus.NO_CONTENT,
                                    extraction_method=extraction_result.method,
                                    extraction_quality=0.0,
                                )
                        except Exception as e:
                            logger.warning(f"Text extraction failed for {file_info.path.name}: {e}")
                            # Rollback to recover session after database errors
                            try:
                                self.processor.repository.db.rollback()
                            except Exception:
                                pass
                            try:
                                await self.processor.repository.update_extraction(
                                    document_id=document.id,
                                    extracted_text=None,
                                    extraction_status=ExtractionStatus.FAILED,
                                    extraction_method="error",
                                    extraction_quality=0.0,
                                )
                            except Exception:
                                pass
                else:
                    result.duplicate_documents += 1

                    # Check for OCR sidecar on existing documents:
                    # - Apply if OCR hasn't been applied yet
                    # - Re-apply if the sidecar file is newer than the stored version
                    if not skip_ocr_sidecars and file_info.extension in OCR_SIDECAR_EXTENSIONS:
                        sidecar, sidecar_mtime = _load_ocr_sidecar(file_info.path)
                        if sidecar:
                            stored_mtime = _get_stored_sidecar_mtime(document)
                            should_apply = (
                                not document.ocr_applied
                                or stored_mtime is None
                                or sidecar_mtime > stored_mtime
                            )
                            if should_apply:
                                if stored_mtime and sidecar_mtime > stored_mtime:
                                    logger.info(f"OCR sidecar updated for {file_info.path.name} (new mtime)")
                                try:
                                    applied = await self._apply_ocr_sidecar(
                                        document, sidecar, sidecar_mtime, result, file_info.path.name,
                                    )
                                    if applied:
                                        result.updated_documents += 1
                                except Exception as e:
                                    logger.warning(f"OCR sidecar failed for existing {file_info.path.name}: {e}")
                                    try:
                                        self.processor.repository.db.rollback()
                                    except Exception:
                                        pass
                            elif document.ocr_applied and stored_mtime and sidecar_mtime != stored_mtime:
                                # Different sidecar at a duplicate file location — log but don't overwrite
                                logger.warning(
                                    f"Duplicate file {file_info.path.name} has different OCR sidecar "
                                    f"than already applied (mtime {sidecar_mtime} vs stored {stored_mtime}), skipping"
                                )

                    # Reindex existing document if requested (skip if already reindexed this session)
                    if reindex and extract_text and document.id not in reindexed_document_ids:
                        result.reindexed += 1
                        result.updated_documents += 1
                        try:
                            with open(file_info.path, 'rb') as f:
                                file_content = f.read()

                            # Analyze content for OCR detection
                            from backend.core.documents.content_analyzer import analyze_content
                            content_analysis = analyze_content(
                                file_content,
                                mime_type,
                                filename=file_info.path.name,
                            )

                            # Update content analysis fields
                            await self.processor.repository.update_content_analysis(
                                document_id=document.id,
                                has_images=content_analysis.has_images,
                                has_native_text=content_analysis.has_native_text,
                                is_image_only=content_analysis.is_image_only,
                                is_scanned_with_ocr=content_analysis.is_scanned_with_ocr,
                                ocr_recommended=content_analysis.ocr_recommended,
                                text_source='native' if content_analysis.has_native_text else 'none',
                            )

                            extraction_result = await self.processor.extract_with_structure(
                                document=document,
                                file_content=file_content,
                            )

                            if extraction_result.text:
                                await self.processor.repository.update_extraction(
                                    document_id=document.id,
                                    extracted_text=extraction_result.text,
                                    extraction_status=ExtractionStatus.COMPLETED,
                                    extraction_method=extraction_result.method,
                                    extraction_quality=extraction_result.quality_score,
                                    page_count=extraction_result.page_count,
                                )
                                result.texts_extracted += 1

                                if extraction_result.has_structure:
                                    await self.processor.repository.store_extraction_structure(
                                        document_id=document.id,
                                        pages=extraction_result.pages,
                                        sheets=extraction_result.sheets,
                                        sections=extraction_result.sections,
                                    )

                                # Regenerate embeddings
                                if self.embedding_service:
                                    try:
                                        # Delete old embeddings first
                                        await self.processor.repository.delete_embeddings(document.id)
                                        doc_for_embed = await self.processor.repository.get_by_id(document.id)
                                        if doc_for_embed:
                                            embed_result = await self.embedding_service.generate_document_embedding(
                                                doc_for_embed,
                                                single_embedding=self.embedding_service.single_embedding,
                                            )
                                            result.embeddings_generated += embed_result.embeddings_created
                                            logger.info(f"Reindexed {file_info.path.name}: {embed_result.embeddings_created} embeddings")
                                    except Exception as e:
                                        logger.warning(f"Embedding regeneration failed for {file_info.path.name}: {e}")

                                # Mark document as reindexed to avoid re-processing copies
                                reindexed_document_ids.add(document.id)
                        except Exception as e:
                            logger.warning(f"Reindex failed for {file_info.path.name}: {e}")
                            # Rollback to recover session after database errors
                            try:
                                self.processor.repository.db.rollback()
                            except Exception:
                                pass

                result.files_processed += 1
                processed_count += 1
                files_since_commit += 1

                # Periodic progress summary (after first file, then every N files)
                if progress_interval > 0 and (
                    result.files_processed == 1
                    or result.files_processed % progress_interval == 0
                ):
                    elapsed = time.time() - start_time
                    logger.info(result.progress_summary(total_to_process, elapsed))

                # Batch commit to make progress visible and recoverable
                if not dry_run and commit_interval > 0 and files_since_commit >= commit_interval:
                    try:
                        self.processor.repository.db.commit()
                        logger.info(f"Committed batch of {files_since_commit} files (total: {processed_count})")
                        files_since_commit = 0
                    except Exception as e:
                        logger.error(f"Batch commit failed: {e}")
                        # Continue processing - will retry commit on next interval

                # Update cache
                if self.cache:
                    checksum = self.processor.calculate_checksum(str(file_info.path))
                    self.cache.mark_processed(file_info, checksum)

            except FileNotFoundError:
                result.add_error(str(file_info.path), "File not found")
            except PermissionError:
                result.add_error(str(file_info.path), "Permission denied")
            except Exception as e:
                result.add_error(str(file_info.path), str(e))
                logger.error(f"Error processing {file_info.path}: {e}")
                # Rollback to recover from database errors (e.g., null character in JSON)
                try:
                    self.processor.repository.db.rollback()
                except Exception:
                    pass

        # Save cache
        if self.cache:
            self.cache.save()

        result.scan_duration_seconds = time.time() - start_time
        return result

    async def scan_for_deleted(
        self,
        config: FolderScanConfig,
        repository: DocumentRepository,
    ) -> int:
        """
        Find origins that no longer exist on disk and mark them deleted.

        Args:
            config: Scan configuration
            repository: Document repository

        Returns:
            Number of origins marked as deleted
        """
        if not self.cache:
            logger.warning("No cache available for deletion detection")
            return 0

        # Get all paths we've seen before
        cached_paths = self.cache.get_all_paths()

        # Get all paths that currently exist
        current_paths = set()
        for file_info in self.discover_files(config):
            current_paths.add(str(file_info.path))

        # Find deleted paths
        deleted_paths = cached_paths - current_paths
        deleted_count = 0

        for path in deleted_paths:
            # Only process paths under our base path
            if not path.startswith(config.base_path):
                continue

            try:
                # Mark origin as deleted in database
                await repository.mark_origin_deleted(
                    origin_type="folder",
                    origin_host=config.host.name,
                    origin_path=str(Path(path).parent),
                    origin_filename=Path(path).name,
                )
                deleted_count += 1

                # Remove from cache
                self.cache.remove(path)

            except Exception as e:
                logger.error(f"Error marking origin deleted for {path}: {e}")

        if self.cache:
            self.cache.save()

        return deleted_count


def create_scanner(
    repository: DocumentRepository,
    cache_dir: Optional[Path] = None,
    generate_embeddings: bool = True,
    single_embedding: bool = False,
) -> FolderScanner:
    """
    Create a folder scanner with appropriate cache.

    Args:
        repository: Document repository
        cache_dir: Directory for scan caches (default: .scan_cache in cwd)
        generate_embeddings: Whether to generate embeddings inline (default: True)
        single_embedding: If True, generate only one embedding per document (default: False)
                         If False, generate per-page/per-sheet embeddings for multi-page docs

    Returns:
        Configured FolderScanner
    """
    processor = DocumentProcessor(repository)

    cache = None
    if cache_dir:
        cache_path = cache_dir / "scan_cache.json"
        cache = ScanCache(cache_path)

    embedding_service = None
    if generate_embeddings:
        from backend.core.documents.embeddings import DocumentEmbeddingService
        embedding_service = DocumentEmbeddingService(repository, single_embedding=single_embedding)

    return FolderScanner(processor, cache, embedding_service)
