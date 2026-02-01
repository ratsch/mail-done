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
    duplicate_documents: int = 0
    reindexed: int = 0
    texts_extracted: int = 0
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

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Scan complete:",
            f"  Files found: {self.total_files_found}",
            f"  Processed: {self.files_processed}",
            f"  New documents: {self.new_documents}",
            f"  Duplicates: {self.duplicate_documents}",
        ]
        if self.reindexed > 0:
            lines.append(f"  Reindexed: {self.reindexed}")
        if self.texts_extracted > 0:
            lines.append(f"  Texts extracted: {self.texts_extracted}")
        if self.embeddings_generated > 0:
            lines.append(f"  Embeddings generated: {self.embeddings_generated}")
        lines.extend([
            f"  Skipped (unchanged): {self.skipped_unchanged}",
            f"  Skipped (too large): {self.skipped_too_large}",
            f"  Skipped (excluded): {self.skipped_excluded}",
            f"  Errors: {self.errors}",
            f"  Duration: {self.scan_duration_seconds:.1f}s",
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

        Returns:
            ScanResult with scan statistics
        """
        import time
        start_time = time.time()

        result = ScanResult()
        processed_count = 0
        reindexed_document_ids = set()  # Track already reindexed documents to avoid duplicates

        for file_info in self.discover_files(config):
            result.total_files_found += 1

            # Check limit
            if limit and processed_count >= limit:
                break

            # Check if file changed (incremental scan) - skip check if reindexing
            if not reindex and self.cache and not self.cache.should_process(file_info):
                result.skipped_unchanged += 1
                continue

            # Check file size (double-check, discover_files should filter these)
            if file_info.size > config.effective_max_file_size_bytes:
                result.skipped_too_large += 1
                continue

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

                    # Extract text if enabled
                    if extract_text:
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

                    # Reindex existing document if requested (skip if already reindexed this session)
                    if reindex and extract_text and document.id not in reindexed_document_ids:
                        result.reindexed += 1
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
    single_embedding: bool = True,
) -> FolderScanner:
    """
    Create a folder scanner with appropriate cache.

    Args:
        repository: Document repository
        cache_dir: Directory for scan caches (default: .scan_cache in cwd)
        generate_embeddings: Whether to generate embeddings inline (default: True)
        single_embedding: If True, generate only one embedding per document (default: True)
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
