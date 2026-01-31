"""
Tests for Folder Scanner.

Phase 2 tests for:
- File discovery (recursive, extensions, exclusions)
- Scan cache (incremental scanning)
- Document registration during scan
- Deleted file detection
"""

import pytest
import tempfile
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import MagicMock, AsyncMock, patch
import json

from backend.core.documents.folder_scanner import (
    FolderScanner,
    ScanCache,
    FileInfo,
    ScanResult,
    create_scanner,
)
from backend.core.documents.config import HostConfig, FolderScanConfig


@pytest.fixture
def temp_folder():
    """Create a temporary folder structure for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)

        # Create test files
        (root / "doc1.pdf").write_bytes(b"PDF content 1")
        (root / "doc2.docx").write_bytes(b"DOCX content")
        (root / "image.png").write_bytes(b"PNG image data")
        (root / "readme.txt").write_bytes(b"Text content")

        # Create subdirectory
        subdir = root / "subdir"
        subdir.mkdir()
        (subdir / "doc3.pdf").write_bytes(b"PDF content 3")
        (subdir / "temp.tmp").write_bytes(b"Temp file")

        # Create hidden file
        (root / ".hidden").write_bytes(b"Hidden content")

        # Create hidden directory
        hidden_dir = root / ".hidden_dir"
        hidden_dir.mkdir()
        (hidden_dir / "secret.pdf").write_bytes(b"Secret PDF")

        yield root


@pytest.fixture
def host_config():
    """Create a test host configuration."""
    return HostConfig(
        name="test-host",
        type="local",
        mount_point="/",
        allowed_extensions=["pdf", "docx", "txt", "png"],
        exclude_patterns=["*.tmp", "*.bak"],
    )


@pytest.fixture
def scan_config(host_config, temp_folder):
    """Create a test scan configuration."""
    return FolderScanConfig(
        host=host_config,
        base_path=str(temp_folder),
        recursive=True,
        skip_hidden=True,
    )


@pytest.fixture
def mock_processor():
    """Create a mock document processor."""
    processor = MagicMock()
    processor.register_document = AsyncMock(return_value=(MagicMock(), True))
    processor.calculate_checksum = MagicMock(return_value="abc123")
    return processor


class TestFileInfo:
    """Tests for FileInfo dataclass."""

    def test_file_info_creation(self):
        """Should create FileInfo with correct attributes."""
        info = FileInfo(
            path=Path("/test/file.pdf"),
            size=1024,
            mtime=1234567890.0,
            extension="pdf",
        )
        assert info.path == Path("/test/file.pdf")
        assert info.size == 1024
        assert info.extension == "pdf"

    def test_mtime_datetime(self):
        """Should convert mtime to datetime."""
        info = FileInfo(
            path=Path("/test/file.pdf"),
            size=1024,
            mtime=1234567890.0,
            extension="pdf",
        )
        assert isinstance(info.mtime_datetime, datetime)


class TestScanResult:
    """Tests for ScanResult dataclass."""

    def test_scan_result_defaults(self):
        """Should have correct default values."""
        result = ScanResult()
        assert result.total_files_found == 0
        assert result.files_processed == 0
        assert result.new_documents == 0
        assert result.errors == 0
        assert result.error_details == []

    def test_add_error(self):
        """Should track errors correctly."""
        result = ScanResult()
        result.add_error("/path/to/file", "File not readable")

        assert result.errors == 1
        assert len(result.error_details) == 1
        assert result.error_details[0]["path"] == "/path/to/file"

    def test_summary(self):
        """Should generate readable summary."""
        result = ScanResult(
            total_files_found=100,
            files_processed=90,
            new_documents=50,
            duplicate_documents=40,
            errors=10,
        )
        summary = result.summary()

        assert "100" in summary
        assert "90" in summary
        assert "50" in summary


class TestScanCache:
    """Tests for ScanCache."""

    def test_cache_in_memory(self):
        """Should work without file persistence."""
        cache = ScanCache()

        file_info = FileInfo(
            path=Path("/test/file.pdf"),
            size=1024,
            mtime=1234567890.0,
            extension="pdf",
        )

        # Initially should need processing
        assert cache.should_process(file_info)

        # Mark as processed
        cache.mark_processed(file_info, "checksum123")

        # Now should not need processing
        assert not cache.should_process(file_info)

    def test_cache_detects_mtime_change(self):
        """Should detect file modification."""
        cache = ScanCache()

        file_info = FileInfo(
            path=Path("/test/file.pdf"),
            size=1024,
            mtime=1234567890.0,
            extension="pdf",
        )

        cache.mark_processed(file_info, "checksum123")

        # Change mtime
        file_info_modified = FileInfo(
            path=Path("/test/file.pdf"),
            size=1024,
            mtime=1234567891.0,  # Different mtime
            extension="pdf",
        )

        assert cache.should_process(file_info_modified)

    def test_cache_detects_size_change(self):
        """Should detect file size change."""
        cache = ScanCache()

        file_info = FileInfo(
            path=Path("/test/file.pdf"),
            size=1024,
            mtime=1234567890.0,
            extension="pdf",
        )

        cache.mark_processed(file_info, "checksum123")

        # Change size
        file_info_modified = FileInfo(
            path=Path("/test/file.pdf"),
            size=2048,  # Different size
            mtime=1234567890.0,
            extension="pdf",
        )

        assert cache.should_process(file_info_modified)

    def test_cache_persistence(self):
        """Should persist to and load from file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.json"

            # Create and populate cache
            cache1 = ScanCache(cache_path)
            file_info = FileInfo(
                path=Path("/test/file.pdf"),
                size=1024,
                mtime=1234567890.0,
                extension="pdf",
            )
            cache1.mark_processed(file_info, "checksum123")
            cache1.save()

            # Load in new instance
            cache2 = ScanCache(cache_path)
            assert not cache2.should_process(file_info)
            assert cache2.get_checksum(Path("/test/file.pdf")) == "checksum123"

    def test_cache_get_all_paths(self):
        """Should return all cached paths."""
        cache = ScanCache()

        for i in range(3):
            file_info = FileInfo(
                path=Path(f"/test/file{i}.pdf"),
                size=1024,
                mtime=1234567890.0,
                extension="pdf",
            )
            cache.mark_processed(file_info, f"checksum{i}")

        paths = cache.get_all_paths()
        assert len(paths) == 3
        assert "/test/file0.pdf" in paths

    def test_cache_remove(self):
        """Should remove entries from cache."""
        cache = ScanCache()

        file_info = FileInfo(
            path=Path("/test/file.pdf"),
            size=1024,
            mtime=1234567890.0,
            extension="pdf",
        )
        cache.mark_processed(file_info, "checksum123")

        cache.remove("/test/file.pdf")
        assert cache.should_process(file_info)


class TestFolderScannerDiscovery:
    """Tests for file discovery."""

    def test_discover_files_recursive(self, temp_folder, host_config):
        """Should find files recursively."""
        config = FolderScanConfig(
            host=host_config,
            base_path=str(temp_folder),
            recursive=True,
            skip_hidden=True,
        )

        scanner = FolderScanner(MagicMock())
        files = list(scanner.discover_files(config))

        # Should find: doc1.pdf, doc2.docx, image.png, readme.txt, subdir/doc3.pdf
        # Should NOT find: .hidden, .hidden_dir/secret.pdf, subdir/temp.tmp
        assert len(files) == 5
        filenames = {f.path.name for f in files}
        assert "doc1.pdf" in filenames
        assert "doc3.pdf" in filenames
        assert "temp.tmp" not in filenames

    def test_discover_files_non_recursive(self, temp_folder, host_config):
        """Should only find files in top directory."""
        config = FolderScanConfig(
            host=host_config,
            base_path=str(temp_folder),
            recursive=False,
            skip_hidden=True,
        )

        scanner = FolderScanner(MagicMock())
        files = list(scanner.discover_files(config))

        # Should only find top-level files
        assert len(files) == 4
        filenames = {f.path.name for f in files}
        assert "doc1.pdf" in filenames
        assert "doc3.pdf" not in filenames  # In subdirectory

    def test_discover_files_with_extension_filter(self, temp_folder, host_config):
        """Should filter by extension."""
        host_config.allowed_extensions = ["pdf"]
        config = FolderScanConfig(
            host=host_config,
            base_path=str(temp_folder),
            recursive=True,
            skip_hidden=True,
        )

        scanner = FolderScanner(MagicMock())
        files = list(scanner.discover_files(config))

        # Should only find PDFs
        assert len(files) == 2
        for f in files:
            assert f.extension == "pdf"

    def test_discover_files_exclude_patterns(self, temp_folder, host_config):
        """Should exclude files matching patterns."""
        host_config.exclude_patterns = ["*.pdf"]
        config = FolderScanConfig(
            host=host_config,
            base_path=str(temp_folder),
            recursive=True,
            skip_hidden=True,
        )

        scanner = FolderScanner(MagicMock())
        files = list(scanner.discover_files(config))

        # Should not find any PDFs
        for f in files:
            assert f.extension != "pdf"

    def test_discover_files_skip_hidden(self, temp_folder, host_config):
        """Should skip hidden files and directories."""
        config = FolderScanConfig(
            host=host_config,
            base_path=str(temp_folder),
            recursive=True,
            skip_hidden=True,
        )

        scanner = FolderScanner(MagicMock())
        files = list(scanner.discover_files(config))

        # Should not find hidden files
        filenames = {f.path.name for f in files}
        assert ".hidden" not in filenames

        # Should not find files in hidden directories
        paths = {str(f.path) for f in files}
        assert not any(".hidden_dir" in p for p in paths)

    def test_discover_files_include_hidden(self, temp_folder, host_config):
        """Should include hidden files when configured."""
        config = FolderScanConfig(
            host=host_config,
            base_path=str(temp_folder),
            recursive=True,
            skip_hidden=False,  # Include hidden
        )

        scanner = FolderScanner(MagicMock())
        files = list(scanner.discover_files(config))

        # Should find more files including hidden ones
        paths = {str(f.path) for f in files}
        assert any(".hidden_dir" in p for p in paths)

    def test_discover_files_max_size(self, temp_folder, host_config):
        """Should skip files over max size."""
        # Create a large file
        large_file = temp_folder / "large.pdf"
        large_file.write_bytes(b"x" * (2 * 1024 * 1024))  # 2MB

        host_config.max_file_size_mb = 1  # 1MB limit
        config = FolderScanConfig(
            host=host_config,
            base_path=str(temp_folder),
            recursive=True,
            skip_hidden=True,
        )

        scanner = FolderScanner(MagicMock())
        files = list(scanner.discover_files(config))

        filenames = {f.path.name for f in files}
        assert "large.pdf" not in filenames

    def test_discover_files_nonexistent_path(self, host_config):
        """Should handle nonexistent paths gracefully."""
        config = FolderScanConfig(
            host=host_config,
            base_path="/nonexistent/path",
            recursive=True,
        )

        scanner = FolderScanner(MagicMock())
        files = list(scanner.discover_files(config))

        assert len(files) == 0


class TestFolderScannerScan:
    """Tests for scan operation."""

    @pytest.mark.asyncio
    async def test_scan_registers_new_documents(self, scan_config, mock_processor):
        """Should register new documents found during scan."""
        scanner = FolderScanner(mock_processor)
        result = await scanner.scan(scan_config)

        assert result.files_processed > 0
        assert mock_processor.register_document.call_count == result.files_processed

    @pytest.mark.asyncio
    async def test_scan_dry_run(self, scan_config, mock_processor):
        """Dry run should not register documents."""
        scanner = FolderScanner(mock_processor)
        result = await scanner.scan(scan_config, dry_run=True)

        assert result.files_processed > 0
        mock_processor.register_document.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_with_limit(self, scan_config, mock_processor):
        """Should respect limit parameter."""
        scanner = FolderScanner(mock_processor)
        result = await scanner.scan(scan_config, limit=2)

        assert result.files_processed <= 2

    @pytest.mark.asyncio
    async def test_scan_counts_duplicates(self, scan_config, mock_processor):
        """Should count duplicate documents separately."""
        # First call: new, second call: duplicate
        mock_processor.register_document.side_effect = [
            (MagicMock(), True),   # New
            (MagicMock(), False),  # Duplicate
            (MagicMock(), True),   # New
            (MagicMock(), False),  # Duplicate
            (MagicMock(), True),   # New
        ]

        scanner = FolderScanner(mock_processor)
        result = await scanner.scan(scan_config)

        assert result.new_documents == 3
        assert result.duplicate_documents == 2

    @pytest.mark.asyncio
    async def test_scan_handles_errors(self, scan_config, mock_processor):
        """Should handle and count errors."""
        mock_processor.register_document.side_effect = [
            (MagicMock(), True),
            Exception("Test error"),
            (MagicMock(), True),
        ]

        scanner = FolderScanner(mock_processor)
        result = await scanner.scan(scan_config)

        assert result.errors >= 1

    @pytest.mark.asyncio
    async def test_scan_incremental_with_cache(self, scan_config, mock_processor):
        """Should skip unchanged files with cache."""
        cache = ScanCache()

        scanner = FolderScanner(mock_processor, cache)

        # First scan
        result1 = await scanner.scan(scan_config)

        # Reset mock
        mock_processor.register_document.reset_mock()

        # Second scan should skip cached files
        result2 = await scanner.scan(scan_config)

        assert result2.skipped_unchanged == result1.files_processed
        assert result2.files_processed == 0

    @pytest.mark.asyncio
    async def test_scan_progress_callback(self, scan_config, mock_processor):
        """Should call progress callback."""
        callback_calls = []

        def progress(file_info, result):
            callback_calls.append((file_info, result))

        scanner = FolderScanner(mock_processor)
        await scanner.scan(scan_config, progress_callback=progress)

        assert len(callback_calls) > 0


class TestCreateScanner:
    """Tests for create_scanner factory function."""

    def test_create_scanner_without_cache(self):
        """Should create scanner without cache."""
        repo = MagicMock()
        scanner = create_scanner(repo, cache_dir=None)

        assert scanner.cache is None

    def test_create_scanner_with_cache(self):
        """Should create scanner with cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            repo = MagicMock()
            scanner = create_scanner(repo, cache_dir=Path(tmpdir))

            assert scanner.cache is not None
