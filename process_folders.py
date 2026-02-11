#!/usr/bin/env python3
"""
Folder Scanner CLI

Scans folders on local or remote hosts to discover and index documents.
Similar to process_inbox.py but for file system documents.

Usage:
    python process_folders.py /path/to/folder --host localhost
    python process_folders.py /mnt/nas/documents --host nas --recursive
    python process_folders.py /home/user/docs --host remote-server --extensions pdf,docx

Examples:
    # Dry run to see what would be indexed
    python process_folders.py ~/Documents --dry-run

    # Scan with specific extensions
    python process_folders.py /data/invoices --extensions pdf,xlsx --limit 100

    # Full scan with progress
    python process_folders.py /archive --recursive --verbose
"""

import argparse
import asyncio
import logging
import os
import socket
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


def setup_logging(verbose: bool = False, quiet: bool = False):
    """Configure logging based on verbosity."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Default document extensions for indexing
DEFAULT_EXTENSIONS = ['pdf', 'docx', 'xlsx', 'pptx', 'doc', 'ppt', 'xls', 'csv', 'txt', 'md', 'jpg', 'jpeg', 'png']


def parse_extensions(extensions_str: str) -> list[str]:
    """Parse comma-separated extensions string."""
    if not extensions_str:
        return []
    return [ext.strip().lower().lstrip('.') for ext in extensions_str.split(',')]


def parse_exclude(exclude_str: str) -> list[str]:
    """Parse comma-separated exclude patterns."""
    if not exclude_str:
        return []
    return [pattern.strip() for pattern in exclude_str.split(',')]


async def scan_folder(args):
    """Main scanning logic."""
    from backend.core.database import init_db, get_db
    from backend.core.documents.repository import DocumentRepository
    from backend.core.documents.config import (
        get_host_config,
        create_scan_config,
        HostConfig,
        FolderScanConfig,
    )
    from backend.core.documents.folder_scanner import (
        FolderScanner,
        ScanCache,
        create_scanner,
        FileInfo,
        ScanResult,
    )

    logger = logging.getLogger(__name__)

    # Initialize database
    init_db()

    # Get database session
    db = next(get_db())
    repo = DocumentRepository(db)

    # Get or create host config
    host_config = get_host_config(args.host)
    if not host_config:
        if args.host == "localhost":
            # Use actual hostname instead of "localhost"
            actual_hostname = socket.gethostname()
            host_config = HostConfig(
                name=actual_hostname,
                type="local",
                mount_point="/",
            )
        else:
            logger.error(f"Unknown host: {args.host}")
            logger.error("Configure hosts in config/document_hosts.yaml")
            return 1

    # Build extensions list (use defaults unless --all-extensions specified)
    if args.all_extensions:
        extensions = None  # No filter - all file types
    elif args.extensions:
        extensions = parse_extensions(args.extensions)
    else:
        extensions = DEFAULT_EXTENSIONS

    # Build exclude patterns
    exclude_patterns = parse_exclude(args.exclude) if args.exclude else None

    # Create scan config
    scan_config = FolderScanConfig(
        host=host_config,
        base_path=args.path,
        recursive=args.recursive,
        skip_hidden=not args.include_hidden,
        extensions=extensions,
        exclude_patterns=exclude_patterns,
        max_file_size_mb=args.max_size,
    )

    # Create scanner with cache if specified
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    generate_embeddings = not args.skip_embeddings
    single_embedding = args.single_embedding
    scanner = create_scanner(repo, cache_dir, generate_embeddings=generate_embeddings, single_embedding=single_embedding)

    # Progress callback (verbose per-file logging; summary is handled by scanner)
    def progress_callback(file_info: FileInfo, result: ScanResult):
        if args.verbose:
            logger.debug(f"Processing: {file_info.path}")

    # Run scan
    logger.info(f"Scanning {args.path} on {host_config.name}")
    logger.info(f"Recursive: {args.recursive}, Extensions: {extensions or 'all'}")

    if args.dry_run:
        logger.info("DRY RUN - no changes will be made")

    if args.reindex:
        logger.info("REINDEX MODE - will re-extract and re-embed existing documents")

    extract_text = not getattr(args, 'no_extract', False)
    if extract_text:
        logger.info("Text extraction: ENABLED (use --no-extract to skip)")
    else:
        logger.info("Text extraction: DISABLED (documents queued for later)")

    if not args.skip_embeddings:
        if args.single_embedding:
            logger.info("Embedding generation: ENABLED (single embedding per document)")
        else:
            logger.info("Embedding generation: ENABLED (per-page/per-sheet mode)")
    else:
        logger.info("Embedding generation: DISABLED")

    if args.commit_interval > 0:
        logger.info(f"Batch commits: every {args.commit_interval} files")
    else:
        logger.info("Batch commits: DISABLED (commit only at end)")

    skip_ocr_sidecars = getattr(args, 'skip_ocr_sidecars', False)
    if not skip_ocr_sidecars:
        logger.info("OCR sidecar detection: ENABLED (use --skip-ocr-sidecars to disable)")

    result = await scanner.scan(
        config=scan_config,
        limit=args.limit,
        dry_run=args.dry_run,
        extract_text=extract_text,
        progress_callback=progress_callback if not args.quiet else None,
        reindex=args.reindex,
        commit_interval=args.commit_interval,
        skip_ocr_sidecars=skip_ocr_sidecars,
    )

    # Final commit for any remaining uncommitted files
    if not args.dry_run:
        try:
            db.commit()
            logger.info("Final commit completed")
        except Exception as e:
            logger.error(f"Failed to commit changes: {e}")
            db.rollback()
            return 1

    # Print summary
    print("\n" + "=" * 60)
    print(result.summary())
    print("=" * 60)

    # Print errors if any
    if result.error_details and args.verbose:
        print("\nErrors:")
        for error in result.error_details[:10]:
            print(f"  - {error['path']}: {error['error']}")
        if len(result.error_details) > 10:
            print(f"  ... and {len(result.error_details) - 10} more errors")

    return 0 if result.errors == 0 else 1


async def detect_deleted(args):
    """Detect and mark deleted files."""
    from backend.core.database import get_db
    from backend.core.documents.repository import DocumentRepository
    from backend.core.documents.config import get_host_config, HostConfig, FolderScanConfig
    from backend.core.documents.folder_scanner import create_scanner

    logger = logging.getLogger(__name__)

    db = next(get_db())
    repo = DocumentRepository(db)

    host_config = get_host_config(args.host)
    if not host_config:
        if args.host == "localhost":
            actual_hostname = socket.gethostname()
            host_config = HostConfig(name=actual_hostname, type="local", mount_point="/")
        else:
            logger.error(f"Unknown host: {args.host}")
            return 1

    scan_config = FolderScanConfig(
        host=host_config,
        base_path=args.path,
        recursive=args.recursive,
    )

    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    scanner = create_scanner(repo, cache_dir)

    logger.info(f"Checking for deleted files in {args.path}")
    deleted_count = await scanner.scan_for_deleted(scan_config, repo)

    if deleted_count > 0:
        if not args.dry_run:
            db.commit()
        logger.info(f"Marked {deleted_count} origins as deleted")
    else:
        logger.info("No deleted files detected")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Scan folders to discover and index documents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ~/Documents                          # Scan Documents (default: pdf,docx,xlsx,etc.)
  %(prog)s /mnt/nas/docs --host nas            # Scan NAS mount
  %(prog)s /data --extensions py,js,ts         # Override: only code files
  %(prog)s /data --all-extensions              # Include ALL file types
  %(prog)s /archive --recursive --dry-run      # Preview what would be indexed
  %(prog)s /data --detect-deleted              # Find and mark deleted files
        """
    )

    # Positional arguments
    parser.add_argument(
        "path",
        type=str,
        help="Path to folder to scan",
    )

    # Host configuration
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Host name from config/document_hosts.yaml (default: localhost)",
    )

    # Scan options
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="Scan subdirectories recursively (default: true)",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_false",
        dest="recursive",
        help="Don't scan subdirectories",
    )
    parser.add_argument(
        "--extensions", "-e",
        type=str,
        help=f"Comma-separated list of extensions to include (default: {','.join(DEFAULT_EXTENSIONS)})",
    )
    parser.add_argument(
        "--all-extensions",
        action="store_true",
        help="Include all file types (override default document extensions filter)",
    )
    parser.add_argument(
        "--exclude", "-x",
        type=str,
        help="Comma-separated list of patterns to exclude (e.g., *.tmp,*.bak)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=100,
        help="Maximum file size in MB (default: 100)",
    )
    parser.add_argument(
        "--include-hidden",
        action="store_true",
        help="Include hidden files and directories",
    )

    # Processing options
    parser.add_argument(
        "--limit", "-l",
        type=int,
        help="Maximum number of files to process",
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        help="Preview what would be indexed without making changes",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip text extraction (only register documents, queue for later extraction)",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Skip embedding generation (embeddings are generated by default)",
    )
    parser.add_argument(
        "--single-embedding",
        action="store_true",
        help="Generate one embedding per document instead of per-page/per-sheet (default: per-page)",
    )
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Re-extract text and regenerate embeddings for already indexed documents",
    )
    parser.add_argument(
        "--commit-interval",
        type=int,
        default=50,
        help="Commit to database every N files (default: 50). Set to 0 to commit only at end.",
    )
    parser.add_argument(
        "--skip-ocr-sidecars",
        action="store_true",
        help="Skip loading .ocr.json sidecar files (OCR sidecars are loaded by default for pdf/png/jpg/jpeg)",
    )
    parser.add_argument(
        "--detect-deleted",
        action="store_true",
        help="Detect and mark deleted files (requires previous scan cache)",
    )

    # Cache options
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Directory for scan cache (enables incremental scanning)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable scan cache (always process all files)",
    )

    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output",
    )

    args = parser.parse_args()

    # Validate path
    if not args.detect_deleted:
        path = Path(args.path).expanduser()
        if not path.exists():
            print(f"Error: Path does not exist: {path}", file=sys.stderr)
            sys.exit(1)
        if not path.is_dir():
            print(f"Error: Path is not a directory: {path}", file=sys.stderr)
            sys.exit(1)
        args.path = str(path)

    # Setup logging
    setup_logging(verbose=args.verbose, quiet=args.quiet)

    # Handle no-cache option
    if args.no_cache:
        args.cache_dir = None

    # Run appropriate command
    if args.detect_deleted:
        exit_code = asyncio.run(detect_deleted(args))
    else:
        exit_code = asyncio.run(scan_folder(args))

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
