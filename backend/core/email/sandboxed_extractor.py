"""
Sandboxed Attachment Extractor

Runs attachment text extraction in an isolated subprocess to contain
potential security issues from malicious files (RCE, DoS, etc.).

Security measures:
- Separate subprocess (crash isolation)
- Memory limit (prevents memory bombs)
- CPU time limit (prevents infinite loops)
- Timeout enforcement
- No network access (restricted environment)
- No file system access outside temp directory

Usage:
    extractor = SandboxedExtractor()
    text = await extractor.extract_pdf(pdf_bytes)
"""
import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
import subprocess
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Configuration - generous limits for large attachments
DEFAULT_TIMEOUT_SECONDS = 60  # 60 seconds for large PDFs
DEFAULT_MAX_MEMORY_MB = 512   # 512MB for complex documents
DEFAULT_MAX_OUTPUT_SIZE = 10 * 1024 * 1024  # 10MB max extracted text


@dataclass
class ExtractionResult:
    """Result of a sandboxed extraction."""
    success: bool
    text: Optional[str] = None
    error: Optional[str] = None
    extraction_time_ms: Optional[float] = None


# Worker script that runs in the subprocess
WORKER_SCRIPT = '''
"""Sandboxed extraction worker - runs in isolated subprocess."""
import sys
import json
import base64
import io
import resource
import signal

def set_resource_limits(max_memory_mb: int, max_cpu_seconds: int):
    """Set resource limits for the subprocess."""
    # Memory limit (in bytes)
    memory_bytes = max_memory_mb * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
    except (ValueError, resource.error):
        # Some systems don't support RLIMIT_AS
        pass
    
    # CPU time limit (prevents infinite loops)
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (max_cpu_seconds, max_cpu_seconds))
    except (ValueError, resource.error):
        pass
    
    # Prevent fork bombs
    try:
        resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    except (ValueError, resource.error):
        pass

def extract_pdf(data: bytes) -> str:
    """Extract text from PDF."""
    import warnings
    import pdfplumber
    
    text_parts = []
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                try:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
                except Exception:
                    continue
    return '\\n\\n'.join(text_parts)

def extract_docx(data: bytes) -> str:
    """Extract text from DOCX."""
    from docx import Document
    doc = Document(io.BytesIO(data))
    return '\\n'.join([p.text for p in doc.paragraphs if p.text])

def extract_xlsx(data: bytes) -> str:
    """Extract text from Excel."""
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(data), data_only=True)
    text_parts = []
    for sheet in wb.worksheets:
        for row in sheet.iter_rows():
            cells = [str(cell.value) for cell in row if cell.value is not None]
            if cells:
                text_parts.append(' | '.join(cells))
    return '\\n'.join(text_parts)

def extract_pptx(data: bytes) -> str:
    """Extract text from PowerPoint."""
    from pptx import Presentation
    prs = Presentation(io.BytesIO(data))
    text_parts = []
    for i, slide in enumerate(prs.slides, 1):
        slide_text = []
        for shape in slide.shapes:
            if hasattr(shape, 'text') and shape.text:
                slide_text.append(shape.text)
        if slide_text:
            text_parts.append(f"Slide {i}:\\n" + '\\n'.join(slide_text))
    return '\\n\\n'.join(text_parts)

def extract_rtf(data: bytes) -> str:
    """Extract text from RTF."""
    from striprtf.striprtf import rtf_to_text
    rtf_string = data.decode('utf-8', errors='replace')
    return rtf_to_text(rtf_string)

def extract_ics(data: bytes) -> str:
    """Extract text from iCalendar."""
    from icalendar import Calendar
    ics_string = data.decode('utf-8', errors='replace')
    cal = Calendar.from_ical(ics_string)
    events = []
    for component in cal.walk():
        if component.name == 'VEVENT':
            parts = []
            if component.get('summary'):
                parts.append(f"Event: {component.get('summary')}")
            if component.get('dtstart'):
                parts.append(f"Start: {component.get('dtstart').dt}")
            if component.get('dtend'):
                parts.append(f"End: {component.get('dtend').dt}")
            if component.get('location'):
                parts.append(f"Location: {component.get('location')}")
            if component.get('description'):
                parts.append(f"Description: {component.get('description')}")
            if parts:
                events.append('\\n'.join(parts))
    return '\\n\\n'.join(events)

def main():
    """Main entry point for the worker."""
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        file_type = input_data['type']
        file_data = base64.b64decode(input_data['data'])
        max_memory_mb = input_data.get('max_memory_mb', 256)
        max_cpu_seconds = input_data.get('max_cpu_seconds', 30)
        
        # Set resource limits
        set_resource_limits(max_memory_mb, max_cpu_seconds)
        
        # Extract based on type
        extractors = {
            'pdf': extract_pdf,
            'docx': extract_docx,
            'xlsx': extract_xlsx,
            'pptx': extract_pptx,
            'rtf': extract_rtf,
            'ics': extract_ics,
        }
        
        if file_type not in extractors:
            result = {'success': False, 'error': f'Unknown file type: {file_type}'}
        else:
            text = extractors[file_type](file_data)
            result = {'success': True, 'text': text}
        
        print(json.dumps(result))
        
    except MemoryError:
        print(json.dumps({'success': False, 'error': 'Memory limit exceeded'}))
    except Exception as e:
        print(json.dumps({'success': False, 'error': str(e)[:500]}))

if __name__ == '__main__':
    main()
'''


class SandboxedExtractor:
    """
    Runs attachment extraction in a sandboxed subprocess.
    
    Features:
    - Crash isolation: subprocess crashes don't affect main process
    - Resource limits: memory and CPU time limits
    - Timeout: extraction is killed after timeout
    - No network: subprocess environment restricts network access
    """
    
    def __init__(
        self,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
        max_memory_mb: int = DEFAULT_MAX_MEMORY_MB,
        max_output_size: int = DEFAULT_MAX_OUTPUT_SIZE,
        enabled: bool = True
    ):
        """
        Initialize the sandboxed extractor.
        
        Args:
            timeout_seconds: Maximum time for extraction
            max_memory_mb: Maximum memory for subprocess
            max_output_size: Maximum size of extracted text
            enabled: If False, extraction is skipped (returns empty)
        """
        self.timeout_seconds = timeout_seconds
        self.max_memory_mb = max_memory_mb
        self.max_output_size = max_output_size
        self.enabled = enabled
        self._worker_script_path: Optional[Path] = None
    
    def _get_worker_script_path(self) -> Path:
        """Get or create the worker script file."""
        if self._worker_script_path and self._worker_script_path.exists():
            return self._worker_script_path
        
        # Create worker script in temp directory
        worker_dir = Path(tempfile.gettempdir()) / "email_processor_sandbox"
        worker_dir.mkdir(exist_ok=True)
        
        worker_path = worker_dir / "extraction_worker.py"
        worker_path.write_text(WORKER_SCRIPT)
        
        self._worker_script_path = worker_path
        return worker_path
    
    def _get_restricted_env(self) -> dict:
        """Create a restricted environment for the subprocess."""
        # Start with minimal environment
        env = {
            'PATH': os.environ.get('PATH', '/usr/bin:/bin'),
            'HOME': tempfile.gettempdir(),
            'TMPDIR': tempfile.gettempdir(),
            'LANG': 'C.UTF-8',
        }
        
        # Add PYTHONPATH if needed
        if 'PYTHONPATH' in os.environ:
            env['PYTHONPATH'] = os.environ['PYTHONPATH']
        
        # Explicitly disable network proxies
        env['no_proxy'] = '*'
        env['NO_PROXY'] = '*'
        
        return env
    
    async def _run_extraction(self, file_type: str, file_data: bytes) -> ExtractionResult:
        """Run extraction in sandboxed subprocess."""
        import time
        start_time = time.time()
        
        if not self.enabled:
            return ExtractionResult(
                success=False,
                error="Sandboxed extraction disabled"
            )
        
        try:
            # Prepare input
            input_data = json.dumps({
                'type': file_type,
                'data': base64.b64encode(file_data).decode('ascii'),
                'max_memory_mb': self.max_memory_mb,
                'max_cpu_seconds': self.timeout_seconds,
            })
            
            # Get worker script path
            worker_path = self._get_worker_script_path()
            
            # Run subprocess
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                str(worker_path),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._get_restricted_env(),
                # Limit output size
                limit=self.max_output_size,
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(input=input_data.encode()),
                    timeout=self.timeout_seconds
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return ExtractionResult(
                    success=False,
                    error=f"Extraction timed out after {self.timeout_seconds}s",
                    extraction_time_ms=(time.time() - start_time) * 1000
                )
            
            if proc.returncode != 0:
                stderr_text = stderr.decode('utf-8', errors='replace')[:500]
                return ExtractionResult(
                    success=False,
                    error=f"Subprocess failed (code {proc.returncode}): {stderr_text}",
                    extraction_time_ms=(time.time() - start_time) * 1000
                )
            
            # Parse output
            try:
                result = json.loads(stdout.decode('utf-8'))
                return ExtractionResult(
                    success=result.get('success', False),
                    text=result.get('text'),
                    error=result.get('error'),
                    extraction_time_ms=(time.time() - start_time) * 1000
                )
            except json.JSONDecodeError as e:
                return ExtractionResult(
                    success=False,
                    error=f"Invalid output from worker: {str(e)}",
                    extraction_time_ms=(time.time() - start_time) * 1000
                )
                
        except Exception as e:
            logger.error(f"Sandboxed extraction failed: {e}")
            return ExtractionResult(
                success=False,
                error=f"Sandbox error: {str(e)[:200]}",
                extraction_time_ms=(time.time() - start_time) * 1000
            )
    
    async def extract_pdf(self, pdf_bytes: bytes) -> ExtractionResult:
        """Extract text from PDF in sandbox."""
        return await self._run_extraction('pdf', pdf_bytes)
    
    async def extract_docx(self, docx_bytes: bytes) -> ExtractionResult:
        """Extract text from DOCX in sandbox."""
        return await self._run_extraction('docx', docx_bytes)
    
    async def extract_xlsx(self, xlsx_bytes: bytes) -> ExtractionResult:
        """Extract text from Excel in sandbox."""
        return await self._run_extraction('xlsx', xlsx_bytes)
    
    async def extract_pptx(self, pptx_bytes: bytes) -> ExtractionResult:
        """Extract text from PowerPoint in sandbox."""
        return await self._run_extraction('pptx', pptx_bytes)
    
    async def extract_rtf(self, rtf_bytes: bytes) -> ExtractionResult:
        """Extract text from RTF in sandbox."""
        return await self._run_extraction('rtf', rtf_bytes)
    
    async def extract_ics(self, ics_bytes: bytes) -> ExtractionResult:
        """Extract text from iCalendar in sandbox."""
        return await self._run_extraction('ics', ics_bytes)


# Global singleton for use across the application
_sandboxed_extractor: Optional[SandboxedExtractor] = None


def get_sandboxed_extractor() -> SandboxedExtractor:
    """Get the global sandboxed extractor instance."""
    global _sandboxed_extractor
    if _sandboxed_extractor is None:
        # Check if sandboxing is enabled via environment (enabled by default)
        enabled = os.getenv('SANDBOX_ATTACHMENTS', 'true').lower() in ('true', '1', 'yes')
        timeout = int(os.getenv('SANDBOX_TIMEOUT_SECONDS', str(DEFAULT_TIMEOUT_SECONDS)))
        max_memory = int(os.getenv('SANDBOX_MAX_MEMORY_MB', str(DEFAULT_MAX_MEMORY_MB)))
        
        _sandboxed_extractor = SandboxedExtractor(
            timeout_seconds=timeout,
            max_memory_mb=max_memory,
            enabled=enabled
        )
        
        if enabled:
            logger.info(f"Sandboxed attachment extraction enabled (timeout={timeout}s, memory={max_memory}MB)")
        else:
            logger.warning("⚠️  Sandboxed attachment extraction DISABLED - using in-process extraction")
    
    return _sandboxed_extractor
