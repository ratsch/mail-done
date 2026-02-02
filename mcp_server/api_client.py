"""
API Client for Email Search Backend

Provides HTTP client to call the FastAPI backend instead of direct DB access.

Supports two authentication methods (auto-detected):
1. Ed25519 request signing (if SIGNING_KEY_PATH is set)
2. API key (fallback, if BACKEND_API_KEY is set)
"""
import base64
import hashlib
import json as json_module
import os
import re
import logging
import secrets
import time
import httpx
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib.parse import urljoin, urlencode
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)

# Configuration from environment
API_BASE_URL = os.getenv("EMAIL_API_URL", "http://localhost:8000")
# Note: API_KEY is loaded lazily in __init__ to allow server creation without it
# This enables testing the server creation without backend running

# SSL verification - set to "true" to skip SSL verification for self-signed certs
MCP_SKIP_SSL_VERIFY = os.getenv("MCP_SKIP_SSL_VERIFY", "false").lower() == "true"

# MCP Query Restrictions (for security and scope)
# NOTE: These are READ FRESH when EmailAPIClient is instantiated, not at module import time
# This ensures environment variables set by Cursor MCP are picked up correctly
def _get_mcp_allowed_account():
    return os.getenv("MCP_ALLOWED_ACCOUNT")  # e.g., "work"

def _get_mcp_date_limit_days():
    return int(os.getenv("MCP_DATE_LIMIT_DAYS", "730"))  # HARD LIMIT: 730 days (2 years)

def _get_mcp_default_days():
    return int(os.getenv("MCP_DEFAULT_DAYS", "90"))  # Default: 90 days (3 months)


class EmailAPIClient:
    """
    HTTP client for the Email Search Backend API.
    
    Wraps all API calls to the FastAPI backend service.
    """
    
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        signing_key_path: str = None,
        client_id: str = "laptop-admin",
        timeout: float = 60.0
    ):
        """
        Initialize API client.
        
        Authentication priority:
        1. If signing_key_path is provided (or SIGNING_KEY_PATH env var), use Ed25519 signing
        2. Otherwise, fall back to api_key (or BACKEND_API_KEY env var)
        
        Args:
            base_url: Backend API URL (default: from EMAIL_API_URL env var)
            api_key: API key for authentication (default: from BACKEND_API_KEY env var)
            signing_key_path: Path to Ed25519 private key for signing
            client_id: Client identifier for signed requests (default: laptop-admin)
            timeout: Request timeout in seconds
        """
        self.base_url = (base_url or API_BASE_URL).rstrip('/')
        self.timeout = timeout
        self.client_id = client_id
        
        # Try to set up signing first
        self._private_key = None
        self._use_signing = False
        
        signing_path = signing_key_path or os.getenv("SIGNING_KEY_PATH")
        if signing_path:
            try:
                self._setup_signing(signing_path)
                self._use_signing = True
                logger.info(f"EmailAPIClient: using Ed25519 signing (client_id={client_id})")
            except Exception as e:
                logger.warning(f"Failed to load signing key, falling back to API key: {e}")
        
        # Fall back to API key if signing not available
        if not self._use_signing:
            self.api_key = api_key or os.getenv("BACKEND_API_KEY")
            if not self.api_key:
                raise ValueError(
                    "Either SIGNING_KEY_PATH or BACKEND_API_KEY environment variable is required."
                )
            self.headers = {
                "X-API-Key": self.api_key,
                "Content-Type": "application/json"
            }
            logger.info("EmailAPIClient: using API key authentication")
        else:
            self.api_key = None
            self.headers = {"Content-Type": "application/json"}
        
        # MCP Query Restrictions - read fresh from environment each time
        self.allowed_account = _get_mcp_allowed_account()
        self.date_limit_days = _get_mcp_date_limit_days()  # HARD LIMIT (can never be exceeded)
        self.default_days = _get_mcp_default_days()  # Default lookback when user doesn't specify
        
        # Calculate date cutoffs
        # - max_date_cutoff: HARD LIMIT - no queries before this date (730 days)
        # - default_date_cutoff: Default lookback when user doesn't specify (90 days)
        self.max_date_cutoff = (datetime.now(timezone.utc) - timedelta(days=self.date_limit_days)).isoformat()
        self.default_date_cutoff = (datetime.now(timezone.utc) - timedelta(days=self.default_days)).isoformat()
        
        logger.info(f"EmailAPIClient initialized: {self.base_url}")
        if self.allowed_account:
            logger.info(f"MCP restricted to account: {self.allowed_account}")
        logger.info(f"MCP HARD LIMIT: emails from {self.max_date_cutoff[:10]} onwards ({self.date_limit_days} days)")
        logger.info(f"MCP DEFAULT: emails from {self.default_date_cutoff[:10]} onwards ({self.default_days} days)")
    
    def _setup_signing(self, key_path: str) -> None:
        """Load Ed25519 private key for request signing."""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        
        path = Path(key_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Signing key not found: {path}")
        
        pem_data = path.read_bytes()
        private_key = serialization.load_pem_private_key(pem_data, password=None)
        
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError(f"Not an Ed25519 key: {type(private_key)}")
        
        self._private_key = private_key
    
    def _sign_request(self, method: str, path: str, body: bytes = b"") -> Dict[str, str]:
        """Sign a request and return auth headers."""
        timestamp = int(time.time())
        nonce = secrets.token_hex(16)  # 32 hex characters
        
        # Compute body hash
        body_hash = "empty" if not body else hashlib.sha256(body).hexdigest()
        
        # Create canonical request
        canonical = f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{body_hash}"
        
        # Sign with Ed25519
        signature = self._private_key.sign(canonical.encode("utf-8"))
        signature_b64 = base64.b64encode(signature).decode("ascii")
        
        return {
            "X-Client-Id": self.client_id,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
            "X-Signature": signature_b64,
        }
    
    def _apply_mcp_restrictions(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply MCP query restrictions to search parameters.
        
        Enforces:
        - Date restriction (HARD LIMIT: max 730 days, DEFAULT: 90 days)
        - Account restriction (only emails in allowed account)
        
        Args:
            params: Original search parameters
            
        Returns:
            Modified parameters with restrictions applied
        """
        # Apply date restriction
        if 'date_from' not in params or not params['date_from']:
            # No date specified - use DEFAULT (90 days)
            params['date_from'] = self.default_date_cutoff
            logger.debug(f"No date specified, using DEFAULT: {self.default_date_cutoff[:10]} ({self.default_days} days)")
        else:
            # User specified a date - enforce HARD LIMIT (730 days max)
            user_date = params['date_from']
            if user_date < self.max_date_cutoff:
                logger.warning(f"HARD LIMIT enforced: {user_date[:10]} -> {self.max_date_cutoff[:10]} (max {self.date_limit_days} days)")
                params['date_from'] = self.max_date_cutoff
            else:
                logger.debug(f"User date accepted: {user_date[:10]} (within HARD LIMIT)")
        
        # Apply account restriction (if configured)
        if self.allowed_account:
            # Check if it's a single account or comma-separated list
            accounts = [a.strip() for a in self.allowed_account.split(',') if a.strip()]
            
            if len(accounts) == 1:
                # Single account - filter to just that account
                params['account_filter'] = accounts[0]
                logger.debug(f"Account restriction applied: {accounts[0]}")
            else:
                # Multiple accounts allowed - don't filter, search all allowed accounts
                # The MCP server will validate results against the allowed list
                logger.debug(f"Multiple accounts allowed: {accounts}, no account_filter applied")
        
        return params
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        json_data: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Make HTTP request to backend API."""
        url = urljoin(self.base_url + "/", endpoint.lstrip('/'))
        
        # Build path for signing (includes query string)
        signing_path = "/" + endpoint.lstrip('/')
        if params:
            query = urlencode(params)
            signing_path = f"{signing_path}?{query}"
        
        # Prepare body
        body = b""
        if json_data is not None:
            body = json_module.dumps(json_data).encode("utf-8")
        
        # Build headers
        headers = dict(self.headers)
        if self._use_signing:
            sign_headers = self._sign_request(method, signing_path, body)
            headers.update(sign_headers)
        
        async with httpx.AsyncClient(timeout=self.timeout, verify=not MCP_SKIP_SSL_VERIFY) as client:
            try:
                response = await client.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    content=body if body else None,
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPStatusError as e:
                # 404 is expected in some cases (e.g., sender stats for domains)
                if e.response.status_code == 404:
                    logger.debug(f"API 404: {endpoint} - {e.response.text}")
                else:
                    logger.error(f"API error {e.response.status_code}: {e.response.text}")
                return {"error": f"API error: {e.response.status_code}", "details": e.response.text}
            except httpx.RequestError as e:
                logger.error(f"Request error: {e}")
                return {"error": f"Request failed: {str(e)}"}
    
    async def semantic_search(
        self,
        query: str,
        mode: str = "semantic",
        top_k: int = 10,
        similarity_threshold: float = 0.6,
        category: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        vip_only: bool = False,
        needs_reply: Optional[bool] = None,
        sender: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search emails semantically via backend API.

        NOTE: MCP restrictions automatically applied:
        - Date filtered to last {MCP_DATE_LIMIT_DAYS} days
        - Account filtered to {MCP_ALLOWED_ACCOUNT} (if configured)

        Calls: GET /api/search
        """
        params = {
            "q": query,
            "mode": mode,
            "page_size": top_k,
            "similarity_threshold": similarity_threshold,
            "date_from": date_from,
            "date_to": date_to
        }

        # Apply MCP restrictions (date and account filters)
        params = self._apply_mcp_restrictions(params)

        if category:
            params["category"] = category
        if vip_only:
            params["vip_only"] = "true"
        if needs_reply is not None:
            params["needs_reply"] = str(needs_reply).lower()
        if sender:
            params["sender"] = sender
        
        result = await self._request("GET", "/api/search", params=params)
        
        # Transform to simpler format for MCP
        if "error" not in result:
            return {
                "query": query,
                "mode": mode,
                "total": result.get("total", 0),
                "results": [
                    self._transform_email_result(r)
                    for r in result.get("results", [])
                ],
                "search_time_ms": None  # API doesn't return this
            }
        return result
    
    async def search_by_sender(
        self,
        sender: str,
        top_k: int = 20,
        category: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search emails by sender via backend API.
        
        NOTE: MCP restrictions automatically applied:
        - Date filtered to last {MCP_DATE_LIMIT_DAYS} days
        - Account filtered to {MCP_ALLOWED_ACCOUNT} (if configured)
        
        Uses the stats endpoint to get sender info, then searches.
        """
        # Search with sender filter
        # Use keyword mode (fast!) with wildcard query since the sender filter does the real work
        # Backend has max page_size of 100
        params = {
            "q": "*",  # Wildcard query - we're filtering by sender, not by keywords
            "mode": "keyword",  # Keyword mode is MUCH faster (no embedding generation!)
            "page_size": min(top_k, 100),  # Respect API limit
            "sender": sender,  # Backend will filter by from_address at DB level (fast!)
            "date_from": date_from,
            "date_to": date_to
        }
        
        # Apply MCP restrictions (date and account filters)
        params = self._apply_mcp_restrictions(params)
        
        if category:
            params["category"] = category
        
        result = await self._request("GET", "/api/search", params=params)
        
        # Try to get sender stats (may not exist for domain-based searches)
        sender_info = await self._request("GET", f"/api/stats/senders/{sender}")
        # Note: 404 is expected when searching by domain rather than specific sender
        
        if "error" not in result:
            return {
                "query": sender,
                "mode": "sender",
                "total": result.get("total", 0),
                "results": [
                    self._transform_email_result(r)
                    for r in result.get("results", [])
                ],
                "sender_info": sender_info if "error" not in sender_info else None
            }
        return result
    
    async def search_by_topic(
        self,
        topic: str,
        categories: Optional[List[str]] = None,
        top_k: int = 20,
        similarity_threshold: float = 0.6
    ) -> Dict[str, Any]:
        """
        Search emails by research topic via backend API.
        
        Calls: GET /api/search/topics
        
        Note: This uses semantic search on topics, so date filtering is applied
        via MCP restrictions to limit the search scope.
        """
        params = {
            "topic": topic,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold
        }
        
        # Apply MCP date and account restrictions
        params = self._apply_mcp_restrictions(params)
        
        if categories:
            params["categories"] = ",".join(categories)
        
        result = await self._request("GET", "/api/search/topics", params=params)
        
        if "error" not in result:
            return {
                "query": topic,
                "mode": "topic",
                "categories": categories,
                "total": result.get("total", 0),
                "results": [
                    self._transform_email_result(r)
                    for r in result.get("results", [])
                ]
            }
        return result
    
    async def find_similar_emails(
        self,
        email_id: str,
        top_k: int = 10,
        same_category_only: bool = False,
        exclude_same_sender: bool = False
    ) -> Dict[str, Any]:
        """
        Find emails similar to a reference email via backend API.
        
        Calls: GET /api/search/similar/{email_id}
        """
        params = {
            "top_k": top_k,
            "same_category_only": str(same_category_only).lower(),
            "exclude_same_sender": str(exclude_same_sender).lower()
        }
        
        result = await self._request("GET", f"/api/search/similar/{email_id}", params=params)
        
        if "error" not in result:
            return {
                "reference_email_id": email_id,
                "mode": "similar",
                "total": result.get("total", 0),
                "results": [
                    self._transform_email_result(r)
                    for r in result.get("results", [])
                ]
            }
        return result
    
    async def get_email_details(self, email_id: str) -> Dict[str, Any]:
        """
        Get full email details via backend API.
        
        Calls: GET /api/emails/{email_id}
        """
        result = await self._request("GET", f"/api/emails/{email_id}")
        
        if "error" not in result:
            return self._transform_email_detail(result)
        return result
    
    async def list_categories(self) -> Dict[str, Any]:
        """
        List email categories via backend API.
        
        Calls: GET /api/stats/categories/breakdown
        """
        result = await self._request("GET", "/api/stats/categories/breakdown")
        
        if "error" not in result and "categories" in result:
            # Backend returns categories as a dict {category_name: count}
            cats_dict = result.get("categories", {})
            categories = [
                {
                    "name": cat_name,
                    "count": count,
                    "description": self._get_category_description(cat_name)
                }
                for cat_name, count in cats_dict.items()
            ]
            return {
                "total_categories": len(categories),
                "categories": categories
            }
        return result
    
    async def list_top_senders(self, top_k: int = 20) -> Dict[str, Any]:
        """
        List top email senders via backend API.
        
        Calls: GET /api/stats/senders
        """
        result = await self._request("GET", "/api/stats/senders", params={"limit": top_k})
        
        if "error" not in result:
            return {
                "total": len(result.get("senders", [])),
                "senders": result.get("senders", [])
            }
        return result
    
    def _transform_email_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Transform API email result to MCP format."""
        email = result.get("email", result)
        return {
            "id": email.get("id"),
            "message_id": email.get("message_id"),
            "subject": email.get("subject", "(No subject)"),
            "from_address": email.get("from_address"),
            "from_name": email.get("from_name"),
            "date": email.get("date"),
            "category": email.get("category") or email.get("ai_category"),
            "subcategory": email.get("subcategory") or email.get("ai_subcategory"),
            "summary": email.get("summary") or email.get("ai_summary"),
            "similarity_score": result.get("score"),
            "is_vip": bool(email.get("vip_level")),
            "needs_reply": email.get("needs_reply", False)
        }
    
    def _transform_email_detail(self, email: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform API email detail to MCP format.
        
        SECURITY: Excludes attachment content/paths for safety.
        """
        return {
            "id": email.get("id"),
            "message_id": email.get("message_id"),
            "subject": email.get("subject", "(No subject)"),
            "from_address": email.get("from_address"),
            "from_name": email.get("from_name"),
            "date": email.get("date"),
            "body": email.get("body_text") or email.get("body_markdown") or "",
            "to_addresses": email.get("to_addresses", []),
            "cc_addresses": email.get("cc_addresses", []),
            # SECURITY: Only expose attachment metadata, not content
            "has_attachments": email.get("has_attachments", False),
            "attachment_count": email.get("attachment_count", 0),
            "attachment_names": [
                att.get("filename", "unknown") 
                for att in email.get("attachment_info", [])
            ] if email.get("attachment_info") else [],
            # No attachment content or paths exposed to MCP
            "category": email.get("category") or email.get("ai_category"),
            "subcategory": email.get("subcategory") or email.get("ai_subcategory"),
            "summary": email.get("summary") or email.get("ai_summary"),
            "urgency": email.get("ai_urgency"),
            "urgency_score": email.get("ai_urgency_score"),
            "action_items": email.get("ai_action_items"),
            "applicant_name": email.get("applicant_name"),
            "research_fit_score": email.get("research_fit_score"),
            "recommendation_score": email.get("overall_recommendation_score"),
            "is_vip": bool(email.get("vip_level")),
            "needs_reply": email.get("needs_reply", False),
            # Account info for cache organization
            "account_id": email.get("account_id"),
        }
    
    def _get_category_description(self, category: str) -> str:
        """Get human-readable description for a category."""
        descriptions = {
            "application-phd": "PhD application emails",
            "application-postdoc": "Postdoc application emails",
            "application-visiting": "Visiting researcher applications",
            "application-internship": "Internship applications",
            "invitation-speaking": "Speaking invitations (conferences, seminars)",
            "invitation-review": "Paper/grant review requests",
            "invitation-committee": "Committee participation requests",
            "collaboration-research": "Research collaboration proposals",
            "collaboration-industry": "Industry partnership inquiries",
            "admin-hr": "HR and administrative matters",
            "admin-finance": "Financial and budget matters",
            "admin-it": "IT and technical support",
            "teaching-course": "Course-related emails",
            "teaching-student": "Student inquiries",
            "newsletter": "Newsletters and announcements",
            "spam": "Spam or unwanted emails",
            "personal": "Personal correspondence",
            "other": "Uncategorized emails"
        }
        return descriptions.get(category, f"Emails in category: {category}")
    
    def _sanitize_filename(self, filename: str, max_length: int = 100) -> str:
        """
        Sanitize filename for safe filesystem storage.
        
        - Replace invalid chars with underscore
        - Truncate to max_length (preserving extension)
        - Handle edge cases (empty, dots only, etc.)
        
        Args:
            filename: Original filename
            max_length: Maximum length (default: 100)
            
        Returns:
            Sanitized filename safe for filesystem
        """
        # Replace invalid characters (Windows + Unix)
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', filename)
        
        # Handle edge cases
        if not sanitized or sanitized.strip('.') == '':
            sanitized = 'attachment'
        
        # Truncate while preserving extension
        if len(sanitized) > max_length:
            name, ext = os.path.splitext(sanitized)
            max_name_len = max_length - len(ext)
            sanitized = name[:max_name_len] + ext
        
        return sanitized
    
    def _extract_filename_from_content_disposition(self, content_disposition: str) -> Optional[str]:
        """
        Extract filename from Content-Disposition header.
        
        Handles:
        - Simple format: attachment; filename="example.pdf"
        - RFC 5987 format: attachment; filename*=UTF-8''example%20file.pdf
        - Mixed format with both filename and filename*
        
        Args:
            content_disposition: Content-Disposition header value
            
        Returns:
            Extracted filename or None if not found
        """
        if not content_disposition:
            return None
        
        # Try RFC 5987 filename* first (takes precedence)
        # Format: filename*=charset'language'encoded_value
        match = re.search(r"filename\*\s*=\s*([^']*)'([^']*)'(.+?)(?:;|$)", content_disposition)
        if match:
            charset, language, encoded_value = match.groups()
            charset = charset or 'utf-8'
            try:
                from urllib.parse import unquote
                return unquote(encoded_value.strip(), encoding=charset)
            except Exception:
                pass
        
        # Try regular filename parameter
        # Format: filename="value" or filename=value
        match = re.search(r'filename\s*=\s*"([^"]+)"', content_disposition)
        if match:
            return match.group(1)
        
        # Try without quotes
        match = re.search(r'filename\s*=\s*([^;\s]+)', content_disposition)
        if match:
            return match.group(1)
        
        return None
    
    async def _apply_cache_ttl(self, cache_dir: Path) -> None:
        """Delete cached files older than TTL (if configured)"""
        ttl_days_str = os.getenv("MCP_ATTACHMENT_CACHE_TTL_DAYS")
        if not ttl_days_str:
            return
        
        try:
            ttl_days = int(ttl_days_str)
            cutoff = datetime.now() - timedelta(days=ttl_days)
            
            for file in cache_dir.iterdir():
                if file.is_file():
                    mtime = datetime.fromtimestamp(file.stat().st_mtime)
                    if mtime < cutoff:
                        logger.info(f"TTL cleanup: deleting {file.name} (older than {ttl_days} days)")
                        file.unlink()
        except ValueError:
            logger.warning(f"Invalid MCP_ATTACHMENT_CACHE_TTL_DAYS: {ttl_days_str}")
        except Exception as e:
            logger.warning(f"Error during TTL cleanup: {e}")
    
    async def list_attachments(self, email_id: str) -> Dict[str, Any]:
        """
        List attachments for an email.
        
        Args:
            email_id: UUID of the email
            
        Returns:
            Dict with "attachments" list or "error" key
        """
        result = await self._request("GET", f"/api/emails/{email_id}/attachments")
        
        # API returns a list, but we need to handle errors
        if isinstance(result, list):
            return {"attachments": result}
        elif isinstance(result, dict) and "error" in result:
            return result
        else:
            # Unexpected format - wrap in dict
            return {"attachments": result if isinstance(result, list) else []}
    
    async def download_attachment(
        self, 
        email_id: str, 
        attachment_index: int
    ) -> Dict[str, Any]:
        """
        Download attachment to cache directory with caching and TTL support.
        
        Args:
            email_id: UUID of the email
            attachment_index: 0-based index of attachment
            
        Returns:
            {
                "success": true,
                "cached_path": "~/Downloads/.mcp_attachments/work/abc123_0_CV.pdf",
                "original_filename": "CV_John_Doe.pdf",
                "size_bytes": 1048576,
                "content_type": "application/pdf",
                "from_cache": false
            }
        """
        # Get and validate cache directory
        cache_dir_str = os.getenv("MCP_FILE_CACHE_DIR")
        if not cache_dir_str:
            return {"error": "MCP_FILE_CACHE_DIR not configured"}
        
        cache_dir = Path(cache_dir_str).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply TTL cleanup (if configured) - but only run periodically to avoid overhead
        # Check if we should run cleanup (every 10 downloads or if cache is large)
        cache_cleanup_counter = getattr(self, '_cache_cleanup_counter', 0)
        cache_cleanup_counter += 1
        
        # Run cleanup every 10 downloads or if cache dir has > 100 files
        should_cleanup = False
        if cache_cleanup_counter >= 10:
            should_cleanup = True
            self._cache_cleanup_counter = 0
        elif cache_dir.exists():
            file_count = sum(1 for _ in cache_dir.iterdir() if _.is_file())
            if file_count > 100:
                should_cleanup = True
        
        if should_cleanup:
            await self._apply_cache_ttl(cache_dir)
        else:
            self._cache_cleanup_counter = cache_cleanup_counter
        
        # Get email details to determine account for subdirectory
        email_result = await self.get_email_details(email_id)
        if "error" in email_result:
            return email_result

        # Get account for subdirectory (default to "unknown" if not available)
        account_id = email_result.get("account_id") or "unknown"
        account_subdir = self._sanitize_filename(account_id, max_length=50)

        # Get attachment metadata
        attachments_result = await self.list_attachments(email_id)
        if "error" in attachments_result:
            return attachments_result

        # Extract list from result dict
        attachments_list = attachments_result.get("attachments", [])
        if attachment_index >= len(attachments_list):
            return {
                "error": f"Attachment index {attachment_index} out of range (max: {len(attachments_list)-1})"
            }

        attachment_meta = attachments_list[attachment_index]
        original_filename = attachment_meta['filename']
        expected_size = attachment_meta.get('size', 0)

        # Generate cache path: {cache_dir}/attachments/{account}/{email_id}_{index}_{filename}
        safe_filename = self._sanitize_filename(original_filename)
        cache_filename = f"{email_id}_{attachment_index}_{safe_filename}"
        account_cache_dir = cache_dir / "attachments" / account_subdir
        account_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = account_cache_dir / cache_filename
        
        # Check if already cached (verify size matches)
        if cache_path.exists():
            cached_size = cache_path.stat().st_size
            if cached_size == expected_size or expected_size == 0:
                logger.info(f"Attachment found in cache: {cache_path}")
                return {
                    "success": True,
                    "cached_path": str(cache_path),
                    "original_filename": original_filename,
                    "size_bytes": cached_size,
                    "content_type": attachment_meta.get('content_type'),
                    "from_cache": True
                }
            else:
                logger.warning(
                    f"Cache size mismatch ({cached_size} vs {expected_size}), re-downloading"
                )
                cache_path.unlink()
        
        # Download from backend with extended timeout for large files
        url = f"/api/emails/{email_id}/attachments/{attachment_index}/download"
        download_timeout = int(os.getenv("MCP_ATTACHMENT_DOWNLOAD_TIMEOUT", "120"))
        
        # Track content type and filename from response for return value
        content_type = None
        server_filename = None  # Filename from Content-Disposition header (properly decoded)
        
        try:
            # Use streaming for large files to avoid memory issues
            # For files > 10MB, stream; otherwise read all at once
            use_streaming = expected_size > 10 * 1024 * 1024 if expected_size > 0 else False
            
            async with httpx.AsyncClient(timeout=download_timeout, verify=not MCP_SKIP_SSL_VERIFY) as http_client:
                full_url = urljoin(self.base_url + "/", url.lstrip('/'))
                
                # Build headers with signing if available
                headers = dict(self.headers)
                if self._use_signing:
                    sign_headers = self._sign_request("GET", url, b"")
                    headers.update(sign_headers)
                
                if use_streaming:
                    # Streaming download for large files (memory-efficient)
                    async with http_client.stream("GET", full_url, headers=headers) as response:
                        # Check status before streaming
                        if response.status_code == 403:
                            return {"error": "Attachment downloads not enabled on backend. Set ENABLE_ATTACHMENT_API=true"}
                        elif response.status_code == 404:
                            return {"error": f"Email or attachment not found"}
                        elif response.status_code == 410:
                            return {"error": "Email no longer available on IMAP server"}
                        elif response.status_code == 413:
                            return {"error": f"Attachment exceeds size limit ({os.getenv('ATTACHMENT_MAX_SIZE_MB', '50')}MB)"}
                        elif response.status_code == 503:
                            return {"error": "IMAP server unavailable. Please try again later."}
                        
                        response.raise_for_status()
                        content_type = response.headers.get('Content-Type')
                        server_filename = self._extract_filename_from_content_disposition(
                            response.headers.get('Content-Disposition', '')
                        )
                        
                        # Stream to file
                        with open(cache_path, 'wb') as f:
                            async for chunk in response.aiter_bytes(chunk_size=8192):
                                f.write(chunk)
                else:
                    # Non-streaming download for small files (simpler, < 10MB)
                    response = await http_client.get(full_url, headers=headers)
                    
                    # Handle HTTP errors with user-friendly messages
                    if response.status_code == 403:
                        return {"error": "Attachment downloads not enabled on backend. Set ENABLE_ATTACHMENT_API=true"}
                    elif response.status_code == 404:
                        return {"error": f"Email or attachment not found"}
                    elif response.status_code == 410:
                        return {"error": "Email no longer available on IMAP server"}
                    elif response.status_code == 413:
                        return {"error": f"Attachment exceeds size limit ({os.getenv('ATTACHMENT_MAX_SIZE_MB', '50')}MB)"}
                    elif response.status_code == 503:
                        return {"error": "IMAP server unavailable. Please try again later."}
                    
                    response.raise_for_status()
                    content_type = response.headers.get('Content-Type')
                    server_filename = self._extract_filename_from_content_disposition(
                        response.headers.get('Content-Disposition', '')
                    )
                    
                    # Write file directly
                    with open(cache_path, 'wb') as f:
                        f.write(response.content)
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading attachment: {e.response.status_code} - {e.response.text}")
            return {"error": f"Backend error: {e.response.status_code}"}
        except httpx.RequestError as e:
            logger.error(f"Request error downloading attachment: {e}")
            return {"error": f"Failed to connect to backend: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error downloading attachment: {e}", exc_info=True)
            return {"error": f"Download failed: {str(e)}"}
        
        # Use server filename (properly decoded) if available, otherwise fall back to DB metadata
        final_filename = server_filename if server_filename else original_filename
        
        return {
            "success": True,
            "cached_path": str(cache_path),
            "original_filename": final_filename,
            "size_bytes": cache_path.stat().st_size,
            "content_type": content_type,
            "from_cache": False
        }
    
    async def search_documents(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
        document_type: Optional[str] = None,
        mime_type: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Search indexed documents semantically.

        Calls: GET /api/search/unified?types=document

        Returns documents with full origin information (locations, hosts, paths).
        """
        params = {
            "q": query,
            "types": "document",
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }

        if document_type:
            params["document_type"] = document_type
        if mime_type:
            params["document_mime_type"] = mime_type
        if date_from:
            params["date_from"] = date_from
        if date_to:
            params["date_to"] = date_to

        result = await self._request("GET", "/api/search/unified", params=params)

        if "error" not in result:
            return {
                "query": query,
                "mode": "document_semantic",
                "total": result.get("total", 0),
                "results": [
                    self._transform_document_result(r)
                    for r in result.get("results", [])
                    if r.get("result_type") == "document"
                ]
            }
        return result

    async def search_unified(
        self,
        query: str,
        types: str = "all",
        top_k: int = 10,
        similarity_threshold: float = 0.5,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Unified search across emails AND documents.

        Calls: GET /api/search/unified

        Returns mixed results from both corpora, sorted by similarity.
        """
        params = {
            "q": query,
            "types": types,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }

        # Apply MCP date restrictions for email results
        if types in ["all", "email"]:
            params = self._apply_mcp_restrictions(params)
        else:
            # Document-only search - still apply date filters if provided
            if date_from:
                params["date_from"] = date_from
            if date_to:
                params["date_to"] = date_to

        result = await self._request("GET", "/api/search/unified", params=params)

        if "error" not in result:
            transformed_results = []
            for r in result.get("results", []):
                if r.get("result_type") == "email":
                    transformed_results.append({
                        "result_type": "email",
                        "similarity": r.get("similarity"),
                        **self._transform_email_result({"email": r.get("email")})
                    })
                elif r.get("result_type") == "document":
                    transformed_results.append({
                        "result_type": "document",
                        "similarity": r.get("similarity"),
                        **self._transform_document_result(r)
                    })

            return {
                "query": query,
                "types": types,
                "total": result.get("total", 0),
                "results": transformed_results
            }
        return result

    async def get_document_details(self, document_id: str) -> Dict[str, Any]:
        """
        Get full document details with all origins.

        Calls: GET /api/documents/{document_id}

        Returns complete document info including all file locations.
        """
        result = await self._request("GET", f"/api/documents/{document_id}")

        if "error" not in result:
            return self._transform_document_detail(result)
        return result

    async def find_similar_documents(
        self,
        document_id: str,
        top_k: int = 10,
        same_type_only: bool = False,
        similarity_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Find documents similar to a reference document.

        Calls: GET /api/documents/{document_id}/similar

        Returns list of similar documents with similarity scores.
        """
        params = {
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
        }
        if same_type_only:
            params["same_type_only"] = "true"

        result = await self._request(
            "GET",
            f"/api/documents/{document_id}/similar",
            params=params
        )

        if "error" not in result and "results" in result:
            transformed_results = [
                self._transform_document_result(r) for r in result["results"]
            ]
            return {
                "reference_document_id": document_id,
                "total": len(transformed_results),
                "results": transformed_results
            }
        return result

    async def search_document_by_name(
        self,
        name: str,
        top_k: int = 20,
        mime_type: Optional[str] = None,
        host: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Search documents by filename.

        Calls: GET /api/documents/search/by-name

        Returns documents matching the filename pattern.
        """
        params = {
            "name": name,
            "limit": top_k,
        }
        if mime_type:
            params["mime_type"] = mime_type
        if host:
            params["host"] = host

        result = await self._request("GET", "/api/documents/search/by-name", params=params)

        if "error" not in result and "results" in result:
            transformed_results = [
                self._transform_document_result({"document": r}) for r in result["results"]
            ]
            return {
                "query": name,
                "total": len(transformed_results),
                "results": transformed_results
            }
        return result

    async def get_document_index_stats(
        self,
        host: Optional[str] = None,
        path_prefix: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get document indexing statistics.

        Calls: GET /api/documents/status

        Returns aggregate statistics about indexed documents.
        """
        params = {}
        if host:
            params["host"] = host
        if path_prefix:
            params["path_prefix"] = path_prefix

        return await self._request("GET", "/api/documents/status", params=params)

    async def list_indexed_folder(
        self,
        host: str,
        folder_path: str,
        include_subfolders: bool = False,
    ) -> Dict[str, Any]:
        """
        List contents of an indexed folder on a remote host.

        Calls: GET /api/documents/folders/list

        Returns directory listing with indexing status for each file.
        """
        params = {
            "host": host,
            "folder_path": folder_path,
            "include_subfolders": str(include_subfolders).lower(),
        }

        return await self._request("GET", "/api/documents/folders/list", params=params)

    def _transform_document_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Transform API document search result to MCP format with origins."""
        doc = result.get("document", {})
        origins = doc.get("origins", [])

        # Build locations list with full path info
        locations = []
        for origin in origins:
            loc = {
                "host": origin.get("origin_host") or "unknown",
                "path": origin.get("origin_path") or "",
                "filename": origin.get("origin_filename") or doc.get("original_filename") or "unknown",
                "type": origin.get("origin_type") or "unknown",
                "is_primary": origin.get("is_primary", False),
                "file_modified_at": origin.get("file_modified_at"),  # File's mtime
            }
            # Add email info for attachments
            if origin.get("email_id"):
                loc["email_id"] = origin.get("email_id")
                loc["attachment_index"] = origin.get("attachment_index")
            locations.append(loc)

        return {
            "id": doc.get("id"),
            "filename": doc.get("original_filename") or "Untitled",
            "mime_type": doc.get("mime_type"),
            "file_size": doc.get("file_size"),
            "page_count": doc.get("page_count"),
            "document_type": doc.get("document_type"),
            # Dates
            "document_date": doc.get("document_date"),
            "first_seen_at": doc.get("first_seen_at"),
            "last_seen_at": doc.get("last_seen_at"),
            # Content info
            "title": doc.get("title"),
            "summary": doc.get("summary"),
            "ai_category": doc.get("ai_category"),
            "extraction_quality": doc.get("extraction_quality"),
            # Locations (origins)
            "locations": locations,
            "location_count": len(locations),
            # Checksum for deduplication reference
            "checksum": doc.get("checksum"),
            # Similarity from search
            "similarity_score": result.get("similarity"),
        }

    def _transform_document_detail(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Transform API document detail to MCP format with full origin info."""
        doc = result.get("document", {})
        origins = result.get("origins", [])

        # Build locations list with full path info
        locations = []
        for origin in origins:
            loc = {
                "host": origin.get("origin_host") or "unknown",
                "path": origin.get("origin_path") or "",
                "filename": origin.get("origin_filename") or doc.get("original_filename") or "unknown",
                "type": origin.get("origin_type") or "unknown",
                "is_primary": origin.get("is_primary", False),
                "file_modified_at": origin.get("file_modified_at"),  # File's mtime (last edited)
                "discovered_at": origin.get("discovered_at"),
                "last_verified_at": origin.get("last_verified_at"),
                "is_deleted": origin.get("is_deleted", False),
            }
            # Add email info for attachments
            if origin.get("email_id"):
                loc["email_id"] = str(origin.get("email_id"))
                loc["attachment_index"] = origin.get("attachment_index")
            locations.append(loc)

        return {
            "id": str(doc.get("id")),
            "filename": doc.get("original_filename") or "Untitled",
            "mime_type": doc.get("mime_type"),
            "file_size": doc.get("file_size"),
            "page_count": doc.get("page_count"),
            "document_type": doc.get("document_type"),
            # Dates
            "document_date": doc.get("document_date"),
            "first_seen_at": doc.get("first_seen_at"),
            "last_seen_at": doc.get("last_seen_at"),
            "created_at": doc.get("created_at"),
            # Content info
            "title": doc.get("title"),
            "summary": doc.get("summary"),
            "ai_category": doc.get("ai_category"),
            "ai_tags": doc.get("ai_tags"),
            "extraction_status": doc.get("extraction_status"),
            "extraction_quality": doc.get("extraction_quality"),
            "extraction_method": doc.get("extraction_method"),
            # Locations (origins)
            "locations": locations,
            "location_count": len(locations),
            # Checksum for deduplication reference
            "checksum": doc.get("checksum"),
        }

    async def download_document(
        self,
        document_id: str,
        origin_index: int = 0,
        fallback: bool = True
    ) -> Dict[str, Any]:
        """
        Download document file to cache directory.

        Retrieves from origin (filesystem, SSH, or email attachment) and caches locally.

        Args:
            document_id: UUID of the document
            origin_index: Which origin to try first (0=primary)
            fallback: Try other origins if first fails

        Returns:
            {
                "success": true,
                "cached_path": "/path/to/cache/doc_abc123_report.pdf",
                "original_filename": "report.pdf",
                "size_bytes": 1048576,
                "content_type": "application/pdf",
                "from_cache": false,
                "origin_used": "folder:laptop:/Users/user/Documents/report.pdf"
            }
        """
        # Get and validate cache directory
        cache_dir_str = os.getenv("MCP_FILE_CACHE_DIR")
        if not cache_dir_str:
            return {"error": "MCP_FILE_CACHE_DIR not configured"}

        cache_dir = Path(cache_dir_str).expanduser()
        cache_dir.mkdir(parents=True, exist_ok=True)

        # Get document details first to know filename and expected size
        doc_result = await self.get_document_details(document_id)
        if "error" in doc_result:
            return doc_result

        original_filename = doc_result.get("filename") or "document"
        expected_size = doc_result.get("file_size", 0)
        checksum = doc_result.get("checksum", "")[:8]  # First 8 chars for cache key
        locations = doc_result.get("locations", [])

        # Get hostname for subdirectory (from primary origin)
        hostname = "unknown"
        if locations:
            hostname = locations[0].get("host") or "unknown"
        hostname_subdir = self._sanitize_filename(hostname, max_length=50)

        # Generate cache path: {cache_dir}/documents/{hostname}/{doc_id}_{checksum}_{filename}
        safe_filename = self._sanitize_filename(original_filename)
        cache_filename = f"{document_id[:8]}_{checksum}_{safe_filename}"
        host_cache_dir = cache_dir / "documents" / hostname_subdir
        host_cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = host_cache_dir / cache_filename

        # Check if already cached (verify size matches)
        if cache_path.exists():
            cached_size = cache_path.stat().st_size
            if cached_size == expected_size or expected_size == 0:
                logger.info(f"Document found in cache: {cache_path}")
                # Determine which origin would have been used
                origin_info = "cached"
                if locations:
                    loc = locations[0]
                    origin_info = f"{loc.get('type', 'unknown')}:{loc.get('host', 'unknown')}:{loc.get('path', '')}"

                return {
                    "success": True,
                    "cached_path": str(cache_path),
                    "original_filename": original_filename,
                    "size_bytes": cached_size,
                    "content_type": doc_result.get("mime_type"),
                    "from_cache": True,
                    "origin_used": origin_info,
                }
            else:
                logger.warning(
                    f"Cache size mismatch ({cached_size} vs {expected_size}), re-downloading"
                )
                cache_path.unlink()

        # Try local file access first if origin is on this machine
        import socket
        local_hostname = socket.gethostname()
        # Also check without .local suffix for macOS
        local_hostname_short = local_hostname.replace(".local", "")
        local_file_path = None
        origin_info = "unknown"

        for loc in locations:
            loc_host = loc.get("host", "")
            loc_path = loc.get("path", "")
            loc_filename = loc.get("filename", "")

            # Check if this origin is on the local machine
            # Handle hostname variations (with/without .local suffix)
            is_local = (
                loc_host in (local_hostname, local_hostname_short, "localhost", "127.0.0.1") or
                loc_host.replace(".local", "") == local_hostname_short
            )
            if is_local:
                # Build full path - path may already include filename
                if loc_path:
                    full_path = Path(loc_path)
                    # If path doesn't exist but is a directory path + filename, try that
                    if not full_path.exists() and loc_filename:
                        alt_path = full_path / loc_filename
                        if alt_path.exists():
                            full_path = alt_path
                else:
                    continue

                if full_path.exists() and full_path.is_file():
                    local_file_path = full_path
                    origin_info = f"{loc.get('type', 'folder')}:{loc_host}:{full_path}"
                    logger.info(f"Found local file: {full_path}")
                    break

        content_type = doc_result.get("mime_type")

        if local_file_path:
            # Read file directly from local filesystem
            try:
                import shutil
                shutil.copy2(local_file_path, cache_path)
                logger.info(f"Copied local file to cache: {cache_path}")
            except Exception as e:
                logger.error(f"Failed to copy local file: {e}")
                return {"error": f"Failed to read local file: {str(e)}"}
        else:
            # Download from backend API (for remote files)
            url = f"/api/documents/{document_id}/content"
            params = {
                "origin_index": origin_index,
                "fallback": str(fallback).lower()
            }
            download_timeout = int(os.getenv("MCP_ATTACHMENT_DOWNLOAD_TIMEOUT", "120"))

            try:
                async with httpx.AsyncClient(timeout=download_timeout, verify=not MCP_SKIP_SSL_VERIFY) as http_client:
                    full_url = urljoin(self.base_url + "/", url.lstrip('/'))

                    # Build headers with signing if available
                    headers = dict(self.headers)
                    if self._use_signing:
                        # Build path with query params for signing
                        signing_path = f"/{url.lstrip('/')}?origin_index={origin_index}&fallback={str(fallback).lower()}"
                        sign_headers = self._sign_request("GET", signing_path, b"")
                        headers.update(sign_headers)

                    response = await http_client.get(full_url, headers=headers, params=params)

                    # Handle HTTP errors
                    if response.status_code == 404:
                        return {"error": "Document not found or no accessible origins"}
                    elif response.status_code == 403:
                        return {"error": "Access denied"}
                    elif response.status_code == 503:
                        return {"error": "Origin temporarily unavailable"}

                    response.raise_for_status()
                    content_type = response.headers.get('Content-Type') or content_type

                    # Write file
                    with open(cache_path, 'wb') as f:
                        f.write(response.content)

                    # Determine which origin was used
                    if locations:
                        loc = locations[min(origin_index, len(locations) - 1)] if not fallback else locations[0]
                        origin_info = f"{loc.get('type', 'unknown')}:{loc.get('host', 'unknown')}:{loc.get('path', '')}"

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error downloading document: {e.response.status_code} - {e.response.text}")
                return {"error": f"Backend error: {e.response.status_code}"}
            except httpx.RequestError as e:
                logger.error(f"Request error downloading document: {e}")
                return {"error": f"Failed to connect to backend: {str(e)}"}
            except Exception as e:
                logger.error(f"Unexpected error downloading document: {e}", exc_info=True)
                return {"error": f"Download failed: {str(e)}"}

        return {
            "success": True,
            "cached_path": str(cache_path),
            "original_filename": original_filename,
            "size_bytes": cache_path.stat().st_size,
            "content_type": content_type or doc_result.get("mime_type"),
            "from_cache": False,
            "origin_used": origin_info,
        }

    async def list_imap_folders(self) -> Dict[str, Any]:
        """
        List all folders on the IMAP server.
        
        Calls: GET /api/imap/folders
        
        Uses MCP_ALLOWED_ACCOUNT to determine which account to query.
        
        Returns:
            Dict with folders list and total count
        """
        params = {}
        if self.allowed_account:
            # Use only the first account if comma-separated
            account = self.allowed_account.split(',')[0].strip()
            params["account"] = account
        
        result = await self._request("GET", "/api/imap/folders", params=params)
        return result
    
    async def get_folder_status(self, folder: str) -> Dict[str, Any]:
        """
        Get folder status (message counts) using IMAP STATUS command.
        
        This is the most efficient way to get folder statistics without
        fetching any messages.
        
        Calls: GET /api/imap/folders/{folder}/status
        
        Uses MCP_ALLOWED_ACCOUNT to determine which account to query.
        
        Args:
            folder: Folder name (e.g., "INBOX", "Sent", "Archive/2024")
            
        Returns:
            Dict with:
            - folder: Folder name
            - total: Total number of messages in folder
            - unseen: Number of unread messages
            - recent: Number of recent messages
            - uidnext: Next UID that will be assigned
            - uidvalidity: UID validity value
            - account: Account used
        """
        params = {}
        
        # Add account parameter from MCP_ALLOWED_ACCOUNT
        if self.allowed_account:
            account = self.allowed_account.split(',')[0].strip()
            params["account"] = account
        
        # URL encode the folder path (handle slashes)
        from urllib.parse import quote
        encoded_folder = quote(folder, safe='')
        
        result = await self._request("GET", f"/api/imap/folders/{encoded_folder}/status", params=params)
        return result
    
    async def list_folder_emails(
        self,
        folder: str,
        limit: int = 50,
        since_date: Optional[str] = None,
        include_headers: bool = True
    ) -> Dict[str, Any]:
        """
        List emails in a specific IMAP folder.
        
        Calls: GET /api/imap/folders/{folder}/messages
        
        Uses MCP_ALLOWED_ACCOUNT to determine which account to query.
        
        Args:
            folder: Folder name (e.g., "INBOX", "Sent", "Archive/2024")
            limit: Maximum number of messages (default: 50, max: 500)
            since_date: Only messages since this date (YYYY-MM-DD)
            include_headers: Include full headers like subject, from, date (default: True)
            
        Returns:
            Dict with folder name, messages list, and total count
        """
        params = {
            "limit": min(limit, 500),  # Enforce max
            "include_headers": str(include_headers).lower()
        }
        
        if since_date:
            params["since_date"] = since_date
        
        # Add account parameter from MCP_ALLOWED_ACCOUNT
        if self.allowed_account:
            account = self.allowed_account.split(',')[0].strip()
            params["account"] = account
        
        # URL encode the folder path (handle slashes)
        from urllib.parse import quote
        encoded_folder = quote(folder, safe='')
        
        result = await self._request("GET", f"/api/imap/folders/{encoded_folder}/messages", params=params)
        return result
    
    async def clear_attachment_cache(self, older_than_days: Optional[int] = None) -> Dict[str, Any]:
        """
        Clear attachment cache directory.
        
        Args:
            older_than_days: If set, only delete files older than this many days
            
        Returns:
            {"success": true, "files_deleted": 15, "bytes_freed": 52428800, "cache_dir": "..."}
        """
        cache_dir_str = os.getenv("MCP_FILE_CACHE_DIR")
        if not cache_dir_str:
            return {"error": "MCP_FILE_CACHE_DIR not configured"}
        
        cache_dir = Path(cache_dir_str).expanduser()
        if not cache_dir.exists():
            return {
                "success": True,
                "files_deleted": 0,
                "bytes_freed": 0,
                "cache_dir": str(cache_dir)
            }
        
        files_deleted = 0
        bytes_freed = 0
        cutoff = datetime.now() - timedelta(days=older_than_days) if older_than_days else None
        
        for file in cache_dir.iterdir():
            if file.is_file():
                should_delete = True
                if cutoff:
                    mtime = datetime.fromtimestamp(file.stat().st_mtime)
                    should_delete = mtime < cutoff
                
                if should_delete:
                    size = file.stat().st_size
                    file.unlink()
                    files_deleted += 1
                    bytes_freed += size
        
        return {
            "success": True,
            "files_deleted": files_deleted,
            "bytes_freed": bytes_freed,
            "cache_dir": str(cache_dir)
        }
