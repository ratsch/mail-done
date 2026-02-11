"""
General Rate Limiting Middleware

Rate limits all API endpoints (not just review system) to prevent abuse.
"""
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, Tuple
from threading import Lock
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse

logger = logging.getLogger(__name__)


class APIRateLimiter:
    """
    Token bucket rate limiter for API endpoints.

    Tracks requests per IP and per API key with configurable limits.
    """

    def __init__(self):
        self.ip_buckets: Dict[str, Tuple[int, float]] = {}  # ip -> (tokens, last_refill)
        self.key_buckets: Dict[str, Tuple[int, float]] = {}  # api_key -> (tokens, last_refill)
        self.shared_buckets: Dict[str, Tuple[int, float]] = {}  # ip -> (tokens, last_refill) for /shared/ endpoints
        self.lock = Lock()

        # Rate limits (tokens per minute)
        self.IP_RATE_LIMIT = 60  # 60 requests per minute per IP
        self.KEY_RATE_LIMIT = 600  # 600 requests per minute per API key
        self.SHARED_RATE_LIMIT = 30  # 30 requests per minute per IP for /shared/ (stricter)
        self.BURST_SIZE = 10  # Allow bursts up to 10 requests
        self.SHARED_BURST_SIZE = 5  # Smaller burst for shared endpoints

        # Security: Track suspicious patterns
        self.failed_auth_attempts: Dict[str, list] = defaultdict(list)  # ip -> [timestamps]
        self.MAX_FAILED_ATTEMPTS = 50  # per hour (was 10, too low for same-host frontend)

        # IPs exempt from failed-auth blocking (trusted same-host services)
        self.TRUSTED_IPS = {"127.0.0.1", "::1"}

        logger.info(f"Rate limiter initialized: {self.IP_RATE_LIMIT} req/min per IP, {self.KEY_RATE_LIMIT} req/min per key, {self.SHARED_RATE_LIMIT} req/min for /shared/")
    
    def _refill_bucket(self, tokens: int, last_refill: float, rate: int) -> Tuple[int, float]:
        """Refill token bucket based on elapsed time."""
        now = time.time()
        elapsed = now - last_refill
        
        # Refill at specified rate (tokens per second)
        tokens_to_add = int(elapsed * (rate / 60.0))
        tokens = min(tokens + tokens_to_add, rate + self.BURST_SIZE)
        
        return tokens, now
    
    def check_rate_limit(self, ip: str, api_key: str = None, path: str = None) -> Tuple[bool, str]:
        """
        Check if request should be allowed.

        Args:
            ip: Client IP address
            api_key: Optional API key for authenticated requests
            path: Request path (used for stricter limits on /shared/)

        Returns:
            (allowed, reason) - True if allowed, False + reason if blocked
        """
        with self.lock:
            # Check for IP-based blocking (failed auth)
            # Skip for trusted IPs (same-host services like the app frontend)
            if ip not in self.TRUSTED_IPS and ip in self.failed_auth_attempts:
                recent_failures = [
                    ts for ts in self.failed_auth_attempts[ip]
                    if time.time() - ts < 3600  # Last hour
                ]
                self.failed_auth_attempts[ip] = recent_failures

                if len(recent_failures) >= self.MAX_FAILED_ATTEMPTS:
                    logger.warning(f"SECURITY: IP {ip} blocked due to {len(recent_failures)} failed auth attempts")
                    return False, "Too many failed authentication attempts. Try again later."

            # Check if this is a /shared/ endpoint (stricter limits)
            if path and path.startswith("/shared/"):
                tokens, last_refill = self.shared_buckets.get(ip, (self.SHARED_RATE_LIMIT, time.time()))
                tokens, last_refill = self._refill_bucket(tokens, last_refill, self.SHARED_RATE_LIMIT)

                if tokens < 1:
                    logger.warning(f"Shared endpoint rate limit exceeded for IP: {ip}")
                    return False, f"Rate limit exceeded: {self.SHARED_RATE_LIMIT} requests per minute for shared links"

                self.shared_buckets[ip] = (tokens - 1, last_refill)
                return True, ""

            # Check IP rate limit
            tokens, last_refill = self.ip_buckets.get(ip, (self.IP_RATE_LIMIT, time.time()))
            tokens, last_refill = self._refill_bucket(tokens, last_refill, self.IP_RATE_LIMIT)

            if tokens < 1:
                logger.warning(f"Rate limit exceeded for IP: {ip}")
                return False, f"Rate limit exceeded: {self.IP_RATE_LIMIT} requests per minute per IP"

            self.ip_buckets[ip] = (tokens - 1, last_refill)

            # Check API key rate limit (if authenticated)
            if api_key:
                tokens, last_refill = self.key_buckets.get(api_key, (self.KEY_RATE_LIMIT, time.time()))
                tokens, last_refill = self._refill_bucket(tokens, last_refill, self.KEY_RATE_LIMIT)

                if tokens < 1:
                    logger.warning(f"Rate limit exceeded for API key: {api_key[:8]}...")
                    return False, f"Rate limit exceeded: {self.KEY_RATE_LIMIT} requests per minute per key"

                self.key_buckets[api_key] = (tokens - 1, last_refill)

            return True, ""
    
    def record_failed_auth(self, ip: str):
        """Record a failed authentication attempt."""
        with self.lock:
            self.failed_auth_attempts[ip].append(time.time())
            logger.warning(f"SECURITY: Failed auth attempt from IP {ip} ({len(self.failed_auth_attempts[ip])} total)")
    
    def cleanup_old_entries(self):
        """Remove stale entries to prevent memory bloat."""
        with self.lock:
            now = time.time()

            # Remove IP buckets inactive for 10 minutes
            self.ip_buckets = {
                ip: bucket for ip, bucket in self.ip_buckets.items()
                if now - bucket[1] < 600
            }

            # Remove key buckets inactive for 10 minutes
            self.key_buckets = {
                key: bucket for key, bucket in self.key_buckets.items()
                if now - bucket[1] < 600
            }

            # Remove shared buckets inactive for 10 minutes
            self.shared_buckets = {
                ip: bucket for ip, bucket in self.shared_buckets.items()
                if now - bucket[1] < 600
            }

            # Clean up failed auth attempts older than 1 hour
            for ip in list(self.failed_auth_attempts.keys()):
                recent = [ts for ts in self.failed_auth_attempts[ip] if now - ts < 3600]
                if recent:
                    self.failed_auth_attempts[ip] = recent
                else:
                    del self.failed_auth_attempts[ip]


# Global rate limiter instance
rate_limiter = APIRateLimiter()


class GeneralRateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware to apply rate limiting to all API endpoints.
    """
    
    # Exempt paths (health checks, docs)
    EXEMPT_PATHS = ["/health", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for exempt paths
        if any(request.url.path.startswith(path) for path in self.EXEMPT_PATHS):
            return await call_next(request)

        # Get client IP
        client_ip = request.client.host if request.client else "unknown"

        # Get API key if present
        api_key = request.headers.get("X-API-Key")

        # Get request path for endpoint-specific rate limiting
        request_path = request.url.path

        # Check rate limit (pass path for /shared/ endpoint stricter limits)
        allowed, reason = rate_limiter.check_rate_limit(client_ip, api_key, request_path)

        if not allowed:
            logger.warning(f"Rate limit blocked: {client_ip} - {request.url.path}")
            # Determine which limit was hit for headers
            limit = rate_limiter.SHARED_RATE_LIMIT if request_path.startswith("/shared/") else rate_limiter.IP_RATE_LIMIT
            return JSONResponse(
                status_code=429,
                content={
                    "detail": reason,
                    "retry_after": 60
                },
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(limit),
                    "X-RateLimit-Remaining": "0"
                }
            )

        # Process request
        response = await call_next(request)

        # Record failed auth for security monitoring
        if response.status_code in [401, 403]:
            rate_limiter.record_failed_auth(client_ip)

        return response


def cleanup_rate_limiter():
    """Periodic cleanup task (call from background thread)."""
    rate_limiter.cleanup_old_entries()
