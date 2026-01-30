"""
Suspicious Activity Monitor

Detects anomalous API usage patterns and provides:
1. Email alerts for suspicious activity
2. API lockdown capability with admin unlock
3. Anomaly detection based on configurable thresholds

Configuration via environment variables:
- SECURITY_ALERT_EMAIL: Email to send alerts to (required for alerts)
- SECURITY_LOCKDOWN_THRESHOLD: Number of suspicious events before lockdown (default: 50)
- SECURITY_ALERT_THRESHOLD: Number of suspicious events before alert (default: 10)
- SECURITY_WINDOW_MINUTES: Time window for counting events (default: 60)
- SECURITY_LOCKDOWN_ENABLED: Enable automatic lockdown (default: true)
- SECURITY_UNLOCK_KEY: Key to unlock API after lockdown (required for production)

SMTP configuration is loaded from the 'personal' account in AccountManager.
No separate SMTP_* variables needed.
"""
import logging
import os
import smtplib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum
from threading import Lock, Thread
from typing import Dict, List, Optional, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Severity levels for security alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class SuspiciousEventType(Enum):
    """Types of suspicious activity detected."""
    RATE_LIMIT_EXCEEDED = "rate_limit_exceeded"
    FAILED_AUTH = "failed_auth"
    INVALID_API_KEY = "invalid_api_key"
    UNUSUAL_ENDPOINT = "unusual_endpoint"
    HIGH_ERROR_RATE = "high_error_rate"
    LARGE_RESPONSE = "large_response"
    BRUTE_FORCE = "brute_force"
    SCANNING = "scanning"


@dataclass
class SuspiciousEvent:
    """Record of a suspicious activity event."""
    event_type: SuspiciousEventType
    ip_address: str
    endpoint: str
    timestamp: float = field(default_factory=time.time)
    details: str = ""
    level: AlertLevel = AlertLevel.WARNING


@dataclass
class SecurityStatus:
    """Current security status of the API."""
    is_locked: bool = False
    locked_at: Optional[float] = None
    locked_reason: Optional[str] = None
    lockdown_expires: Optional[float] = None  # Auto-unlock after this time
    events_in_window: int = 0
    last_alert_sent: Optional[float] = None


class SecurityMonitor:
    """
    Monitors API for suspicious activity and manages lockdown state.
    """
    
    def __init__(self):
        self.lock = Lock()
        
        # Configuration from environment
        self.alert_email = os.getenv("SECURITY_ALERT_EMAIL")
        self.lockdown_threshold = int(os.getenv("SECURITY_LOCKDOWN_THRESHOLD", "50"))
        self.alert_threshold = int(os.getenv("SECURITY_ALERT_THRESHOLD", "10"))
        self.window_minutes = int(os.getenv("SECURITY_WINDOW_MINUTES", "60"))
        self.lockdown_enabled = os.getenv("SECURITY_LOCKDOWN_ENABLED", "true").lower() in ("true", "1", "yes")
        self.auto_unlock_hours = int(os.getenv("SECURITY_AUTO_UNLOCK_HOURS", "24"))
        
        # State
        self.events: List[SuspiciousEvent] = []
        self.status = SecurityStatus()
        self.alerted_ips: Set[str] = set()  # IPs we've already alerted about
        
        # SMTP configuration - loaded lazily from AccountManager 'personal' account
        self._smtp_config_loaded = False
        self.smtp_host: Optional[str] = None
        self.smtp_port: int = 465
        self.smtp_username: Optional[str] = None
        self.smtp_password: Optional[str] = None
        self.smtp_from_address: Optional[str] = None  # Full email for From: header
        self.smtp_use_ssl: bool = True  # Port 465 uses SSL directly
        self.smtp_use_tls: bool = False
        
        # Rate tracking per IP
        self.ip_event_counts: Dict[str, int] = defaultdict(int)
        
        # Admin unlock key (for emergency unlock)
        self.unlock_key = os.getenv("SECURITY_UNLOCK_KEY", None)
        if not self.unlock_key:
            import secrets
            self.unlock_key = secrets.token_urlsafe(32)
            logger.warning(f"No SECURITY_UNLOCK_KEY set. Generated temporary key: {self.unlock_key[:16]}...")
        
        logger.info(
            f"Security Monitor initialized: alert_threshold={self.alert_threshold}, "
            f"lockdown_threshold={self.lockdown_threshold}, lockdown_enabled={self.lockdown_enabled}, "
            f"alert_email={self.alert_email}"
        )
    
    def _load_smtp_config(self):
        """Load SMTP configuration from AccountManager 'personal' account."""
        if self._smtp_config_loaded:
            return
        
        try:
            from backend.core.accounts.manager import AccountManager
            manager = AccountManager()
            personal = manager.get_account('personal')
            
            self.smtp_host = personal.smtp_host
            self.smtp_port = personal.smtp_port
            self.smtp_username = personal.smtp_username
            self.smtp_password = personal.smtp_password
            self.smtp_use_tls = personal.smtp_use_tls
            # Port 465 uses SSL, not STARTTLS
            self.smtp_use_ssl = (personal.smtp_port == 465 and not personal.smtp_use_tls)
            
            # Derive email address from username if needed
            # smtp_username might be just "user" (login name) instead of "user@example.com" (email)
            if self.smtp_username and '@' not in self.smtp_username:
                # Extract domain from SMTP host (smtp.example.com -> example.com)
                smtp_domain = self.smtp_host.replace('smtp.', '') if self.smtp_host.startswith('smtp.') else self.smtp_host
                self.smtp_from_address = f"{self.smtp_username}@{smtp_domain}"
            else:
                self.smtp_from_address = self.smtp_username
            
            self._smtp_config_loaded = True
            logger.info(f"Security monitor SMTP loaded from 'personal' account: {self.smtp_host}:{self.smtp_port}, from={self.smtp_from_address}")
        except Exception as e:
            logger.error(f"Failed to load SMTP config from AccountManager: {e}")
            self._smtp_config_loaded = True  # Don't retry
    
    def record_event(self, event: SuspiciousEvent) -> bool:
        """
        Record a suspicious event and check thresholds.
        
        Returns:
            True if API was locked as a result
        """
        with self.lock:
            self.events.append(event)
            self.ip_event_counts[event.ip_address] += 1
            
            # Prune old events
            cutoff = time.time() - (self.window_minutes * 60)
            self.events = [e for e in self.events if e.timestamp > cutoff]
            
            # Count events in window
            self.status.events_in_window = len(self.events)
            
            # Log the event
            logger.warning(
                f"SECURITY EVENT [{event.level.value}]: {event.event_type.value} "
                f"from {event.ip_address} at {event.endpoint} - {event.details}"
            )
            
            # Check if we should alert
            if self.status.events_in_window >= self.alert_threshold:
                self._maybe_send_alert(event)
            
            # Check if we should lock down
            if self.lockdown_enabled and self.status.events_in_window >= self.lockdown_threshold:
                if not self.status.is_locked:
                    self._lockdown(f"Threshold exceeded: {self.status.events_in_window} events in {self.window_minutes} minutes")
                    return True
            
            return False
    
    def _maybe_send_alert(self, trigger_event: SuspiciousEvent):
        """Send email alert if conditions are met."""
        if not self.alert_email:
            return
        
        # Don't spam alerts - max one per 5 minutes
        if self.status.last_alert_sent and time.time() - self.status.last_alert_sent < 300:
            return
        
        # Send alert in background thread
        Thread(target=self._send_alert_email, args=(trigger_event,), daemon=True).start()
        self.status.last_alert_sent = time.time()
    
    def _send_alert_email(self, trigger_event: SuspiciousEvent):
        """Send alert email (runs in background thread)."""
        try:
            # Load SMTP config from AccountManager if not already loaded
            self._load_smtp_config()
            
            if not all([self.smtp_host, self.smtp_username, self.smtp_password]):
                logger.warning("SMTP not configured - cannot send security alert")
                return
            
            # Build email
            msg = MIMEMultipart()
            msg["From"] = self.smtp_from_address
            msg["To"] = self.alert_email
            
            level_emoji = {"info": "ℹ️", "warning": "⚠️", "critical": "🚨"}.get(trigger_event.level.value, "⚠️")
            # Distinctive subject for rule matching - DO NOT CHANGE without updating classification_rules.yaml
            msg["Subject"] = f"{level_emoji} [SECURITY-ALERT] Mail-Done API: {trigger_event.event_type.value}"
            
            # Get recent events summary
            with self.lock:
                recent_events = self.events[-20:]  # Last 20 events
                ip_counts = dict(self.ip_event_counts)
            
            body = f"""
Security Alert from Mail-Done API

TRIGGER EVENT:
- Type: {trigger_event.event_type.value}
- Level: {trigger_event.level.value.upper()}
- IP Address: {trigger_event.ip_address}
- Endpoint: {trigger_event.endpoint}
- Details: {trigger_event.details}
- Time: {datetime.fromtimestamp(trigger_event.timestamp).isoformat()}

CURRENT STATUS:
- Events in last {self.window_minutes} minutes: {len(recent_events)}
- API Locked: {self.status.is_locked}
- Lockdown enabled: {self.lockdown_enabled}
- Lockdown threshold: {self.lockdown_threshold}

TOP SUSPICIOUS IPs:
"""
            # Add top IPs
            top_ips = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            for ip, count in top_ips:
                body += f"  - {ip}: {count} events\n"
            
            body += f"""
RECENT EVENTS:
"""
            for e in recent_events[-10:]:
                body += f"  - [{e.level.value}] {e.event_type.value} from {e.ip_address} at {e.endpoint}\n"
            
            if self.lockdown_enabled:
                body += """
LOCKDOWN STATUS:
If the API is locked, you can unlock it by calling:
  POST /api/security/unlock
  with header X-Unlock-Key: <your SECURITY_UNLOCK_KEY from .env>

NOTE: The unlock key is NOT included in this email for security reasons.
Retrieve it from your .env file or server environment variables.
"""
            
            msg.attach(MIMEText(body, "plain"))
            
            # Send email - use SSL for port 465, TLS for port 587
            if self.smtp_use_ssl:
                # Port 465: SSL from the start
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port, timeout=30)
            else:
                # Port 587: STARTTLS
                server = smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=30)
                if self.smtp_use_tls:
                    server.starttls()
            
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Security alert sent to {self.alert_email}")
            
        except Exception as e:
            logger.error(f"Failed to send security alert email: {e}")
    
    def _lockdown(self, reason: str):
        """Lock down the API."""
        self.status.is_locked = True
        self.status.locked_at = time.time()
        self.status.locked_reason = reason
        
        # Set auto-unlock time if configured
        if self.auto_unlock_hours > 0:
            self.status.lockdown_expires = time.time() + (self.auto_unlock_hours * 3600)
        
        logger.critical(f"API LOCKED DOWN: {reason}")
        
        # Send critical alert
        event = SuspiciousEvent(
            event_type=SuspiciousEventType.BRUTE_FORCE,
            ip_address="system",
            endpoint="lockdown",
            details=reason,
            level=AlertLevel.CRITICAL
        )
        Thread(target=self._send_alert_email, args=(event,), daemon=True).start()
    
    def unlock(self, unlock_key: str) -> bool:
        """
        Unlock the API with the admin key.
        
        Args:
            unlock_key: The SECURITY_UNLOCK_KEY value
            
        Returns:
            True if unlock was successful
        """
        import secrets
        
        # Constant-time comparison to prevent timing attacks
        if not secrets.compare_digest(unlock_key, self.unlock_key):
            logger.warning("SECURITY: Invalid unlock key attempted")
            return False
        
        with self.lock:
            self.status.is_locked = False
            self.status.locked_at = None
            self.status.locked_reason = None
            self.status.lockdown_expires = None
            self.events.clear()
            self.ip_event_counts.clear()
            self.alerted_ips.clear()
        
        logger.info("API unlocked by admin")
        return True
    
    def check_auto_unlock(self):
        """Check if lockdown has expired and auto-unlock if so."""
        with self.lock:
            if self.status.is_locked and self.status.lockdown_expires:
                if time.time() > self.status.lockdown_expires:
                    logger.info("API auto-unlocked after lockdown expiry")
                    self.status.is_locked = False
                    self.status.locked_at = None
                    self.status.locked_reason = None
                    self.status.lockdown_expires = None
    
    def is_locked(self) -> bool:
        """Check if API is currently locked."""
        self.check_auto_unlock()
        return self.status.is_locked
    
    def get_status(self) -> dict:
        """Get current security status."""
        self.check_auto_unlock()
        with self.lock:
            return {
                "is_locked": self.status.is_locked,
                "locked_at": datetime.fromtimestamp(self.status.locked_at).isoformat() if self.status.locked_at else None,
                "locked_reason": self.status.locked_reason,
                "lockdown_expires": datetime.fromtimestamp(self.status.lockdown_expires).isoformat() if self.status.lockdown_expires else None,
                "events_in_window": self.status.events_in_window,
                "window_minutes": self.window_minutes,
                "alert_threshold": self.alert_threshold,
                "lockdown_threshold": self.lockdown_threshold,
                "lockdown_enabled": self.lockdown_enabled
            }


# Global monitor instance
security_monitor = SecurityMonitor()


class SecurityMonitorMiddleware(BaseHTTPMiddleware):
    """
    Middleware that checks security status and records suspicious activity.
    """
    
    # Exempt paths (health checks, docs, unlock endpoint)
    EXEMPT_PATHS = ["/health", "/docs", "/redoc", "/openapi.json", "/api/security/unlock"]
    
    # Paths that typically have higher error rates (don't flag these)
    HIGH_ERROR_PATHS = ["/auth/", "/api/search"]
    
    async def dispatch(self, request: Request, call_next):
        # Skip for exempt paths
        path = request.url.path
        if any(path.startswith(p) for p in self.EXEMPT_PATHS):
            return await call_next(request)
        
        # Check if API is locked
        if security_monitor.is_locked():
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "API temporarily unavailable due to security lockdown",
                    "locked_reason": security_monitor.status.locked_reason,
                    "contact": "Contact admin to unlock"
                }
            )
        
        # Get client info
        client_ip = request.client.host if request.client else "unknown"
        
        # Process request
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time
        
        # Analyze response for suspicious patterns
        await self._analyze_response(request, response, client_ip, duration)
        
        return response
    
    async def _analyze_response(self, request: Request, response, client_ip: str, duration: float):
        """Analyze request/response for suspicious patterns."""
        path = request.url.path
        status = response.status_code
        
        # Pattern 1: Failed authentication
        if status == 401:
            security_monitor.record_event(SuspiciousEvent(
                event_type=SuspiciousEventType.FAILED_AUTH,
                ip_address=client_ip,
                endpoint=path,
                details=f"401 Unauthorized",
                level=AlertLevel.WARNING
            ))
        
        # Pattern 2: Forbidden (invalid API key)
        elif status == 403:
            security_monitor.record_event(SuspiciousEvent(
                event_type=SuspiciousEventType.INVALID_API_KEY,
                ip_address=client_ip,
                endpoint=path,
                details=f"403 Forbidden",
                level=AlertLevel.WARNING
            ))
        
        # Pattern 3: Rate limited
        elif status == 429:
            security_monitor.record_event(SuspiciousEvent(
                event_type=SuspiciousEventType.RATE_LIMIT_EXCEEDED,
                ip_address=client_ip,
                endpoint=path,
                details=f"Rate limit exceeded",
                level=AlertLevel.WARNING
            ))
        
        # Pattern 4: 404s on sensitive paths (scanning)
        elif status == 404 and any(s in path for s in ["/admin", "/config", "/.env", "/debug", "/backup"]):
            security_monitor.record_event(SuspiciousEvent(
                event_type=SuspiciousEventType.SCANNING,
                ip_address=client_ip,
                endpoint=path,
                details=f"Potential path scanning - 404 on sensitive path",
                level=AlertLevel.CRITICAL
            ))
        
        # Pattern 5: High error rate (multiple 5xx errors)
        elif status >= 500:
            # Only flag if not in high-error paths
            if not any(p in path for p in self.HIGH_ERROR_PATHS):
                security_monitor.record_event(SuspiciousEvent(
                    event_type=SuspiciousEventType.HIGH_ERROR_RATE,
                    ip_address=client_ip,
                    endpoint=path,
                    details=f"Server error {status}",
                    level=AlertLevel.INFO
                ))


def get_security_router():
    """Create FastAPI router for security endpoints."""
    from fastapi import APIRouter, HTTPException, Header
    
    router = APIRouter(prefix="/api/security", tags=["security"])
    
    @router.get("/status")
    async def get_security_status():
        """Get current security status (no auth required for monitoring)."""
        return security_monitor.get_status()
    
    @router.post("/unlock")
    async def unlock_api(x_unlock_key: str = Header(...)):
        """
        Unlock the API after a security lockdown.
        
        Requires the SECURITY_UNLOCK_KEY header value.
        """
        if security_monitor.unlock(x_unlock_key):
            return {"message": "API unlocked successfully"}
        raise HTTPException(status_code=403, detail="Invalid unlock key")
    
    @router.post("/test-alert")
    async def test_alert(x_api_key: str = Header(...)):
        """Send a test security alert email (requires API key)."""
        from backend.api.auth import verify_api_key
        
        # Verify API key
        api_key = os.getenv("API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="API_KEY not configured")
        
        import secrets
        if not secrets.compare_digest(x_api_key, api_key):
            raise HTTPException(status_code=403, detail="Invalid API key")
        
        # Send test alert
        test_event = SuspiciousEvent(
            event_type=SuspiciousEventType.SCANNING,
            ip_address="test",
            endpoint="/api/security/test-alert",
            details="This is a test security alert",
            level=AlertLevel.INFO
        )
        security_monitor._send_alert_email(test_event)
        
        return {"message": f"Test alert sent to {security_monitor.alert_email}"}
    
    return router
