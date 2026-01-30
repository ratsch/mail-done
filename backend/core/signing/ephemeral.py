"""
Ephemeral Session Registry

Manages ephemeral session keys for OAuth-authenticated clients (Web-UI, V0 Portal).
Sessions are created via the OAuth handshake and expire after a configurable TTL.

Features:
- Session storage with TTL
- Nonce cache for replay prevention
- Automatic cleanup of expired sessions
- Persistence on graceful shutdown
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, FrozenSet, Set
import secrets

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from backend.core.signing.keys import base64_to_public_key, public_key_to_base64
from backend.core.signing.scopes import Scope, parse_scopes

logger = logging.getLogger(__name__)


# Default TTL for nonce cache (must be > timestamp tolerance)
NONCE_CACHE_TTL_SECONDS = 600  # 10 minutes (2x the timestamp tolerance)

# Maximum nonces to keep per session (memory limit)
MAX_NONCES_PER_SESSION = 10000

# Cleanup interval for expired entries
CLEANUP_INTERVAL_SECONDS = 60


@dataclass
class EphemeralSession:
    """
    An ephemeral session created via OAuth handshake.
    
    Attributes:
        session_id: Unique session identifier
        client_type: Type of client (web-ui, v0-portal)
        user_email: Authenticated user's email
        public_key: Client's ephemeral public key
        public_key_b64: Base64-encoded public key (for serialization)
        scopes: Granted permission scopes
        created_at: Unix timestamp of creation
        expires_at: Unix timestamp of expiration
        nonces_used: Set of nonces used in this session (for replay prevention)
    """
    session_id: str
    client_type: str
    user_email: str
    public_key: Ed25519PublicKey
    public_key_b64: str
    scopes: FrozenSet[Scope]
    created_at: float
    expires_at: float
    nonces_used: Set[str] = field(default_factory=set)
    nonce_timestamps: Dict[str, float] = field(default_factory=dict)  # nonce -> timestamp
    
    @property
    def is_expired(self) -> bool:
        """Check if session has expired."""
        return time.time() > self.expires_at
    
    @property
    def remaining_seconds(self) -> float:
        """Seconds until expiration."""
        return max(0, self.expires_at - time.time())
    
    def check_and_record_nonce(self, nonce: str) -> bool:
        """
        Check if nonce was used and record it.
        
        Args:
            nonce: The nonce to check
            
        Returns:
            True if nonce was already used (replay attack), False if new
        """
        now = time.time()
        
        # Clean old nonces first
        self._cleanup_nonces(now)
        
        if nonce in self.nonces_used:
            return True  # Replay detected
        
        # Record new nonce
        self.nonces_used.add(nonce)
        self.nonce_timestamps[nonce] = now
        
        # Enforce memory limit
        if len(self.nonces_used) > MAX_NONCES_PER_SESSION:
            self._enforce_nonce_limit()
        
        return False  # New nonce, OK
    
    def _cleanup_nonces(self, now: float) -> None:
        """Remove nonces older than TTL."""
        cutoff = now - NONCE_CACHE_TTL_SECONDS
        expired = [n for n, ts in self.nonce_timestamps.items() if ts < cutoff]
        for nonce in expired:
            self.nonces_used.discard(nonce)
            del self.nonce_timestamps[nonce]
    
    def _enforce_nonce_limit(self) -> None:
        """Remove oldest nonces if over limit."""
        if len(self.nonces_used) <= MAX_NONCES_PER_SESSION:
            return
        
        # Sort by timestamp and remove oldest
        sorted_nonces = sorted(self.nonce_timestamps.items(), key=lambda x: x[1])
        to_remove = len(self.nonces_used) - MAX_NONCES_PER_SESSION + 100  # Remove 100 extra
        for nonce, _ in sorted_nonces[:to_remove]:
            self.nonces_used.discard(nonce)
            del self.nonce_timestamps[nonce]
    
    def to_dict(self) -> dict:
        """Serialize session for persistence (without nonces - they're not persisted)."""
        return {
            "session_id": self.session_id,
            "client_type": self.client_type,
            "user_email": self.user_email,
            "public_key_b64": self.public_key_b64,
            "scopes": [str(s) for s in self.scopes],
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "EphemeralSession":
        """Deserialize session from persistence."""
        public_key = base64_to_public_key(data["public_key_b64"])
        scopes = parse_scopes(data["scopes"])
        
        return cls(
            session_id=data["session_id"],
            client_type=data["client_type"],
            user_email=data["user_email"],
            public_key=public_key,
            public_key_b64=data["public_key_b64"],
            scopes=scopes,
            created_at=data["created_at"],
            expires_at=data["expires_at"],
        )


class EphemeralSessionRegistry:
    """
    Registry of ephemeral sessions.
    
    Thread-safe for concurrent access.
    Automatically cleans up expired sessions.
    """
    
    def __init__(self):
        self._sessions: Dict[str, EphemeralSession] = {}
        self._key_to_session: Dict[str, str] = {}  # public_key_b64 -> session_id
        self._lock = threading.RLock()
        self._cleanup_thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self) -> None:
        """Start the background cleanup thread."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
        logger.info("Ephemeral session registry started")
    
    def stop(self) -> None:
        """Stop the background cleanup thread."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
        logger.info("Ephemeral session registry stopped")
    
    def _cleanup_loop(self) -> None:
        """Background thread that cleans up expired sessions."""
        while self._running:
            time.sleep(CLEANUP_INTERVAL_SECONDS)
            self._cleanup_expired()
    
    def _cleanup_expired(self) -> None:
        """Remove expired sessions."""
        with self._lock:
            expired = [sid for sid, session in self._sessions.items() if session.is_expired]
            for session_id in expired:
                self._remove_session_unsafe(session_id)
            
            if expired:
                logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def _remove_session_unsafe(self, session_id: str) -> None:
        """Remove session without lock (caller must hold lock)."""
        session = self._sessions.pop(session_id, None)
        if session:
            self._key_to_session.pop(session.public_key_b64, None)
    
    def create_session(
        self,
        client_type: str,
        user_email: str,
        public_key: Ed25519PublicKey,
        scopes: FrozenSet[Scope],
        ttl_seconds: int,
    ) -> EphemeralSession:
        """
        Create a new ephemeral session.
        
        Args:
            client_type: Type of client (web-ui, v0-portal)
            user_email: Authenticated user's email
            public_key: Client's ephemeral public key
            scopes: Granted permission scopes
            ttl_seconds: Session lifetime in seconds
            
        Returns:
            Created session
            
        Raises:
            ValueError: If a session with this key already exists
        """
        public_key_b64 = public_key_to_base64(public_key)
        now = time.time()
        
        # Generate unique session ID
        session_id = f"{client_type}-{secrets.token_urlsafe(16)}"
        
        session = EphemeralSession(
            session_id=session_id,
            client_type=client_type,
            user_email=user_email,
            public_key=public_key,
            public_key_b64=public_key_b64,
            scopes=scopes,
            created_at=now,
            expires_at=now + ttl_seconds,
        )
        
        with self._lock:
            # Check for existing session with same key
            if public_key_b64 in self._key_to_session:
                # Revoke old session
                old_session_id = self._key_to_session[public_key_b64]
                logger.info(f"Revoking old session {old_session_id} for new session")
                self._remove_session_unsafe(old_session_id)
            
            self._sessions[session_id] = session
            self._key_to_session[public_key_b64] = session_id
        
        logger.info(
            f"Created session {session_id} for {user_email} "
            f"(type={client_type}, ttl={ttl_seconds}s)"
        )
        return session
    
    def get_session(self, session_id: str) -> Optional[EphemeralSession]:
        """
        Get a session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Session if found and not expired, None otherwise
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session and not session.is_expired:
                return session
            
            # Clean up if expired
            if session and session.is_expired:
                self._remove_session_unsafe(session_id)
            
            return None
    
    def get_session_by_key(self, public_key: Ed25519PublicKey) -> Optional[EphemeralSession]:
        """
        Find session by public key.
        
        Args:
            public_key: Ed25519 public key
            
        Returns:
            Session if found and not expired, None otherwise
        """
        public_key_b64 = public_key_to_base64(public_key)
        
        with self._lock:
            session_id = self._key_to_session.get(public_key_b64)
            if session_id:
                return self.get_session(session_id)
            return None
    
    def revoke_session(self, session_id: str) -> bool:
        """
        Revoke a session.
        
        Args:
            session_id: Session to revoke
            
        Returns:
            True if session was found and revoked
        """
        with self._lock:
            if session_id in self._sessions:
                self._remove_session_unsafe(session_id)
                logger.info(f"Revoked session {session_id}")
                return True
            return False
    
    def check_nonce(self, session_id: str, nonce: str) -> bool:
        """
        Check and record a nonce for replay prevention.
        
        Args:
            session_id: Session to check
            nonce: Nonce to check
            
        Returns:
            True if nonce was already used (replay), False if new
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return True  # No session = treat as replay
            return session.check_and_record_nonce(nonce)
    
    def list_sessions(self) -> List[dict]:
        """Get info about all active sessions (for admin/debugging)."""
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "client_type": s.client_type,
                    "user_email": s.user_email,
                    "created_at": s.created_at,
                    "expires_at": s.expires_at,
                    "remaining_seconds": s.remaining_seconds,
                    "is_expired": s.is_expired,
                }
                for s in self._sessions.values()
            ]
    
    def get_all_sessions(self) -> List[EphemeralSession]:
        """Get all sessions for persistence."""
        with self._lock:
            return [s for s in self._sessions.values() if not s.is_expired]
    
    def restore_sessions(self, sessions: List[EphemeralSession]) -> int:
        """
        Restore sessions from persistence.
        
        Args:
            sessions: List of sessions to restore
            
        Returns:
            Number of sessions restored (excludes expired)
        """
        count = 0
        with self._lock:
            for session in sessions:
                if session.is_expired:
                    continue
                self._sessions[session.session_id] = session
                self._key_to_session[session.public_key_b64] = session.session_id
                count += 1
        
        logger.info(f"Restored {count} sessions from persistence")
        return count


# Global registry instance
ephemeral_registry = EphemeralSessionRegistry()
