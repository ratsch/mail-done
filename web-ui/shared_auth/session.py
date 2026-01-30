"""
In-Memory Session Store

Stores ephemeral session keys in memory. Sessions are automatically
cleaned up when they expire.

Security: Private keys are stored only in RAM, never persisted to disk.
"""

import logging
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """
    Represents an authenticated session.
    
    Attributes:
        session_id: Backend session ID
        private_key: Ephemeral Ed25519 private key
        user_email: Authenticated user's email
        expires_at: Unix timestamp when session expires
        scopes: Permission scopes granted
    """
    session_id: str
    private_key: Ed25519PrivateKey
    user_email: str
    expires_at: float
    scopes: list
    
    @property
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    @property
    def remaining_seconds(self) -> float:
        return max(0, self.expires_at - time.time())


class SessionStore:
    """
    Thread-safe in-memory session storage.
    
    Manages ephemeral sessions created via OAuth handshake.
    Sessions are automatically expired but not persisted.
    """
    
    def __init__(self):
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()
        self._cleanup_interval = 60  # seconds
        self._last_cleanup = 0
    
    def store(
        self,
        session_id: str,
        private_key: Ed25519PrivateKey,
        user_email: str,
        expires_in: int,
        scopes: list = None,
    ) -> None:
        """
        Store a new session.
        
        Args:
            session_id: Backend session ID
            private_key: Ephemeral private key
            user_email: Authenticated user's email
            expires_in: Seconds until expiration
            scopes: Granted permission scopes
        """
        with self._lock:
            self._maybe_cleanup()
            
            self._sessions[session_id] = Session(
                session_id=session_id,
                private_key=private_key,
                user_email=user_email,
                expires_at=time.time() + expires_in,
                scopes=scopes or [],
            )
            
            logger.info(f"Session stored: {session_id[:16]}... for {user_email}")
    
    def get(self, session_id: str) -> Optional[Session]:
        """
        Get session by ID.
        
        Returns None if session doesn't exist or is expired.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            
            if session is None:
                logger.warning(f"Session not found: {session_id[:16]}... (store has {len(self._sessions)} sessions)")
                return None
            
            if session.is_expired:
                del self._sessions[session_id]
                logger.info(f"Session expired: {session_id[:16]}...")
                return None
            
            logger.debug(f"Session found: {session_id[:16]}... for {session.user_email}")
            return session
    
    def delete(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Returns True if session was deleted, False if not found.
        """
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"Session deleted: {session_id[:16]}...")
                return True
            return False
    
    def _maybe_cleanup(self) -> None:
        """Clean up expired sessions if interval has passed."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = now
        expired = [
            sid for sid, session in self._sessions.items()
            if session.is_expired
        ]
        
        for sid in expired:
            del self._sessions[sid]
        
        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")
    
    def count(self) -> int:
        """Get number of active sessions."""
        with self._lock:
            return len(self._sessions)
    
    def clear(self) -> None:
        """Clear all sessions (for logout all)."""
        with self._lock:
            count = len(self._sessions)
            self._sessions.clear()
            logger.info(f"Cleared {count} sessions")


# Global session store instance
session_store = SessionStore()
