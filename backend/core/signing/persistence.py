"""
Session Persistence

Saves and restores ephemeral sessions on graceful shutdown/startup.
This prevents losing active sessions when the server restarts.

Note: Nonces are NOT persisted (they're short-lived and restart clears replay window anyway).
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional

from backend.core.signing.ephemeral import (
    EphemeralSession,
    EphemeralSessionRegistry,
    ephemeral_registry,
)

logger = logging.getLogger(__name__)


# Default persistence file location
DEFAULT_PERSISTENCE_PATH = Path("/tmp/mail-done-sessions.json")


class SessionPersistence:
    """
    Handles saving and loading ephemeral sessions.
    
    Uses a simple JSON file for persistence.
    Designed for graceful restarts, not long-term storage.
    """
    
    def __init__(self, path: Optional[Path] = None):
        """
        Initialize persistence handler.
        
        Args:
            path: Path to persistence file (default: /tmp/mail-done-sessions.json)
        """
        self.path = path or DEFAULT_PERSISTENCE_PATH
        # Allow override via environment variable
        env_path = os.getenv("SESSION_PERSISTENCE_PATH")
        if env_path:
            self.path = Path(env_path)
    
    def save(self, registry: Optional[EphemeralSessionRegistry] = None) -> int:
        """
        Save active sessions to file.
        
        Args:
            registry: Registry to save (default: global ephemeral_registry)
            
        Returns:
            Number of sessions saved
        """
        registry = registry or ephemeral_registry
        sessions = registry.get_all_sessions()
        
        if not sessions:
            # Remove persistence file if no sessions
            if self.path.exists():
                self.path.unlink()
            logger.info("No sessions to persist")
            return 0
        
        # Serialize sessions
        data = {
            "version": 1,
            "sessions": [s.to_dict() for s in sessions],
        }
        
        try:
            # Write atomically (write to temp, then rename)
            temp_path = self.path.with_suffix(".tmp")
            
            # Create file with secure permissions from the start
            # (prevents race condition where file exists briefly with default perms)
            fd = os.open(temp_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
            
            temp_path.rename(self.path)
            
            logger.info(f"Saved {len(sessions)} sessions to {self.path}")
            return len(sessions)
        
        except Exception as e:
            logger.error(f"Failed to save sessions: {e}")
            return 0
    
    def load(self, registry: Optional[EphemeralSessionRegistry] = None) -> int:
        """
        Load sessions from file.
        
        Args:
            registry: Registry to load into (default: global ephemeral_registry)
            
        Returns:
            Number of sessions loaded
        """
        registry = registry or ephemeral_registry
        
        if not self.path.exists():
            logger.info(f"No persistence file found at {self.path}")
            return 0
        
        try:
            with open(self.path) as f:
                data = json.load(f)
            
            version = data.get("version", 0)
            if version != 1:
                logger.warning(f"Unknown persistence format version: {version}")
                return 0
            
            sessions = []
            for session_data in data.get("sessions", []):
                try:
                    session = EphemeralSession.from_dict(session_data)
                    sessions.append(session)
                except Exception as e:
                    logger.warning(f"Failed to deserialize session: {e}")
            
            count = registry.restore_sessions(sessions)
            
            # Remove persistence file after successful load
            self.path.unlink()
            
            return count
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in persistence file: {e}")
            return 0
        except Exception as e:
            logger.error(f"Failed to load sessions: {e}")
            return 0
    
    def clear(self) -> None:
        """Remove persistence file."""
        if self.path.exists():
            self.path.unlink()
            logger.info(f"Cleared persistence file {self.path}")


# Global persistence handler
session_persistence = SessionPersistence()


def save_sessions_on_shutdown() -> None:
    """
    Save sessions during graceful shutdown.
    
    Call this from FastAPI shutdown event handler.
    """
    count = session_persistence.save()
    if count > 0:
        logger.info(f"Saved {count} sessions on shutdown")


def load_sessions_on_startup() -> None:
    """
    Load sessions during startup.
    
    Call this from FastAPI startup event handler.
    """
    count = session_persistence.load()
    if count > 0:
        logger.info(f"Restored {count} sessions on startup")
