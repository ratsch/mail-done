"""
Shared Authentication Library for Web-UI

Provides Google OAuth authentication and Ed25519 request signing
for secure communication with the backend API.

Components:
- signing: Ed25519 keypair generation and request signing
- session: In-memory session storage (ephemeral keys)
- handshake: Backend OAuth handshake client
- oauth: Google OAuth utilities
"""

from .signing import (
    generate_keypair,
    sign_request,
    public_key_to_base64,
)
from .session import SessionStore, session_store
from .handshake import do_handshake
from .oauth import (
    get_google_auth_url,
    exchange_code_for_token,
    verify_google_id_token,
)

__all__ = [
    # Signing
    "generate_keypair",
    "sign_request",
    "public_key_to_base64",
    # Session
    "SessionStore",
    "session_store",
    # Handshake
    "do_handshake",
    # OAuth
    "get_google_auth_url",
    "exchange_code_for_token",
    "verify_google_id_token",
]
