"""
Backend OAuth Handshake

Exchanges a Google ID token and ephemeral public key for a backend
session ID with granted scopes.
"""

import logging
from typing import Dict, Any

import httpx

from .signing import public_key_to_base64

logger = logging.getLogger(__name__)


class HandshakeError(Exception):
    """Raised when OAuth handshake fails."""
    def __init__(self, message: str, status_code: int = None, details: str = None):
        super().__init__(message)
        self.status_code = status_code
        self.details = details


async def do_handshake(
    backend_url: str,
    id_token: str,
    public_key,  # Ed25519PublicKey
    client_type: str = "web-ui",
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Perform OAuth handshake with backend.
    
    Sends the Google ID token and ephemeral public key to the backend,
    which verifies the token and creates an ephemeral session.
    
    Args:
        backend_url: Backend API base URL
        id_token: Google ID token from OAuth flow
        public_key: Ed25519 public key (ephemeral)
        client_type: Client type identifier (web-ui, v0-portal)
        timeout: Request timeout in seconds
        
    Returns:
        Dict with:
            - session_id: Backend session identifier
            - expires_in: Seconds until session expires
            - scopes: List of granted permission scopes
            - user_email: Verified email address
            
    Raises:
        HandshakeError: If handshake fails
    """
    # Convert public key to base64
    public_key_b64 = public_key_to_base64(public_key)
    
    # Build handshake request
    handshake_url = f"{backend_url.rstrip('/')}/auth/oauth-handshake"
    
    payload = {
        "id_token": id_token,
        "public_key": public_key_b64,
        "client_type": client_type,
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(handshake_url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(
                    f"Handshake successful: session={data['session_id'][:16]}... "
                    f"user={data.get('user_email', 'unknown')}"
                )
                return data
            
            # Handle specific error codes
            if response.status_code == 401:
                raise HandshakeError(
                    "Invalid or expired Google token",
                    status_code=401,
                    details=response.text,
                )
            elif response.status_code == 403:
                raise HandshakeError(
                    "Access denied - user not authorized for this client",
                    status_code=403,
                    details=response.text,
                )
            else:
                raise HandshakeError(
                    f"Handshake failed with status {response.status_code}",
                    status_code=response.status_code,
                    details=response.text,
                )
                
    except httpx.RequestError as e:
        raise HandshakeError(
            f"Failed to connect to backend: {str(e)}",
            details=str(e),
        )


async def do_handshake_with_jwt(
    backend_url: str,
    jwt_token: str,
    public_key,  # Ed25519PublicKey
    client_type: str = "web-ui",
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Perform OAuth handshake with backend using a JWT token.
    
    This is the preferred method when the backend handles Google OAuth.
    The client doesn't need Google credentials.
    
    Args:
        backend_url: Backend API base URL
        jwt_token: Backend-issued JWT token (from backend OAuth flow)
        public_key: Ed25519 public key (ephemeral)
        client_type: Client type identifier (web-ui, v0-portal)
        timeout: Request timeout in seconds
        
    Returns:
        Dict with:
            - session_id: Backend session identifier
            - expires_in: Seconds until session expires
            - scopes: List of granted permission scopes
            - user_email: Verified email address
            
    Raises:
        HandshakeError: If handshake fails
    """
    # Convert public key to base64
    public_key_b64 = public_key_to_base64(public_key)
    
    # Build handshake request
    handshake_url = f"{backend_url.rstrip('/')}/auth/oauth-handshake"
    
    payload = {
        "jwt_token": jwt_token,  # Use JWT instead of id_token
        "public_key": public_key_b64,
        "client_type": client_type,
    }
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(handshake_url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(
                    f"JWT handshake successful: session={data['session_id'][:16]}... "
                    f"user={data.get('user_email', 'unknown')}"
                )
                return data
            
            # Handle specific error codes
            if response.status_code == 401:
                raise HandshakeError(
                    "Invalid or expired JWT token",
                    status_code=401,
                    details=response.text,
                )
            elif response.status_code == 403:
                raise HandshakeError(
                    "Access denied - user not authorized for this client",
                    status_code=403,
                    details=response.text,
                )
            else:
                raise HandshakeError(
                    f"Handshake failed with status {response.status_code}",
                    status_code=response.status_code,
                    details=response.text,
                )
                
    except httpx.RequestError as e:
        raise HandshakeError(
            f"Failed to connect to backend: {str(e)}",
            details=str(e),
        )


def do_handshake_sync(
    backend_url: str,
    id_token: str,
    public_key,  # Ed25519PublicKey
    client_type: str = "web-ui",
    timeout: float = 10.0,
) -> Dict[str, Any]:
    """
    Synchronous version of do_handshake.
    
    Use this when not in an async context.
    """
    import asyncio
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(
        do_handshake(backend_url, id_token, public_key, client_type, timeout)
    )
