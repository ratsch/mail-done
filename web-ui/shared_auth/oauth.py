"""
Google OAuth Utilities

Handles Google OAuth flow for Web-UI authentication.
"""

import logging
import os
import secrets
from typing import Dict, Optional
from urllib.parse import urlencode

import httpx
from google.oauth2 import id_token as google_id_token
from google.auth.transport import requests as google_requests

logger = logging.getLogger(__name__)

# Google OAuth endpoints
GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Environment configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")


class OAuthError(Exception):
    """Raised when OAuth operation fails."""
    pass


def get_google_auth_url(
    redirect_uri: str,
    state: Optional[str] = None,
    client_id: Optional[str] = None,
) -> str:
    """
    Generate Google OAuth authorization URL.
    
    Args:
        redirect_uri: URL to redirect after authentication
        state: CSRF protection state (auto-generated if not provided)
        client_id: Google Client ID (defaults to env var)
        
    Returns:
        Full Google OAuth URL for redirect
    """
    client_id = client_id or GOOGLE_CLIENT_ID
    
    if not client_id:
        raise OAuthError("GOOGLE_CLIENT_ID not configured")
    
    if state is None:
        state = secrets.token_urlsafe(32)
    
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    
    return f"{GOOGLE_AUTH_URL}?{urlencode(params)}"


async def exchange_code_for_token(
    code: str,
    redirect_uri: str,
    client_id: Optional[str] = None,
    client_secret: Optional[str] = None,
) -> Dict[str, str]:
    """
    Exchange authorization code for tokens.
    
    Args:
        code: Authorization code from Google callback
        redirect_uri: Same redirect_uri used in auth request
        client_id: Google Client ID
        client_secret: Google Client Secret
        
    Returns:
        Dict with id_token, access_token, etc.
        
    Raises:
        OAuthError: If token exchange fails
    """
    client_id = client_id or GOOGLE_CLIENT_ID
    client_secret = client_secret or GOOGLE_CLIENT_SECRET
    
    if not client_id or not client_secret:
        raise OAuthError("Google OAuth credentials not configured")
    
    payload = {
        "code": code,
        "client_id": client_id,
        "client_secret": client_secret,
        "redirect_uri": redirect_uri,
        "grant_type": "authorization_code",
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                GOOGLE_TOKEN_URL,
                data=payload,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            
            if response.status_code != 200:
                error_data = response.json() if response.content else {}
                raise OAuthError(
                    f"Token exchange failed: {error_data.get('error_description', response.text)}"
                )
            
            return response.json()
            
    except httpx.RequestError as e:
        raise OAuthError(f"Failed to connect to Google: {str(e)}")


def verify_google_id_token(
    token: str,
    client_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Verify a Google ID token and extract claims.
    
    Args:
        token: Google ID token (JWT)
        client_id: Expected audience (defaults to env var)
        
    Returns:
        Dict with token claims (email, sub, name, etc.)
        
    Raises:
        OAuthError: If token is invalid
    """
    client_id = client_id or GOOGLE_CLIENT_ID
    
    if not client_id:
        raise OAuthError("GOOGLE_CLIENT_ID not configured")
    
    try:
        # Verify the token
        idinfo = google_id_token.verify_oauth2_token(
            token,
            google_requests.Request(),
            client_id,
        )
        
        # Check issuer
        if idinfo["iss"] not in ["accounts.google.com", "https://accounts.google.com"]:
            raise OAuthError("Invalid token issuer")
        
        return idinfo
        
    except ValueError as e:
        raise OAuthError(f"Invalid ID token: {str(e)}")


def generate_state() -> str:
    """Generate a secure state parameter for CSRF protection."""
    return secrets.token_urlsafe(32)
