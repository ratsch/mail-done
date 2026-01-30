"""
API Key Authentication for FastAPI

Supports dual authentication:
1. X-Signature header → Ed25519 signed request (new, preferred)
2. X-API-Key header → Legacy API key (deprecated but still works)
"""
import os
import secrets
from typing import Optional
from fastapi import HTTPException, Security, Request, status, Depends
from fastapi.security import APIKeyHeader

# API Key header name
API_KEY_NAME = "X-API-Key"

# Get API key from environment
API_KEY = os.getenv("API_KEY")

# Create API key header security scheme
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


async def verify_api_key(
    request: Request,
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """
    Verify authentication from request headers.
    
    Supports dual authentication:
    1. X-Signature header → Ed25519 signed request (new)
    2. X-API-Key header → Legacy API key (deprecated)
    
    Args:
        request: FastAPI request object
        api_key: API key from X-API-Key header
        
    Returns:
        The verified API key or client_id for signed requests
        
    Raises:
        HTTPException: If authentication fails
    """
    # Check for signed request first (new method)
    signature = request.headers.get("X-Signature")
    if signature:
        try:
            from backend.api.signed_auth import get_auth_context
            auth_context = await get_auth_context(request)
            # Return client_id for compatibility with existing code
            return auth_context.client_id
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Signature verification error: {str(e)}"
            )
    
    # Fall back to legacy API key authentication
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY not configured on server"
        )
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API Key. Include 'X-API-Key' header."
        )
    
    # Use constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(api_key, API_KEY):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API Key"
        )
    
    return api_key

