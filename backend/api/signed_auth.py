"""
Dual Authentication Middleware

Supports both legacy API_KEY authentication and new signed request authentication.
This allows gradual migration without breaking existing clients.

Authentication Methods (checked in order):
1. X-Signature header → Signed request authentication (new)
2. X-API-Key header → Legacy API key authentication (deprecated)

See: docs/SECURITY_DESIGN_REQUEST_SIGNING.md
"""

import logging
import os
import secrets
from dataclasses import dataclass
from functools import wraps
from typing import Callable, FrozenSet, List, Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import APIKeyHeader

from backend.core.signing.scopes import Scope, check_scope, parse_scopes
from backend.core.signing.verify import verify_signature, VerificationError
from backend.core.signing.registry import static_registry, load_static_clients
from backend.core.signing.ephemeral import ephemeral_registry
from backend.core.signing.keys import base64_to_public_key

logger = logging.getLogger(__name__)


# Legacy API key configuration (for backward compatibility)
API_KEY_NAME = "X-API-Key"
API_KEY = os.getenv("API_KEY")

# Signed auth header names
HEADER_CLIENT_ID = "X-Client-Id"
HEADER_TIMESTAMP = "X-Timestamp"
HEADER_NONCE = "X-Nonce"
HEADER_SIGNATURE = "X-Signature"

# API key header extractor
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


@dataclass
class AuthContext:
    """
    Authentication context for a verified request.
    
    Attributes:
        client_id: Identifier of the authenticated client
        auth_method: How the client authenticated ("signed" or "api_key")
        scopes: Permission scopes granted to this client
        user_email: Email of authenticated user (OAuth clients only)
        session_id: Session ID (ephemeral clients only)
    """
    client_id: str
    auth_method: str  # "signed" or "api_key"
    scopes: FrozenSet[Scope]
    user_email: Optional[str] = None
    session_id: Optional[str] = None
    
    def has_scope(self, required: str) -> bool:
        """Check if this context has the required scope."""
        return check_scope(self.scopes, required)


async def verify_signed_request(request: Request) -> AuthContext:
    """
    Verify a signed request.
    
    Checks:
    1. All required headers present
    2. Client ID is known (static or ephemeral session)
    3. Timestamp within ±5 minutes
    4. Nonce not reused
    5. Signature valid
    
    Args:
        request: FastAPI request
        
    Returns:
        AuthContext if verification succeeds
        
    Raises:
        HTTPException: If verification fails
    """
    # Extract headers
    client_id = request.headers.get(HEADER_CLIENT_ID)
    timestamp = request.headers.get(HEADER_TIMESTAMP)
    nonce = request.headers.get(HEADER_NONCE)
    signature = request.headers.get(HEADER_SIGNATURE)
    
    if not all([client_id, timestamp, nonce, signature]):
        missing = []
        if not client_id: missing.append(HEADER_CLIENT_ID)
        if not timestamp: missing.append(HEADER_TIMESTAMP)
        if not nonce: missing.append(HEADER_NONCE)
        if not signature: missing.append(HEADER_SIGNATURE)
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing required headers: {', '.join(missing)}",
        )
    
    # Get request body for signature verification
    body = await request.body()
    
    # Build path including query string
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"
    
    logger.debug(f"Verifying signature: client={client_id}, method={request.method}, path={path}")

    # Look up client
    public_key = None
    scopes = None
    user_email = None
    session_id = None
    auth_type = None
    nonce_checker = None
    
    # Try static registry first
    static_client = static_registry.get_client(client_id)
    if static_client:
        if not static_client.public_keys:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Client '{client_id}' has no registered public keys",
            )
        # For static clients, we try all registered keys (supports key rotation)
        # We'll verify against each key until one succeeds
        scopes = static_client.scopes
        auth_type = "static"
        
        # Try each public key for signature verification
        for pk in static_client.public_keys:
            result = verify_signature(
                public_key=pk,
                method=request.method,
                path=path,
                timestamp_str=timestamp,
                nonce=nonce,
                signature_b64=signature,
                body=body,
                check_nonce_reuse=None,  # No nonce tracking for static clients during key search
            )
            if result.success:
                public_key = pk
                break
        
        if not public_key:
            # None of the keys worked - log and reject
            logger.warning(
                f"Static client {client_id}: signature verification failed "
                f"against all {len(static_client.public_keys)} registered keys"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Signature verification failed",
            )
        # Static client verification succeeded - we're done
        # (nonce replay protection not implemented for static clients as they
        # are typically single-instance laptop scripts, not web browsers)

        # Set user_email if configured (enables access to user-scoped endpoints like review)
        user_email = static_client.user_email
    else:
        # Try ephemeral session registry
        # Client ID format for ephemeral: "webui-xxx" or "v0-portal-xxx"
        session = ephemeral_registry.get_session(client_id)
        if session:
            public_key = session.public_key
            scopes = session.scopes
            user_email = session.user_email
            session_id = session.session_id
            auth_type = "ephemeral"
            nonce_checker = lambda n: session.check_and_record_nonce(n)
            
            # Verify signature for ephemeral session (with nonce tracking)
            result = verify_signature(
                public_key=public_key,
                method=request.method,
                path=path,
                timestamp_str=timestamp,
                nonce=nonce,
                signature_b64=signature,
                body=body,
                check_nonce_reuse=nonce_checker,
            )
            
            if not result.success:
                logger.warning(
                    f"Signature verification failed for {client_id}: "
                    f"{result.error} - {result.error_message}"
                )
                
                # Map errors to appropriate status codes
                if result.error in (VerificationError.TIMESTAMP_TOO_OLD, VerificationError.TIMESTAMP_TOO_NEW):
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail=f"Request timestamp invalid: {result.error_message}",
                    )
                if result.error == VerificationError.NONCE_REUSED:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Nonce already used (possible replay attack)",
                    )
                
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=f"Signature verification failed: {result.error_message}",
                )
        else:
            # Unknown client
            logger.warning(f"Unknown client: {client_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Unknown client: {client_id}",
            )
    
    logger.debug(f"Authenticated client: {client_id} (method={auth_type})")
    
    return AuthContext(
        client_id=client_id,
        auth_method="signed",
        scopes=scopes,
        user_email=user_email,
        session_id=session_id,
    )


async def verify_api_key_legacy(api_key: str) -> AuthContext:
    """
    Verify legacy API key authentication.
    
    This is deprecated but maintained for backward compatibility
    during migration.
    
    Args:
        api_key: API key from header
        
    Returns:
        AuthContext with full admin scopes
        
    Raises:
        HTTPException: If API key is invalid
    """
    if not API_KEY:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY not configured on server",
        )
    
    if not secrets.compare_digest(api_key, API_KEY):
        logger.warning("Invalid API key attempt")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
    
    # Legacy API key gets full access (same as before)
    return AuthContext(
        client_id="legacy-api-key",
        auth_method="api_key",
        scopes=parse_scopes(["*"]),
    )


async def get_auth_context(
    request: Request,
    api_key: Optional[str] = Depends(api_key_header),
) -> AuthContext:
    """
    Get authentication context from request.
    
    Tries signed auth first, then falls back to legacy API key.
    
    This is the main dependency for protected routes.
    
    Example:
        @app.get("/api/emails")
        async def list_emails(auth: AuthContext = Depends(get_auth_context)):
            if not auth.has_scope("emails:read"):
                raise HTTPException(403, "Insufficient permissions")
            ...
    """
    # Check for signed request first
    if HEADER_SIGNATURE in request.headers:
        return await verify_signed_request(request)
    
    # Fall back to legacy API key
    if api_key:
        return await verify_api_key_legacy(api_key)
    
    # No authentication provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing authentication (provide X-Signature or X-API-Key)",
    )


def require_scope(scope: str):
    """
    Decorator/dependency to require a specific scope.
    
    Example:
        @app.get("/api/emails")
        async def list_emails(
            auth: AuthContext = Depends(require_scope("emails:read"))
        ):
            ...
    """
    async def checker(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
        if not auth.has_scope(scope):
            logger.warning(
                f"Scope check failed: {auth.client_id} lacks '{scope}'"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions: requires '{scope}'",
            )
        return auth
    
    return checker


def require_any_scope(*scopes: str):
    """
    Dependency to require any one of multiple scopes.
    
    Example:
        @app.get("/api/data")
        async def get_data(
            auth: AuthContext = Depends(require_any_scope("data:read", "admin:*"))
        ):
            ...
    """
    async def checker(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
        for scope in scopes:
            if auth.has_scope(scope):
                return auth
        
        logger.warning(
            f"Scope check failed: {auth.client_id} lacks any of {scopes}"
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Insufficient permissions: requires one of {scopes}",
        )
    
    return checker


# Initialize registries on module load
def init_auth_system():
    """
    Initialize the authentication system.
    
    Call this during FastAPI startup.
    """
    # Load static clients
    load_static_clients()
    
    # Start ephemeral session cleanup
    ephemeral_registry.start()
    
    logger.info("Signed auth system initialized")


def shutdown_auth_system():
    """
    Shutdown the authentication system.
    
    Call this during FastAPI shutdown.
    """
    # Stop ephemeral session cleanup
    ephemeral_registry.stop()
    
    # Save sessions for restart
    from backend.core.signing.persistence import save_sessions_on_shutdown
    save_sessions_on_shutdown()
    
    logger.info("Signed auth system shutdown complete")
