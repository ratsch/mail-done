"""
Signature Verification

Verifies Ed25519 signatures on incoming requests.
Implements the canonical request format, timestamp validation, and nonce replay prevention.

Canonical Request Format:
    {method}\n{path}\n{timestamp}\n{nonce}\n{body_hash}

Where:
    - method: HTTP method (GET, POST, etc.)
    - path: Request path including query string (e.g., /api/emails?limit=10)
    - timestamp: Unix timestamp in seconds
    - nonce: 32-character hex string (unique per request)
    - body_hash: SHA-256 hex digest of request body, or "empty" if no body
"""

import base64
import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.exceptions import InvalidSignature

import logging

logger = logging.getLogger(__name__)


# Timestamp tolerance: ±5 minutes
TIMESTAMP_TOLERANCE_SECONDS = 300

# Expected nonce length (32 hex characters = 16 bytes)
NONCE_LENGTH = 32


class VerificationError(Enum):
    """Enumeration of possible verification failures."""
    MISSING_HEADERS = "missing_headers"
    INVALID_TIMESTAMP_FORMAT = "invalid_timestamp_format"
    TIMESTAMP_TOO_OLD = "timestamp_too_old"
    TIMESTAMP_TOO_NEW = "timestamp_too_new"
    INVALID_NONCE_FORMAT = "invalid_nonce_format"
    NONCE_REUSED = "nonce_reused"
    INVALID_SIGNATURE_FORMAT = "invalid_signature_format"
    SIGNATURE_VERIFICATION_FAILED = "signature_verification_failed"
    UNKNOWN_CLIENT = "unknown_client"


@dataclass
class VerificationResult:
    """
    Result of signature verification.
    
    Attributes:
        success: Whether verification succeeded
        error: Error type if verification failed
        error_message: Human-readable error message
        client_id: Client ID from headers (if present)
        timestamp: Parsed timestamp (if valid)
        nonce: Parsed nonce (if valid)
    """
    success: bool
    error: Optional[VerificationError] = None
    error_message: Optional[str] = None
    client_id: Optional[str] = None
    timestamp: Optional[int] = None
    nonce: Optional[str] = None
    
    @classmethod
    def ok(cls, client_id: str, timestamp: int, nonce: str) -> "VerificationResult":
        """Create a successful result."""
        return cls(
            success=True,
            client_id=client_id,
            timestamp=timestamp,
            nonce=nonce,
        )
    
    @classmethod
    def fail(cls, error: VerificationError, message: str, client_id: Optional[str] = None) -> "VerificationResult":
        """Create a failed result."""
        return cls(
            success=False,
            error=error,
            error_message=message,
            client_id=client_id,
        )


def hash_body(body: bytes) -> str:
    """
    Compute SHA-256 hash of request body.
    
    Args:
        body: Request body bytes
        
    Returns:
        Hex-encoded SHA-256 hash, or "empty" if body is empty
    """
    if not body:
        return "empty"
    return hashlib.sha256(body).hexdigest()


def create_canonical_request(
    method: str,
    path: str,
    timestamp: int,
    nonce: str,
    body: bytes = b"",
) -> str:
    """
    Create the canonical request string for signing/verification.
    
    This is the exact format that MUST be used by all clients.
    
    Args:
        method: HTTP method (uppercase)
        path: Request path including query string
        timestamp: Unix timestamp in seconds
        nonce: 32-character hex nonce
        body: Request body bytes
        
    Returns:
        Canonical request string
        
    Example:
        >>> create_canonical_request("GET", "/api/emails?limit=10", 1703001234, "a1b2c3...", b"")
        'GET\n/api/emails?limit=10\n1703001234\na1b2c3...\nempty'
    """
    body_hash = hash_body(body)
    return f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{body_hash}"


def validate_timestamp(timestamp: int) -> Optional[VerificationError]:
    """
    Validate that timestamp is within acceptable range.
    
    Args:
        timestamp: Unix timestamp in seconds
        
    Returns:
        VerificationError if invalid, None if valid
    """
    now = int(time.time())
    delta = now - timestamp
    
    if delta > TIMESTAMP_TOLERANCE_SECONDS:
        return VerificationError.TIMESTAMP_TOO_OLD
    if delta < -TIMESTAMP_TOLERANCE_SECONDS:
        return VerificationError.TIMESTAMP_TOO_NEW
    
    return None


def validate_nonce(nonce: str) -> Optional[VerificationError]:
    """
    Validate nonce format.
    
    Args:
        nonce: Nonce string
        
    Returns:
        VerificationError if invalid, None if valid
    """
    if len(nonce) != NONCE_LENGTH:
        return VerificationError.INVALID_NONCE_FORMAT
    
    # Must be valid hex
    try:
        int(nonce, 16)
    except ValueError:
        return VerificationError.INVALID_NONCE_FORMAT
    
    return None


def verify_signature(
    public_key: Ed25519PublicKey,
    method: str,
    path: str,
    timestamp_str: str,
    nonce: str,
    signature_b64: str,
    body: bytes = b"",
    check_nonce_reuse: Optional[callable] = None,
) -> VerificationResult:
    """
    Verify a signed request.
    
    Performs the following checks in order:
    1. Parse and validate timestamp format
    2. Check timestamp is within ±5 minutes
    3. Validate nonce format
    4. Check nonce hasn't been used (if checker provided)
    5. Verify Ed25519 signature
    
    Args:
        public_key: Client's Ed25519 public key
        method: HTTP method
        path: Request path including query string
        timestamp_str: Timestamp as string (from header)
        nonce: Nonce string (from header)
        signature_b64: Base64-encoded signature (from header)
        body: Request body bytes
        check_nonce_reuse: Optional callback to check/record nonce usage
                          Should return True if nonce was already used
        
    Returns:
        VerificationResult with success status and details
        
    Example:
        >>> result = verify_signature(
        ...     public_key=client_public_key,
        ...     method="GET",
        ...     path="/api/emails",
        ...     timestamp_str="1703001234",
        ...     nonce="a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
        ...     signature_b64="...",
        ... )
        >>> if result.success:
        ...     # Proceed with request
        ... else:
        ...     # Reject with result.error_message
    """
    # 1. Parse timestamp
    try:
        timestamp = int(timestamp_str)
    except (ValueError, TypeError):
        return VerificationResult.fail(
            VerificationError.INVALID_TIMESTAMP_FORMAT,
            f"Invalid timestamp format: '{timestamp_str}'"
        )
    
    # 2. Validate timestamp is within range
    ts_error = validate_timestamp(timestamp)
    if ts_error == VerificationError.TIMESTAMP_TOO_OLD:
        return VerificationResult.fail(
            ts_error,
            f"Timestamp too old: {timestamp} (now: {int(time.time())})"
        )
    if ts_error == VerificationError.TIMESTAMP_TOO_NEW:
        return VerificationResult.fail(
            ts_error,
            f"Timestamp too far in future: {timestamp} (now: {int(time.time())})"
        )
    
    # 3. Validate nonce format
    nonce_error = validate_nonce(nonce)
    if nonce_error:
        return VerificationResult.fail(
            nonce_error,
            f"Invalid nonce format: '{nonce}' (expected {NONCE_LENGTH} hex characters)"
        )
    
    # 4. Check nonce reuse (if checker provided)
    if check_nonce_reuse is not None:
        if check_nonce_reuse(nonce):
            return VerificationResult.fail(
                VerificationError.NONCE_REUSED,
                f"Nonce already used: '{nonce}'"
            )
    
    # 5. Decode signature
    try:
        signature_bytes = base64.b64decode(signature_b64)
    except Exception as e:
        return VerificationResult.fail(
            VerificationError.INVALID_SIGNATURE_FORMAT,
            f"Invalid base64 signature: {e}"
        )
    
    # 6. Create canonical request and verify signature
    canonical = create_canonical_request(method, path, timestamp, nonce, body)
    
    try:
        public_key.verify(signature_bytes, canonical.encode("utf-8"))
    except InvalidSignature:
        return VerificationResult.fail(
            VerificationError.SIGNATURE_VERIFICATION_FAILED,
            "Signature verification failed"
        )
    except Exception as e:
        return VerificationResult.fail(
            VerificationError.SIGNATURE_VERIFICATION_FAILED,
            f"Signature verification error: {e}"
        )
    
    # All checks passed
    return VerificationResult.ok(
        client_id="",  # Caller should set this
        timestamp=timestamp,
        nonce=nonce,
    )


def sign_request(
    private_key,  # Ed25519PrivateKey
    method: str,
    path: str,
    body: bytes = b"",
) -> dict:
    """
    Sign a request (for testing and client implementation reference).
    
    Args:
        private_key: Ed25519 private key
        method: HTTP method
        path: Request path
        body: Request body
        
    Returns:
        Dict with headers to add to request:
        {
            "X-Timestamp": "...",
            "X-Nonce": "...",
            "X-Signature": "...",
        }
    """
    import secrets
    
    timestamp = int(time.time())
    nonce = secrets.token_hex(16)  # 32 hex characters
    
    canonical = create_canonical_request(method, path, timestamp, nonce, body)
    signature = private_key.sign(canonical.encode("utf-8"))
    signature_b64 = base64.b64encode(signature).decode("ascii")
    
    return {
        "X-Timestamp": str(timestamp),
        "X-Nonce": nonce,
        "X-Signature": signature_b64,
    }
