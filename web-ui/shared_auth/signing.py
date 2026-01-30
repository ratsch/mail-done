"""
Ed25519 Request Signing

Generates ephemeral keypairs and signs HTTP requests for secure
communication with the backend API.

Canonical Request Format:
    {method}\n{path}\n{timestamp}\n{nonce}\n{body_hash}
"""

import base64
import hashlib
import secrets
import time
from typing import Dict, Tuple

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)


def generate_keypair() -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """
    Generate a new Ed25519 keypair.
    
    Returns:
        Tuple of (private_key, public_key)
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def public_key_to_base64(public_key: Ed25519PublicKey) -> str:
    """
    Serialize public key to base64 string.
    
    Args:
        public_key: Ed25519 public key
        
    Returns:
        Base64-encoded public key (44 characters)
    """
    raw_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw_bytes).decode("ascii")


def _hash_body(body: bytes) -> str:
    """Compute SHA-256 hash of request body."""
    if not body:
        return "empty"
    return hashlib.sha256(body).hexdigest()


def sign_request(
    private_key: Ed25519PrivateKey,
    client_id: str,
    method: str,
    path: str,
    body: bytes = b"",
) -> Dict[str, str]:
    """
    Sign an HTTP request and return headers.
    
    Args:
        private_key: Ed25519 private key
        client_id: Client/session identifier
        method: HTTP method (GET, POST, etc.)
        path: Request path including query string
        body: Request body bytes
        
    Returns:
        Dict with signing headers (X-Client-Id, X-Timestamp, X-Nonce, X-Signature)
    """
    timestamp = int(time.time())
    nonce = secrets.token_hex(16)  # 32 hex characters
    body_hash = _hash_body(body)
    
    # Create canonical request
    canonical = f"{method.upper()}\n{path}\n{timestamp}\n{nonce}\n{body_hash}"
    
    # Sign with Ed25519
    signature = private_key.sign(canonical.encode("utf-8"))
    signature_b64 = base64.b64encode(signature).decode("ascii")
    
    return {
        "X-Client-Id": client_id,
        "X-Timestamp": str(timestamp),
        "X-Nonce": nonce,
        "X-Signature": signature_b64,
    }
