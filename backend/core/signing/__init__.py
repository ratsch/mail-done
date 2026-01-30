"""
Asymmetric Request Signing Module

Ed25519-based request signing for secure client authentication.
Supports both static keypairs (laptop-admin) and ephemeral session keys (web-ui, v0-portal).

See: docs/SECURITY_DESIGN_REQUEST_SIGNING.md
"""

from backend.core.signing.keys import (
    generate_keypair,
    load_private_key,
    load_public_key,
    public_key_to_base64,
    base64_to_public_key,
)
from backend.core.signing.scopes import (
    Scope,
    check_scope,
    parse_scopes,
    SCOPE_ALL,
)
from backend.core.signing.verify import (
    verify_signature,
    create_canonical_request,
    VerificationResult,
)

__all__ = [
    # Keys
    "generate_keypair",
    "load_private_key",
    "load_public_key",
    "public_key_to_base64",
    "base64_to_public_key",
    # Scopes
    "Scope",
    "check_scope",
    "parse_scopes",
    "SCOPE_ALL",
    # Verification
    "verify_signature",
    "create_canonical_request",
    "VerificationResult",
]
