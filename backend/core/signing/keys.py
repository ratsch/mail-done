"""
Ed25519 Key Management

Provides key generation, loading, and serialization for Ed25519 keypairs.
Uses the cryptography library for all cryptographic operations.
"""

import base64
import os
from pathlib import Path
from typing import Tuple, Optional

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
    
    Example:
        >>> private_key, public_key = generate_keypair()
        >>> pub_b64 = public_key_to_base64(public_key)
    """
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def public_key_to_base64(public_key: Ed25519PublicKey) -> str:
    """
    Serialize a public key to base64-encoded string.
    
    Args:
        public_key: Ed25519 public key object
        
    Returns:
        Base64-encoded public key string (44 characters)
    """
    raw_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    return base64.b64encode(raw_bytes).decode("ascii")


def base64_to_public_key(b64_key: str) -> Ed25519PublicKey:
    """
    Deserialize a base64-encoded public key string.
    
    Args:
        b64_key: Base64-encoded public key (44 characters)
        
    Returns:
        Ed25519 public key object
        
    Raises:
        ValueError: If the key is invalid or wrong length
    """
    try:
        raw_bytes = base64.b64decode(b64_key)
        if len(raw_bytes) != 32:
            raise ValueError(f"Invalid public key length: {len(raw_bytes)} bytes (expected 32)")
        return Ed25519PublicKey.from_public_bytes(raw_bytes)
    except Exception as e:
        raise ValueError(f"Invalid public key: {e}") from e


def private_key_to_bytes(private_key: Ed25519PrivateKey) -> bytes:
    """
    Serialize a private key to raw bytes (for secure storage).
    
    WARNING: Private key bytes are sensitive! Handle with care.
    
    Args:
        private_key: Ed25519 private key object
        
    Returns:
        32-byte raw private key
    """
    return private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )


def bytes_to_private_key(key_bytes: bytes) -> Ed25519PrivateKey:
    """
    Deserialize raw bytes to a private key.
    
    Args:
        key_bytes: 32-byte raw private key
        
    Returns:
        Ed25519 private key object
        
    Raises:
        ValueError: If the key is invalid
    """
    if len(key_bytes) != 32:
        raise ValueError(f"Invalid private key length: {len(key_bytes)} bytes (expected 32)")
    return Ed25519PrivateKey.from_private_bytes(key_bytes)


def save_keypair(
    private_key: Ed25519PrivateKey,
    public_key: Ed25519PublicKey,
    directory: Path,
    name: str = "signing",
) -> Tuple[Path, Path]:
    """
    Save a keypair to files.
    
    Creates two files:
    - {name}.key (private key, PEM format)
    - {name}.pub (public key, base64)
    
    Args:
        private_key: Ed25519 private key
        public_key: Ed25519 public key
        directory: Directory to save keys in
        name: Base name for key files
        
    Returns:
        Tuple of (private_key_path, public_key_path)
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    
    private_path = directory / f"{name}.key"
    public_path = directory / f"{name}.pub"
    
    # Save private key in PEM format (more standard for key files)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    private_path.write_bytes(private_pem)
    os.chmod(private_path, 0o600)  # Owner read/write only
    
    # Save public key as base64 (easy to copy/paste)
    public_b64 = public_key_to_base64(public_key)
    public_path.write_text(public_b64 + "\n")
    
    return private_path, public_path


def load_private_key(path: Path) -> Ed25519PrivateKey:
    """
    Load a private key from a PEM file.
    
    Args:
        path: Path to private key file
        
    Returns:
        Ed25519 private key object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If key is invalid
    """
    path = Path(path)
    pem_data = path.read_bytes()
    
    try:
        private_key = serialization.load_pem_private_key(pem_data, password=None)
        if not isinstance(private_key, Ed25519PrivateKey):
            raise ValueError(f"Not an Ed25519 key: {type(private_key)}")
        return private_key
    except Exception as e:
        raise ValueError(f"Failed to load private key from {path}: {e}") from e


def load_public_key(path: Path) -> Ed25519PublicKey:
    """
    Load a public key from a base64 file.
    
    Args:
        path: Path to public key file (base64 format)
        
    Returns:
        Ed25519 public key object
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If key is invalid
    """
    path = Path(path)
    b64_key = path.read_text().strip()
    return base64_to_public_key(b64_key)
