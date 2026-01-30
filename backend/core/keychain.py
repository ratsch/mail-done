"""
Secure Credential Storage using macOS Keychain

Provides unified interface for accessing credentials from:
1. macOS Keychain (primary, encrypted storage)
2. Environment variables (fallback for migration period)

Usage:
    from backend.core.keychain import get_credential
    
    password = get_credential("IMAP_PASSWORD_WORK")
    api_key = get_credential("OPENAI_API_KEY")

Security Features:
- OS-level AES-256 encryption
- Credentials isolated per user account
- No plaintext storage
- Automatic migration from .env
"""
import os
import logging
from typing import Optional

try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False
    logging.warning("keyring module not available - install with: pip install keyring")

logger = logging.getLogger(__name__)

SERVICE_NAME = "email-processor"

# Track warnings to avoid spam
_warned_keys = set()


def store_credential(key: str, value: str) -> bool:
    """
    Store credential in macOS Keychain.
    
    Args:
        key: Credential identifier (e.g. "OPENAI_API_KEY")
        value: Secret value to store
        
    Returns:
        True if stored successfully, False otherwise
    """
    if not KEYRING_AVAILABLE:
        logger.error("Cannot store credential - keyring not available")
        return False
    
    try:
        keyring.set_password(SERVICE_NAME, key, value)
        logger.info(f"Stored {key} in keychain")
        return True
    except Exception as e:
        logger.error(f"Failed to store {key} in keychain: {e}")
        return False


def get_credential(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get credential from macOS Keychain (with .env fallback).
    
    Priority:
    1. Try macOS Keychain first (most secure)
    2. Fall back to environment variable (for migration period)
    3. Return default if not found
    
    During migration period, if found in env but not keychain,
    automatically migrates to keychain for future use.
    
    Args:
        key: Credential identifier (e.g. "OPENAI_API_KEY")
        default: Default value if not found
        
    Returns:
        Credential value or default
    """
    # Try keychain first
    if KEYRING_AVAILABLE:
        try:
            value = keyring.get_password(SERVICE_NAME, key)
            if value is not None:
                return value
        except Exception as e:
            if key not in _warned_keys:
                logger.warning(f"Failed to retrieve {key} from keychain: {e}")
                _warned_keys.add(key)
    
    # Fall back to environment variable
    value = os.getenv(key)
    if value is not None:
        # Warn once about using env vars (should migrate to keychain)
        if key not in _warned_keys and KEYRING_AVAILABLE:
            logger.warning(
                f"{key} loaded from environment variable - "
                f"consider migrating to keychain for better security"
            )
            _warned_keys.add(key)
            
            # Auto-migrate: store in keychain for next time
            try:
                store_credential(key, value)
                logger.info(f"Auto-migrated {key} to keychain")
            except Exception as e:
                logger.debug(f"Auto-migration failed for {key}: {e}")
        
        return value
    
    return default


def delete_credential(key: str) -> bool:
    """
    Delete credential from keychain.
    
    Args:
        key: Credential identifier
        
    Returns:
        True if deleted, False otherwise
    """
    if not KEYRING_AVAILABLE:
        return False
    
    try:
        keyring.delete_password(SERVICE_NAME, key)
        logger.info(f"Deleted {key} from keychain")
        return True
    except keyring.errors.PasswordDeleteError:
        # Key didn't exist
        return False
    except Exception as e:
        logger.error(f"Failed to delete {key}: {e}")
        return False


def list_credentials() -> list[str]:
    """
    List all credentials stored in keychain.
    
    Note: Not all keyring backends support listing.
    
    Returns:
        List of credential keys
    """
    # Unfortunately, most keyring backends don't support listing
    # Return empty list
    logger.warning("Listing credentials not supported by keyring backend")
    return []


def test_keychain_access() -> bool:
    """
    Test that keychain is accessible.
    
    Returns:
        True if keychain can be accessed, False otherwise
    """
    if not KEYRING_AVAILABLE:
        logger.error("Keychain not available - install keyring module")
        return False
    
    test_key = "_test_access"
    test_value = "test123"
    
    try:
        # Try to store
        keyring.set_password(SERVICE_NAME, test_key, test_value)
        
        # Try to retrieve
        retrieved = keyring.get_password(SERVICE_NAME, test_key)
        
        # Clean up
        try:
            keyring.delete_password(SERVICE_NAME, test_key)
        except:
            pass
        
        success = (retrieved == test_value)
        if success:
            logger.info("Keychain access test: PASSED")
        else:
            logger.error("Keychain access test: FAILED (value mismatch)")
        
        return success
    
    except Exception as e:
        logger.error(f"Keychain access test: FAILED ({e})")
        return False


# Module initialization
if KEYRING_AVAILABLE:
    logger.info("macOS Keychain support enabled")
else:
    logger.warning(
        "macOS Keychain support disabled - credentials will only be read from environment. "
        "Install keyring module for secure storage: pip install keyring"
    )
