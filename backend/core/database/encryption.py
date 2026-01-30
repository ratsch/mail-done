"""
Database Field Encryption Module

Provides SQLAlchemy TypeDecorators for encrypting sensitive database fields.
Uses Fernet symmetric encryption (AES-128 in CBC mode with HMAC-SHA256).

KEY ROTATION SUPPORT:
- DB_ENCRYPTION_KEY: Primary key used for all NEW encryptions
- DB_ENCRYPTION_KEY_OLD: Comma-separated list of previous keys for decryption
  Example: DB_ENCRYPTION_KEY_OLD=oldkey1,oldkey2,oldkey3
  
Key rotation process:
1. Generate new key: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'
2. Move current DB_ENCRYPTION_KEY to DB_ENCRYPTION_KEY_OLD (prepend to list)
3. Set new key as DB_ENCRYPTION_KEY
4. Run: python -m backend.core.database.encryption --rotate-keys
5. After successful rotation, old keys can be removed from DB_ENCRYPTION_KEY_OLD

Encrypted fields (14 total):
- Email body content (markdown, text, HTML)
- AI-generated summaries and reasoning
- User notes and tags
- Reply draft bodies and reasoning
- Sender notes
- Action items
- Category-specific data (names, dates, locations)
- Relevance/prestige assessment reasons

Unencrypted fields (for search/filtering):
- Subjects (needed for subject search)
- Email addresses (needed for email filtering)
- Names (needed for name search)
- Embeddings (needed for vector similarity search)
- Quantitative scores (needed for filtering/sorting)
"""
import os
import json
import logging
from typing import Any, Optional, List
from sqlalchemy import TypeDecorator, Text
from cryptography.fernet import Fernet, InvalidToken, MultiFernet

logger = logging.getLogger(__name__)

# Determine if we're in production
APP_ENV = os.getenv('APP_ENV', 'development')
IS_PRODUCTION = APP_ENV in ('production', 'prod', 'staging')

# ============================================================================
# Key Management with Rotation Support
# ============================================================================

# Primary key for encryption (required)
ENCRYPTION_KEY = os.getenv('DB_ENCRYPTION_KEY')

# Old keys for decryption during rotation (optional, comma-separated)
OLD_ENCRYPTION_KEYS = os.getenv('DB_ENCRYPTION_KEY_OLD', '')

# Initialize ciphers
primary_cipher: Optional[Fernet] = None
multi_cipher: Optional[MultiFernet] = None
old_ciphers: List[Fernet] = []


def _initialize_ciphers():
    """
    Initialize encryption ciphers with key rotation support.
    
    Uses MultiFernet to support decryption with old keys while
    encrypting only with the primary (newest) key.
    """
    global primary_cipher, multi_cipher, old_ciphers, ENCRYPTION_KEY, OLD_ENCRYPTION_KEYS
    
    if not ENCRYPTION_KEY:
        error_msg = (
            "DB_ENCRYPTION_KEY not set in environment. "
            "Encryption is MANDATORY for all environments. "
            "Generate a key with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
        )
        logger.critical(error_msg)
        raise ValueError("DB_ENCRYPTION_KEY is required - encryption is mandatory for confidential data")
    
    try:
        # Initialize primary cipher (used for all new encryptions)
        primary_cipher = Fernet(ENCRYPTION_KEY.encode('utf-8'))
        
        # Build list of all ciphers: primary first, then old keys
        all_ciphers = [primary_cipher]
        old_ciphers = []
        
        # Parse and add old keys for decryption support
        if OLD_ENCRYPTION_KEYS:
            old_keys = [k.strip() for k in OLD_ENCRYPTION_KEYS.split(',') if k.strip()]
            for i, old_key in enumerate(old_keys):
                try:
                    old_cipher = Fernet(old_key.encode('utf-8'))
                    old_ciphers.append(old_cipher)
                    all_ciphers.append(old_cipher)
                    logger.info(f"Loaded old encryption key #{i+1} for rotation support")
                except Exception as e:
                    logger.error(f"Invalid old encryption key #{i+1}: {e}")
                    raise ValueError(f"Invalid old encryption key at position {i+1}")
        
        # MultiFernet tries keys in order: encrypts with first, decrypts with any
        multi_cipher = MultiFernet(all_ciphers)
        
        key_count = len(all_ciphers)
        if key_count > 1:
            logger.info(f"Database encryption initialized with {key_count} keys (1 primary + {key_count-1} old)")
        else:
            logger.info("Database encryption initialized successfully")
            
    except ValueError:
        raise
    except Exception as e:
        error_msg = f"Failed to initialize encryption cipher: {e}"
        logger.critical(error_msg)
        raise RuntimeError(error_msg)


# Initialize on module load
_initialize_ciphers()

# Backwards compatibility alias
cipher = primary_cipher


class EncryptedText(TypeDecorator):
    """
    Encrypted text column type.
    
    Encrypts text data before storing in database.
    Decrypts when reading from database.
    
    Usage:
        class Email(Base):
            body_text = Column(EncryptedText)
    
    Storage format: Base64-encoded encrypted bytes (stored as TEXT)
    """
    
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value: Optional[str], dialect) -> Optional[str]:
        """
        Encrypt value before storing in database.
        
        Always uses the PRIMARY key for encryption.
        
        Args:
            value: Plain text to encrypt
            dialect: SQLAlchemy dialect
            
        Returns:
            Encrypted text (Base64 encoded) or None
            
        Raises:
            RuntimeError: If encryption is not configured in production
        """
        if value is None:
            return None
        
        if primary_cipher is None:
            error_msg = "Encryption cipher not initialized - cannot encrypt data"
            logger.critical(error_msg)
            raise RuntimeError(f"{error_msg}. Set DB_ENCRYPTION_KEY environment variable.")
        
        try:
            # Encrypt with PRIMARY key (Fernet returns bytes)
            encrypted = primary_cipher.encrypt(value.encode('utf-8'))
            return encrypted.decode('utf-8')
        except Exception as e:
            logger.error(f"Encryption failed for value of length {len(value)}: {e}")
            raise RuntimeError(f"Failed to encrypt data: {e}")

    
    def process_result_value(self, value: Optional[str], dialect) -> Optional[str]:
        """
        Decrypt value when reading from database.
        
        Uses MultiFernet to try all available keys (primary + old keys)
        for seamless key rotation support.
        
        Args:
            value: Encrypted text (Base64 encoded) or unencrypted text (during migration)
            dialect: SQLAlchemy dialect
            
        Returns:
            Decrypted plain text or None
            
        Raises:
            RuntimeError: If encryption is not configured in production
        """
        if value is None:
            return None
        
        if multi_cipher is None:
            error_msg = "Encryption cipher not initialized - cannot decrypt data"
            logger.critical(f"{error_msg}. Set DB_ENCRYPTION_KEY environment variable.")
            raise RuntimeError(error_msg)
        
        # Check if data is actually encrypted (starts with Fernet prefix)
        # During migration, some data may still be unencrypted
        if not value.startswith('gAAAAA'):
            # Not encrypted - this is legacy unencrypted data
            # Return as-is - will be auto-encrypted on next write (lazy migration)
            return value
        
        try:
            # MultiFernet tries all keys in order until one works
            decrypted = multi_cipher.decrypt(value.encode('utf-8'))
            return decrypted.decode('utf-8')
        except InvalidToken as e:
            # None of the keys worked - data encrypted with unknown key
            logger.error(f"Failed to decrypt value - no matching key found: {e}")
            logger.error(f"Value preview (first 50 chars): {value[:50]}...")
            logger.critical("DECRYPTION FAILURE - data encrypted with unknown key")
            return None
        except Exception as e:
            logger.error(f"Decryption failed with unexpected error: {e}")
            logger.critical("DECRYPTION FAILURE - data may be lost or inaccessible")
            return None


class EncryptedJSON(TypeDecorator):
    """
    Encrypted JSON column type.
    
    Encrypts JSON data before storing in database.
    Decrypts and parses when reading from database.
    
    Usage:
        class EmailMetadata(Base):
            user_tags = Column(EncryptedJSON, default=list)
            category_specific_data = Column(EncryptedJSON, default=dict)
    
    Storage format: Encrypted JSON string (stored as TEXT)
    """
    
    impl = Text
    cache_ok = True
    
    def process_bind_param(self, value: Optional[Any], dialect) -> Optional[str]:
        """
        Serialize to JSON and encrypt before storing.
        
        Always uses the PRIMARY key for encryption.
        
        Args:
            value: Python object (dict, list, etc.)
            dialect: SQLAlchemy dialect
            
        Returns:
            Encrypted JSON string or None
            
        Raises:
            RuntimeError: If encryption is not configured in production
        """
        if value is None:
            return None
        
        if primary_cipher is None:
            error_msg = "Encryption cipher not initialized - cannot encrypt JSON data"
            logger.critical(error_msg)
            raise RuntimeError(f"{error_msg}. Set DB_ENCRYPTION_KEY environment variable.")
        
        try:
            # Serialize to JSON, then encrypt with PRIMARY key
            json_str = json.dumps(value)
            encrypted = primary_cipher.encrypt(json_str.encode('utf-8'))
            return encrypted.decode('utf-8')
        except TypeError as e:
            logger.error(f"JSON serialization failed - value is not JSON-serializable: {e}")
            raise ValueError(f"Cannot encrypt non-JSON-serializable value: {e}")
        except Exception as e:
            logger.error(f"JSON encryption failed: {e}")
            raise RuntimeError(f"Failed to encrypt JSON data: {e}")
    
    def process_result_value(self, value: Optional[str], dialect) -> Optional[Any]:
        """
        Decrypt and parse JSON when reading from database.
        
        Uses MultiFernet to try all available keys (primary + old keys)
        for seamless key rotation support.
        
        Args:
            value: Encrypted JSON string or unencrypted JSON string (during migration)
            dialect: SQLAlchemy dialect
            
        Returns:
            Parsed Python object (dict, list, etc.) or None
            
        Raises:
            RuntimeError: If encryption is not configured in production
        """
        if value is None:
            return None
        
        if multi_cipher is None:
            error_msg = "Encryption cipher not initialized - cannot decrypt JSON data"
            logger.critical(f"{error_msg}. Set DB_ENCRYPTION_KEY environment variable.")
            raise RuntimeError(error_msg)
        
        # Check if data is actually encrypted (starts with Fernet prefix)
        # During migration, some data may still be unencrypted JSON strings
        if not value.startswith('gAAAAA'):
            # Not encrypted - this is legacy unencrypted JSON data
            # Parse and return - will be auto-encrypted on next write (lazy migration)
            try:
                return json.loads(value)
            except json.JSONDecodeError as e:
                logger.warning(f"Unencrypted JSON field failed to parse (len={len(value[:100])}): {e}")
                # Value is neither encrypted nor valid JSON - return None
                return None
        
        try:
            # MultiFernet tries all keys in order until one works
            decrypted = multi_cipher.decrypt(value.encode('utf-8'))
            json_str = decrypted.decode('utf-8')
            return json.loads(json_str)
        except InvalidToken as e:
            # None of the keys worked - data encrypted with unknown key
            logger.error(f"Failed to decrypt JSON - no matching key found: {e}")
            logger.error(f"Value preview (first 50 chars): {value[:50]}...")
            logger.critical("DECRYPTION FAILURE - JSON encrypted with unknown key")
            return None
        except json.JSONDecodeError as e:
            # Decryption succeeded but JSON parsing failed
            logger.error(f"Decryption succeeded but JSON parsing failed: {e}")
            logger.error(f"This indicates corrupted data in the database")
            logger.critical("DATA CORRUPTION - encrypted JSON is not valid JSON")
            return None
        except Exception as e:
            logger.error(f"JSON decryption failed with unexpected error: {e}")
            logger.critical("DECRYPTION FAILURE - JSON data may be lost or inaccessible")
            return None


def is_encryption_enabled() -> bool:
    """
    Check if encryption is properly configured.
    
    Returns:
        True if encryption key is set and cipher is initialized
    """
    return primary_cipher is not None


def has_old_keys() -> bool:
    """
    Check if old encryption keys are configured for rotation.
    
    Returns:
        True if there are old keys available for decryption
    """
    return len(old_ciphers) > 0


def get_key_count() -> int:
    """
    Get the total number of encryption keys loaded.
    
    Returns:
        Number of keys (1 primary + N old keys)
    """
    return 1 + len(old_ciphers) if primary_cipher else 0


def reinitialize_cipher():
    """
    Reinitialize the encryption cipher from environment variable.
    Useful when environment variables are loaded after module import.
    
    Returns:
        True if cipher was successfully initialized, False otherwise
    """
    global ENCRYPTION_KEY, OLD_ENCRYPTION_KEYS, cipher
    
    # Reload keys from environment
    ENCRYPTION_KEY = os.getenv('DB_ENCRYPTION_KEY')
    OLD_ENCRYPTION_KEYS = os.getenv('DB_ENCRYPTION_KEY_OLD', '')
    
    if not ENCRYPTION_KEY:
        logger.warning("DB_ENCRYPTION_KEY not found - cannot initialize encryption")
        return False
    
    try:
        _initialize_ciphers()
        cipher = primary_cipher  # Update backwards compat alias
        return True
    except Exception as e:
        logger.error(f"Failed to reinitialize encryption cipher: {e}")
        return False


def generate_encryption_key() -> str:
    """
    Generate a new Fernet encryption key.
    
    Returns:
        Base64-encoded encryption key (suitable for DB_ENCRYPTION_KEY env var)
    """
    return Fernet.generate_key().decode('utf-8')


def rotate_encrypted_value(encrypted_value: str) -> Optional[str]:
    """
    Re-encrypt a value with the primary key if it was encrypted with an old key.
    
    This is used during key rotation to migrate data to the new key.
    
    Args:
        encrypted_value: Fernet-encrypted string
        
    Returns:
        Re-encrypted value (or original if already using primary key), or None on error
    """
    if not encrypted_value or not encrypted_value.startswith('gAAAAA'):
        return encrypted_value  # Not encrypted, return as-is
    
    if primary_cipher is None or multi_cipher is None:
        logger.error("Cannot rotate: encryption not initialized")
        return None
    
    try:
        # Decrypt with any available key
        decrypted = multi_cipher.decrypt(encrypted_value.encode('utf-8'))
        
        # Re-encrypt with primary key
        re_encrypted = primary_cipher.encrypt(decrypted)
        return re_encrypted.decode('utf-8')
    except InvalidToken:
        logger.error(f"Cannot rotate value - no matching key found")
        return None
    except Exception as e:
        logger.error(f"Failed to rotate encrypted value: {e}")
        return None


def needs_rotation(encrypted_value: str) -> bool:
    """
    Check if an encrypted value was encrypted with an old key.
    
    Args:
        encrypted_value: Fernet-encrypted string
        
    Returns:
        True if value needs re-encryption with primary key
    """
    if not encrypted_value or not encrypted_value.startswith('gAAAAA'):
        return False  # Not encrypted or unencrypted
    
    if primary_cipher is None:
        return False
    
    try:
        # Try to decrypt with primary key only
        primary_cipher.decrypt(encrypted_value.encode('utf-8'))
        return False  # Primary key works, no rotation needed
    except InvalidToken:
        # Primary key didn't work - needs rotation
        return True
    except Exception:
        return False


# Export for easy importing
__all__ = [
    'EncryptedText',
    'EncryptedJSON',
    'is_encryption_enabled',
    'has_old_keys',
    'get_key_count',
    'generate_encryption_key',
    'rotate_encrypted_value',
    'needs_rotation',
]


# ============================================================================
# CLI for key rotation
# ============================================================================

# Define tables and their encrypted columns (used by CLI)
ENCRYPTED_TABLES = {
    'emails': ['body_markdown', 'body_text', 'body_html'],
    'email_metadata': [
        'ai_reasoning', 'ai_summary', 'relevance_reason', 'prestige_reason',
        'user_notes', 'user_tags', 'category_specific_data', 'action_items'
    ],
    'senders': ['notes'],
    'reply_drafts': ['body', 'reasoning'],
    'application_reviews': ['comment'],
    'application_private_notes': ['notes'],
}


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(
        description="Database encryption key management and rotation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
FULL KEY ROTATION WORKFLOW:
===========================

1. GENERATE new key:
   python -m backend.core.database.encryption --generate-key

2. UPDATE environment (backup first!):
   - Set new key as DB_ENCRYPTION_KEY
   - Move current key to DB_ENCRYPTION_KEY_OLD

3. MIGRATE data (dry-run first):
   python -m backend.core.database.encryption --rotate-keys --dry-run
   python -m backend.core.database.encryption --rotate-keys

4. VERIFY all data uses new key:
   python -m backend.core.database.encryption --verify

5. REMOVE old key from DB_ENCRYPTION_KEY_OLD
   (only after verification passes!)

SECURITY NOTE:
If a key was compromised, complete steps 1-5 as quickly as possible.
Until step 5, the old key can still decrypt data.
"""
    )
    parser.add_argument('--generate-key', action='store_true', 
                        help='Generate a new encryption key')
    parser.add_argument('--rotate-keys', action='store_true',
                        help='Re-encrypt all data with the primary key')
    parser.add_argument('--verify', action='store_true',
                        help='Verify ALL data can be decrypted with PRIMARY key only')
    parser.add_argument('--check-status', action='store_true',
                        help='Check encryption status and key configuration')
    parser.add_argument('--dry-run', action='store_true',
                        help='For --rotate-keys: show what would be done without making changes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='Number of records to process per batch (default: 100)')
    
    args = parser.parse_args()
    
    if args.generate_key:
        new_key = generate_encryption_key()
        print("=" * 60)
        print("NEW ENCRYPTION KEY GENERATED")
        print("=" * 60)
        print(f"\n{new_key}\n")
        print("=" * 60)
        print("\nNEXT STEPS:")
        print("1. Back up your current DB_ENCRYPTION_KEY")
        print("2. Set the new key as DB_ENCRYPTION_KEY")
        print("3. Set your old key as DB_ENCRYPTION_KEY_OLD")
        print("4. Run: python -m backend.core.database.encryption --rotate-keys --dry-run")
        print("5. Run: python -m backend.core.database.encryption --rotate-keys")
        print("6. Run: python -m backend.core.database.encryption --verify")
        print("7. Remove old key from DB_ENCRYPTION_KEY_OLD")
        sys.exit(0)
    
    if args.check_status:
        print("=" * 60)
        print("ENCRYPTION STATUS")
        print("=" * 60)
        print(f"Encryption enabled: {is_encryption_enabled()}")
        print(f"Primary key loaded: {primary_cipher is not None}")
        print(f"Old keys loaded: {len(old_ciphers)}")
        print(f"Total keys: {get_key_count()}")
        
        if has_old_keys():
            print("\n⚠️  WARNING: Old keys are configured!")
            print("   This means data may still exist encrypted with old keys.")
            print("   Run --rotate-keys to migrate, then --verify before removing old keys.")
        else:
            print("\n✅ No old keys configured. System using single primary key.")
        sys.exit(0)
    
    if args.verify:
        # Verify ALL encrypted data can be decrypted with PRIMARY key only
        from backend.core.database import get_db
        from sqlalchemy import text
        
        print("=" * 60)
        print("VERIFYING ALL DATA USES PRIMARY KEY")
        print("=" * 60)
        print("\nThis checks that EVERY encrypted value can be decrypted")
        print("using ONLY the primary key (DB_ENCRYPTION_KEY).")
        print("If this passes, you can safely remove old keys.\n")
        
        if has_old_keys():
            print(f"⚠️  Note: {len(old_ciphers)} old key(s) are currently loaded.")
            print("   Verification will check if they're still needed.\n")
        
        db = next(get_db())
        
        total_checked = 0
        total_needs_rotation = 0
        total_unencrypted = 0
        total_errors = 0
        failed_records = []
        
        for table_name, columns in ENCRYPTED_TABLES.items():
            print(f"Checking {table_name}...")
            
            # Get all records
            try:
                count_result = db.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0
            except Exception as e:
                print(f"  ⚠️  Table not found or error: {e}")
                continue
            
            if count_result == 0:
                print(f"  Empty table, skipping")
                continue
            
            table_needs_rotation = 0
            table_unencrypted = 0
            
            # Process in batches
            offset = 0
            while offset < count_result:
                batch_query = text(f"""
                    SELECT id, {', '.join(columns)} 
                    FROM {table_name} 
                    ORDER BY id
                    LIMIT :limit OFFSET :offset
                """)
                batch = db.execute(batch_query, {"limit": args.batch_size, "offset": offset}).fetchall()
                
                for row in batch:
                    row_id = row[0]
                    for i, col_name in enumerate(columns, start=1):
                        value = row[i]
                        if value is None:
                            continue
                        
                        total_checked += 1
                        value_str = str(value)
                        
                        # Check if it's encrypted
                        if not value_str.startswith('gAAAAA'):
                            total_unencrypted += 1
                            table_unencrypted += 1
                            continue
                        
                        # Check if it needs rotation (can't be decrypted by primary key)
                        if needs_rotation(value_str):
                            total_needs_rotation += 1
                            table_needs_rotation += 1
                            if len(failed_records) < 10:  # Keep first 10 for reporting
                                failed_records.append(f"{table_name}.{row_id}.{col_name}")
                
                offset += args.batch_size
                print(f"  Checked {min(offset, count_result)}/{count_result}...", end='\r')
            
            status = "✅" if table_needs_rotation == 0 else "❌"
            print(f"  {status} Checked {count_result} records, {table_needs_rotation} need rotation, {table_unencrypted} unencrypted")
        
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print(f"Total values checked: {total_checked}")
        print(f"Encrypted with primary key: {total_checked - total_needs_rotation - total_unencrypted}")
        print(f"Still need rotation: {total_needs_rotation}")
        print(f"Unencrypted (legacy): {total_unencrypted}")
        
        if total_needs_rotation > 0:
            print(f"\n❌ VERIFICATION FAILED")
            print(f"   {total_needs_rotation} values still encrypted with old key(s).")
            print(f"   Run --rotate-keys to migrate them before removing old keys.")
            if failed_records:
                print(f"\n   Sample records needing rotation:")
                for rec in failed_records[:5]:
                    print(f"     - {rec}")
            sys.exit(1)
        elif total_unencrypted > 0:
            print(f"\n⚠️  PARTIAL SUCCESS")
            print(f"   No values need key rotation, but {total_unencrypted} values are unencrypted.")
            print(f"   These will be encrypted on next write (lazy migration).")
            print(f"   You can safely remove old keys from DB_ENCRYPTION_KEY_OLD.")
            sys.exit(0)
        else:
            print(f"\n✅ VERIFICATION PASSED")
            print(f"   All encrypted data uses the primary key.")
            print(f"   You can safely remove old keys from DB_ENCRYPTION_KEY_OLD.")
            sys.exit(0)
    
    if args.rotate_keys:
        # Import here to avoid circular imports
        import time
        from datetime import datetime
        from backend.core.database import get_db
        from sqlalchemy import text
        
        print("=" * 60)
        print("ENCRYPTION KEY ROTATION (CONCURRENT-SAFE)")
        print("=" * 60)
        print(f"Primary key: loaded")
        print(f"Old keys: {len(old_ciphers)}")
        print(f"Dry run: {args.dry_run}")
        print(f"Batch size: {args.batch_size}")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        print("ℹ️  CONCURRENT OPERATION: This script is safe to run while")
        print("   the production system is active. New writes use the new key,")
        print("   and this script only updates records still using old keys.")
        print()
        
        if not has_old_keys():
            print("❌ No old keys configured in DB_ENCRYPTION_KEY_OLD.")
            print("\nTo rotate keys:")
            print("1. Generate new key: --generate-key")
            print("2. Set new key as DB_ENCRYPTION_KEY")
            print("3. Set old key as DB_ENCRYPTION_KEY_OLD")
            print("4. Run this command again")
            sys.exit(1)
        
        db = next(get_db())
        
        total_rotated = 0
        total_skipped = 0
        total_errors = 0
        total_concurrent_updates = 0
        start_time = time.time()
        
        for table_name, columns in ENCRYPTED_TABLES.items():
            print(f"\nProcessing {table_name}...")
            
            # Get total count
            try:
                count_result = db.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0
            except Exception as e:
                print(f"  ⚠️  Table not found or error: {e}")
                continue
            
            print(f"  Total records: {count_result}")
            
            if count_result == 0:
                continue
            
            pk_col = 'id'
            last_id = None  # Use cursor-based pagination for efficiency
            table_rotated = 0
            table_skipped = 0
            table_errors = 0
            table_concurrent = 0
            processed = 0
            
            while True:
                # Use cursor-based pagination (more efficient for large tables)
                if last_id is None:
                    batch_query = text(f"""
                        SELECT {pk_col}, {', '.join(columns)} 
                        FROM {table_name} 
                        ORDER BY {pk_col}
                        LIMIT :limit
                    """)
                    batch = db.execute(batch_query, {"limit": args.batch_size}).fetchall()
                else:
                    batch_query = text(f"""
                        SELECT {pk_col}, {', '.join(columns)} 
                        FROM {table_name} 
                        WHERE {pk_col} > :last_id
                        ORDER BY {pk_col}
                        LIMIT :limit
                    """)
                    batch = db.execute(batch_query, {"limit": args.batch_size, "last_id": str(last_id)}).fetchall()
                
                if not batch:
                    break
                
                for row in batch:
                    row_id = row[0]
                    last_id = row_id
                    processed += 1
                    updates = {}
                    old_values = {}  # Track old values for optimistic locking
                    
                    for i, col_name in enumerate(columns, start=1):
                        value = row[i]
                        if value and needs_rotation(str(value)):
                            rotated = rotate_encrypted_value(str(value))
                            if rotated:
                                updates[col_name] = rotated
                                old_values[col_name] = str(value)
                            else:
                                table_errors += 1
                    
                    if updates:
                        if args.dry_run:
                            if processed <= 5:  # Only show first few in dry run
                                print(f"    Would rotate {len(updates)} columns in {table_name}.{row_id}")
                        else:
                            # CONCURRENT-SAFE: Use optimistic locking
                            # Only update if the value hasn't changed (another process might have updated it)
                            for col_name, new_value in updates.items():
                                old_value = old_values[col_name]
                                update_query = text(f"""
                                    UPDATE {table_name} 
                                    SET {col_name} = :new_value 
                                    WHERE {pk_col} = :id AND {col_name} = :old_value
                                """)
                                result = db.execute(update_query, {
                                    "new_value": new_value,
                                    "old_value": old_value,
                                    "id": row_id
                                })
                                if result.rowcount == 0:
                                    # Value was modified by another process - that's OK!
                                    # The other process already wrote with the new key
                                    table_concurrent += 1
                        table_rotated += 1
                    else:
                        table_skipped += 1
                
                # Commit every batch (not in dry run)
                if not args.dry_run:
                    db.commit()
                
                # Progress with ETA
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = (count_result - processed) / rate if rate > 0 else 0
                eta_str = f"ETA: {int(remaining/60)}m {int(remaining%60)}s" if rate > 0 else ""
                
                print(f"    Processed {processed}/{count_result} ({rate:.0f}/sec) {eta_str}          ", end='\r')
            
            print(f"\n  Rotated: {table_rotated}, Skipped: {table_skipped}, Concurrent: {table_concurrent}, Errors: {table_errors}")
            total_rotated += table_rotated
            total_skipped += table_skipped
            total_errors += table_errors
            total_concurrent_updates += table_concurrent
        
        elapsed_total = time.time() - start_time
        print("\n" + "=" * 60)
        print(f"ROTATION {'SIMULATION ' if args.dry_run else ''}COMPLETE")
        print("=" * 60)
        print(f"Records with rotated columns: {total_rotated}")
        print(f"Records already on primary key: {total_skipped}")
        print(f"Concurrent updates (OK): {total_concurrent_updates}")
        print(f"Errors: {total_errors}")
        print(f"Duration: {int(elapsed_total/60)}m {int(elapsed_total%60)}s")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if total_concurrent_updates > 0:
            print(f"\nℹ️  {total_concurrent_updates} records were updated by the production system")
            print("   during migration. This is normal and expected behavior.")
        
        if args.dry_run:
            print("\n⚠️  This was a DRY RUN. No changes were made.")
            print("   Run without --dry-run to apply changes.")
        elif total_errors > 0:
            print(f"\n❌ Rotation completed with {total_errors} errors.")
            print("   Check logs for details. Some data may not have been migrated.")
        elif total_rotated > 0:
            print("\n✅ Rotation complete!")
            print("\nNEXT STEPS:")
            print("1. Run: python -m backend.core.database.encryption --verify")
            print("2. If verification passes, remove old keys from DB_ENCRYPTION_KEY_OLD")
        else:
            print("\n✅ No rotation needed - all data already uses primary key.")
            print("   Run --verify to confirm, then remove old keys.")
        
        sys.exit(0 if total_errors == 0 else 1)
    
    # No command specified
    parser.print_help()

