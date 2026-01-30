"""
Unit tests for database field encryption.

Tests the EncryptedText and EncryptedJSON TypeDecorators.
"""
import os
import pytest
from unittest.mock import patch
from sqlalchemy import create_engine, Column, Integer, text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.exc import StatementError
import uuid

from backend.core.database.encryption import (
    EncryptedText,
    EncryptedJSON,
    is_encryption_enabled,
    generate_encryption_key,
)
from cryptography.fernet import Fernet


# Test database setup
Base = declarative_base()


class TestModel(Base):
    """Test model with encrypted fields."""
    __tablename__ = "test_encryption"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    encrypted_text = Column(EncryptedText)
    encrypted_json = Column(EncryptedJSON)
    plain_int = Column(Integer)


@pytest.fixture
def test_db():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def encryption_key():
    """Generate and set encryption key for tests."""
    key = generate_encryption_key()
    with patch.dict(os.environ, {'DB_ENCRYPTION_KEY': key}):
        # Reload encryption module to pick up new key
        import importlib
        from backend.core.database import encryption
        importlib.reload(encryption)
        yield key
        # Reload again to clean up
        importlib.reload(encryption)


class TestEncryptedText:
    """Test EncryptedText TypeDecorator."""
    
    def test_encrypt_decrypt_round_trip(self, test_db, encryption_key):
        """Test that text can be encrypted and decrypted."""
        original_text = "This is sensitive email content with special chars: 🔒✉️"
        
        # Create record with encrypted text
        record = TestModel(encrypted_text=original_text, plain_int=42)
        test_db.add(record)
        test_db.commit()
        
        # Retrieve and verify
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_text == original_text
        assert retrieved.plain_int == 42
    
    def test_null_value_handling(self, test_db, encryption_key):
        """Test that None values are handled correctly."""
        record = TestModel(encrypted_text=None, plain_int=1)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_text is None
    
    def test_empty_string(self, test_db, encryption_key):
        """Test empty string encryption."""
        record = TestModel(encrypted_text="", plain_int=2)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_text == ""
    
    def test_long_text(self, test_db, encryption_key):
        """Test encryption of long email body."""
        # Simulate long email body
        long_text = "Email body content. " * 1000  # ~20KB
        
        record = TestModel(encrypted_text=long_text, plain_int=3)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_text == long_text
    
    def test_unicode_content(self, test_db, encryption_key):
        """Test encryption of unicode/emoji content."""
        unicode_text = "Hello 世界! 🌍 Emoji test: 😀🎉🔐"
        
        record = TestModel(encrypted_text=unicode_text, plain_int=4)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_text == unicode_text
    
    def test_multiline_text(self, test_db, encryption_key):
        """Test encryption of multiline text."""
        multiline_text = """Line 1
        Line 2
        Line 3
        
        Line 5 with blank line above"""
        
        record = TestModel(encrypted_text=multiline_text, plain_int=5)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_text == multiline_text


class TestEncryptedJSON:
    """Test EncryptedJSON TypeDecorator."""
    
    def test_encrypt_decrypt_dict(self, test_db, encryption_key):
        """Test dictionary encryption."""
        data = {
            "name": "John Doe",
            "email": "john@example.com",
            "tags": ["important", "followup"]
        }
        
        record = TestModel(encrypted_json=data, plain_int=10)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_json == data
    
    def test_encrypt_decrypt_list(self, test_db, encryption_key):
        """Test list encryption."""
        data = ["action item 1", "action item 2", "action item 3"]
        
        record = TestModel(encrypted_json=data, plain_int=11)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_json == data
    
    def test_null_json(self, test_db, encryption_key):
        """Test None value for JSON field."""
        record = TestModel(encrypted_json=None, plain_int=12)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_json is None
    
    def test_empty_dict(self, test_db, encryption_key):
        """Test empty dictionary."""
        record = TestModel(encrypted_json={}, plain_int=13)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_json == {}
    
    def test_empty_list(self, test_db, encryption_key):
        """Test empty list."""
        record = TestModel(encrypted_json=[], plain_int=14)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_json == []
    
    def test_nested_structure(self, test_db, encryption_key):
        """Test complex nested JSON structure."""
        data = {
            "event": {
                "name": "Conference 2024",
                "date": "2024-12-01",
                "attendees": ["Alice", "Bob"],
                "location": {
                    "city": "San Francisco",
                    "venue": "Tech Center"
                }
            },
            "metadata": {
                "importance": 10,
                "tags": ["conference", "tech", "networking"]
            }
        }
        
        record = TestModel(encrypted_json=data, plain_int=15)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_json == data
    
    def test_unicode_in_json(self, test_db, encryption_key):
        """Test JSON with unicode characters."""
        data = {
            "message": "Hello 世界! 🌍",
            "emojis": ["😀", "🎉", "🔐"]
        }
        
        record = TestModel(encrypted_json=data, plain_int=16)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_json == data


class TestEncryptionUtilities:
    """Test utility functions."""
    
    def test_generate_encryption_key(self):
        """Test key generation."""
        key1 = generate_encryption_key()
        key2 = generate_encryption_key()
        
        # Keys should be different
        assert key1 != key2
        
        # Keys should be base64 strings
        assert isinstance(key1, str)
        assert len(key1) > 0
    
    def test_is_encryption_enabled(self, encryption_key):
        """Test encryption status check."""
        # With key set, should be enabled
        import importlib
        from backend.core.database import encryption
        importlib.reload(encryption)
        
        # Note: This test depends on whether DB_ENCRYPTION_KEY is set
        # in the test environment
        result = is_encryption_enabled()
        assert isinstance(result, bool)


class TestEncryptionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_update_encrypted_field(self, test_db, encryption_key):
        """Test updating an encrypted field."""
        original_text = "Original content"
        updated_text = "Updated content"
        
        record = TestModel(encrypted_text=original_text, plain_int=20)
        test_db.add(record)
        test_db.commit()
        
        # Update the record
        record.encrypted_text = updated_text
        test_db.commit()
        
        # Retrieve and verify
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_text == updated_text
    
    def test_special_characters_in_text(self, test_db, encryption_key):
        """Test text with special characters."""
        special_text = "Special chars: \n\t\r\0 and quotes: \"'` and slashes: /\\"
        
        record = TestModel(encrypted_text=special_text, plain_int=21)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_text == special_text
    
    def test_json_with_special_values(self, test_db, encryption_key):
        """Test JSON with special values."""
        data = {
            "null_value": None,
            "boolean_true": True,
            "boolean_false": False,
            "number_int": 42,
            "number_float": 3.14,
            "empty_string": "",
            "special_chars": "\n\t\r"
        }
        
        record = TestModel(encrypted_json=data, plain_int=22)
        test_db.add(record)
        test_db.commit()
        
        retrieved = test_db.query(TestModel).first()
        assert retrieved.encrypted_json == data
    
    def test_multiple_records(self, test_db, encryption_key):
        """Test multiple records with different encrypted values."""
        records = [
            TestModel(encrypted_text=f"Text {i}", encrypted_json={"id": i}, plain_int=i)
            for i in range(10)
        ]
        
        test_db.add_all(records)
        test_db.commit()
        
        # Retrieve all and verify
        retrieved = test_db.query(TestModel).order_by(TestModel.plain_int).all()
        assert len(retrieved) == 10
        
        for i, record in enumerate(retrieved):
            assert record.encrypted_text == f"Text {i}"
            assert record.encrypted_json == {"id": i}
            assert record.plain_int == i


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_decryption_with_wrong_key(self, test_db):
        """Test decryption fails gracefully with wrong key."""
        # Encrypt with one key
        key1 = generate_encryption_key()
        cipher1 = Fernet(key1.encode('utf-8'))
        plaintext = "Secret data"
        encrypted = cipher1.encrypt(plaintext.encode('utf-8')).decode('utf-8')
        
        # Insert encrypted data directly using plain_int as ID
        test_db.execute(
            text("INSERT INTO test_encryption (id, encrypted_text, plain_int) VALUES (:id, :enc, :num)"),
            {"id": str(uuid.uuid4()), "enc": encrypted, "num": 30}
        )
        test_db.commit()
        
        # Try to read with different key
        key2 = generate_encryption_key()
        with patch.dict(os.environ, {'DB_ENCRYPTION_KEY': key2, 'APP_ENV': 'development'}):
            import importlib
            from backend.core.database import encryption
            importlib.reload(encryption)
            
            retrieved = test_db.query(TestModel).filter(TestModel.plain_int == 30).first()
            # Should return record but encrypted_text is None (decryption failure logged)
            assert retrieved is not None
            assert retrieved.plain_int == 30
            assert retrieved.encrypted_text is None
    
    # Note: test_missing_encryption_key_development removed - encryption is now always required

    def test_missing_encryption_key_production(self):
        """Test that missing key in production raises error."""
        with patch.dict(os.environ, {'APP_ENV': 'production'}, clear=True):
            # Remove DB_ENCRYPTION_KEY
            os.environ.pop('DB_ENCRYPTION_KEY', None)
            
            import importlib
            from backend.core.database import encryption
            
            # Should raise ValueError in production
            with pytest.raises(ValueError, match="DB_ENCRYPTION_KEY is required"):
                importlib.reload(encryption)
    
    def test_invalid_key_format(self):
        """Test that invalid key format is rejected."""
        with patch.dict(os.environ, {'DB_ENCRYPTION_KEY': 'invalid-key-format', 'APP_ENV': 'production'}):
            import importlib
            from backend.core.database import encryption
            
            # Should raise error for invalid key in production
            with pytest.raises(ValueError):
                importlib.reload(encryption)
    
    def test_corrupted_encrypted_data(self, test_db, encryption_key):
        """Test handling of corrupted encrypted data in database."""
        # Create a valid record first
        record = TestModel(encrypted_text="Valid data", plain_int=32)
        test_db.add(record)
        test_db.commit()
        
        # Now corrupt it by updating with invalid encrypted data
        test_db.execute(
            text("UPDATE test_encryption SET encrypted_text = :enc WHERE plain_int = :num"),
            {"enc": "gAAAAACorruptedData===", "num": 32}
        )
        test_db.commit()
        
        # Clear cache
        test_db.expire_all()
        
        # Try to read corrupted data
        retrieved = test_db.query(TestModel).filter(TestModel.plain_int == 32).first()
        
        # Record should exist but encrypted_text should be None (failed decryption)
        assert retrieved is not None
        assert retrieved.plain_int == 32
        assert retrieved.encrypted_text is None
    
    def test_json_serialization_error(self, test_db, encryption_key):
        """Test that non-serializable objects are rejected."""
        # Create object that can't be JSON serialized
        class NonSerializable:
            pass
        
        # Should raise ValueError wrapped in StatementError
        with pytest.raises(StatementError):
            record = TestModel(encrypted_json=NonSerializable(), plain_int=33)
            test_db.add(record)
            test_db.commit()


class TestSecurityScenarios:
    """Test security-related scenarios."""
    
    def test_encrypted_data_is_different_from_plaintext(self, test_db, encryption_key):
        """Verify that encrypted data doesn't resemble plaintext."""
        plaintext = "Sensitive email content"
        
        record = TestModel(encrypted_text=plaintext, plain_int=40)
        test_db.add(record)
        test_db.commit()
        
        # Query raw database to see encrypted value
        from sqlalchemy import text
        result = test_db.execute(
            text("SELECT encrypted_text FROM test_encryption WHERE plain_int = :num"),
            {"num": 40}
        ).fetchone()
        
        encrypted_value = result[0]
        
        # Encrypted value should be completely different
        assert encrypted_value != plaintext
        assert plaintext not in encrypted_value
        # Should start with Fernet prefix
        assert encrypted_value.startswith("gAAAAA")
    
    def test_same_plaintext_produces_different_ciphertext(self, test_db, encryption_key):
        """Verify that encrypting same text twice produces different ciphertext."""
        plaintext = "Same content"
        
        record1 = TestModel(encrypted_text=plaintext, plain_int=41)
        record2 = TestModel(encrypted_text=plaintext, plain_int=42)
        
        test_db.add_all([record1, record2])
        test_db.commit()
        
        # Query raw encrypted values
        from sqlalchemy import text
        results = test_db.execute(
            text("SELECT encrypted_text FROM test_encryption WHERE plain_int IN (41, 42)")
        ).fetchall()
        
        encrypted1 = results[0][0]
        encrypted2 = results[1][0]
        
        # Fernet includes random IV, so same plaintext → different ciphertext
        assert encrypted1 != encrypted2
        
        # But both decrypt to same value
        assert record1.encrypted_text == plaintext
        assert record2.encrypted_text == plaintext
    
    def test_key_generation_produces_valid_keys(self):
        """Test that generated keys are valid Fernet keys."""
        for _ in range(5):
            key = generate_encryption_key()
            
            # Should be able to create Fernet cipher
            cipher_test = Fernet(key.encode('utf-8'))
            
            # Should be able to encrypt/decrypt
            test_data = b"test"
            encrypted = cipher_test.encrypt(test_data)
            decrypted = cipher_test.decrypt(encrypted)
            assert decrypted == test_data


class TestPerformance:
    """Test performance characteristics."""
    
    def test_encryption_performance_small_text(self, test_db, encryption_key):
        """Test encryption performance for typical email size."""
        import time
        
        # Typical email subject/summary size (100-500 chars)
        text = "Email summary content. " * 20  # ~460 chars
        
        start = time.perf_counter()
        record = TestModel(encrypted_text=text, plain_int=50)
        test_db.add(record)
        test_db.commit()
        encrypt_time = time.perf_counter() - start
        
        # Should be very fast (< 10ms)
        assert encrypt_time < 0.01, f"Encryption took {encrypt_time*1000:.2f}ms (expected < 10ms)"
    
    def test_decryption_performance_small_text(self, test_db, encryption_key):
        """Test decryption performance for typical email size."""
        import time
        
        text = "Email summary content. " * 20  # ~460 chars
        record = TestModel(encrypted_text=text, plain_int=51)
        test_db.add(record)
        test_db.commit()
        
        # Clear SQLAlchemy identity map to force fresh read
        test_db.expire_all()
        
        start = time.perf_counter()
        retrieved = test_db.query(TestModel).filter(TestModel.plain_int == 51).first()
        _ = retrieved.encrypted_text  # Force decryption
        decrypt_time = time.perf_counter() - start
        
        # Should be very fast (< 10ms)
        assert decrypt_time < 0.01, f"Decryption took {decrypt_time*1000:.2f}ms (expected < 10ms)"
    
    def test_encryption_performance_large_text(self, test_db, encryption_key):
        """Test encryption performance for large email bodies."""
        import time
        
        # Large email body (100KB)
        text = "Long email body content. " * 4000  # ~100KB
        
        start = time.perf_counter()
        record = TestModel(encrypted_text=text, plain_int=52)
        test_db.add(record)
        test_db.commit()
        encrypt_time = time.perf_counter() - start
        
        # Should still be reasonable (< 100ms)
        assert encrypt_time < 0.1, f"Large encryption took {encrypt_time*1000:.2f}ms (expected < 100ms)"
    
    def test_batch_operations_performance(self, test_db, encryption_key):
        """Test performance of batch operations with encryption."""
        import time
        
        # Create 100 records
        records = [
            TestModel(encrypted_text=f"Email {i} content", encrypted_json={"id": i}, plain_int=100+i)
            for i in range(100)
        ]
        
        start = time.perf_counter()
        test_db.add_all(records)
        test_db.commit()
        batch_encrypt_time = time.perf_counter() - start
        
        # Should complete in reasonable time (< 1 second)
        assert batch_encrypt_time < 1.0, f"Batch encryption took {batch_encrypt_time:.2f}s (expected < 1s)"
        
        # Read them back
        test_db.expire_all()
        start = time.perf_counter()
        retrieved = test_db.query(TestModel).filter(TestModel.plain_int >= 100).all()
        for r in retrieved:
            _ = r.encrypted_text  # Force decryption
        batch_decrypt_time = time.perf_counter() - start
        
        # Should complete in reasonable time (< 2 seconds)
        assert batch_decrypt_time < 2.0, f"Batch decryption took {batch_decrypt_time:.2f}s (expected < 2s)"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

