"""
Test suite for security utilities and validation.

This module tests input sanitization, injection prevention,
authentication, authorization, and security compliance.
"""

import hashlib
import hmac
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest

from abov3.utils.security import (
    InputSanitizer,
    SecurityValidator,
    CredentialManager,
    EncryptionHandler,
    AuditLogger,
    RateLimiter,
    AccessControl,
    sanitize_input,
    validate_input,
    check_injection,
    encrypt_data,
    decrypt_data,
    hash_password,
    verify_password
)


class TestInputSanitizer:
    """Test cases for InputSanitizer class."""
    
    @pytest.fixture
    def sanitizer(self):
        """Create a test sanitizer instance."""
        return InputSanitizer()
    
    def test_sanitize_html(self, sanitizer):
        """Test HTML sanitization."""
        malicious_inputs = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<iframe src='malicious.com'></iframe>",
            "javascript:alert('XSS')",
            "<body onload=alert('XSS')>"
        ]
        
        for input_str in malicious_inputs:
            sanitized = sanitizer.sanitize_html(input_str)
            assert "<script>" not in sanitized
            assert "javascript:" not in sanitized
            assert "onerror=" not in sanitized
    
    def test_sanitize_sql(self, sanitizer):
        """Test SQL injection prevention."""
        sql_injections = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM passwords--",
            "1; DELETE FROM users WHERE 1=1"
        ]
        
        for injection in sql_injections:
            sanitized = sanitizer.sanitize_sql(injection)
            assert "DROP" not in sanitized.upper()
            assert "DELETE" not in sanitized.upper()
            assert "--" not in sanitized
    
    def test_sanitize_command(self, sanitizer):
        """Test command injection prevention."""
        command_injections = [
            "; ls -la",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "$(whoami)",
            "`id`",
            "; shutdown -h now"
        ]
        
        for injection in command_injections:
            sanitized = sanitizer.sanitize_command(injection)
            assert ";" not in sanitized
            assert "|" not in sanitized
            assert "&" not in sanitized
            assert "$(" not in sanitized
            assert "`" not in sanitized
    
    def test_sanitize_path(self, sanitizer):
        """Test path traversal prevention."""
        path_traversals = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32",
            "file:///etc/passwd",
            "/etc/passwd",
            "C:\\Windows\\System32\\config\\sam"
        ]
        
        for path in path_traversals:
            sanitized = sanitizer.sanitize_path(path)
            assert ".." not in sanitized
            assert not sanitized.startswith("/")
            assert not sanitized.startswith("\\")
            assert "://" not in sanitized
    
    def test_sanitize_ldap(self, sanitizer):
        """Test LDAP injection prevention."""
        ldap_injections = [
            "*)(uid=*",
            "*)(|(uid=*))",
            "admin)(|(password=*))",
            "*)(&(cn=*))"
        ]
        
        for injection in ldap_injections:
            sanitized = sanitizer.sanitize_ldap(injection)
            assert "*" not in sanitized
            assert "(" not in sanitized
            assert ")" not in sanitized
            assert "|" not in sanitized
    
    def test_sanitize_xml(self, sanitizer):
        """Test XXE prevention."""
        xxe_payloads = [
            "<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>",
            "<!ENTITY xxe SYSTEM 'http://evil.com'>",
            "<?xml version='1.0'?><!DOCTYPE foo [<!ELEMENT foo ANY>]>"
        ]
        
        for payload in xxe_payloads:
            sanitized = sanitizer.sanitize_xml(payload)
            assert "<!DOCTYPE" not in sanitized
            assert "<!ENTITY" not in sanitized
            assert "SYSTEM" not in sanitized
    
    def test_sanitize_unicode(self, sanitizer):
        """Test Unicode attack prevention."""
        unicode_attacks = [
            "\u202e\u0041\u0042\u0043",  # Right-to-left override
            "\ufeff",  # Zero-width no-break space
            "\u200b",  # Zero-width space
            "\u00ad"  # Soft hyphen
        ]
        
        for attack in unicode_attacks:
            sanitized = sanitizer.sanitize_unicode(attack)
            # Should remove or normalize dangerous Unicode
            assert len(sanitized) < len(attack) or sanitized != attack


class TestSecurityValidator:
    """Test cases for SecurityValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a test validator instance."""
        return SecurityValidator()
    
    def test_validate_email(self, validator):
        """Test email validation."""
        valid_emails = [
            "user@example.com",
            "test.user@domain.co.uk",
            "user+tag@example.org"
        ]
        
        invalid_emails = [
            "not-an-email",
            "@example.com",
            "user@",
            "user@.com",
            "user@domain",
            "<script>@example.com"
        ]
        
        for email in valid_emails:
            assert validator.validate_email(email) is True
        
        for email in invalid_emails:
            assert validator.validate_email(email) is False
    
    def test_validate_url(self, validator):
        """Test URL validation."""
        valid_urls = [
            "http://example.com",
            "https://secure.example.com",
            "http://localhost:8080",
            "https://example.com/path?query=value"
        ]
        
        invalid_urls = [
            "javascript:alert('XSS')",
            "file:///etc/passwd",
            "data:text/html,<script>alert('XSS')</script>",
            "ftp://insecure.com",
            "not-a-url"
        ]
        
        for url in valid_urls:
            assert validator.validate_url(url) is True
        
        for url in invalid_urls:
            assert validator.validate_url(url) is False
    
    def test_validate_json(self, validator):
        """Test JSON validation."""
        valid_json = [
            '{"key": "value"}',
            '["item1", "item2"]',
            '{"nested": {"key": "value"}}',
            '123',
            'true'
        ]
        
        invalid_json = [
            '{key: "value"}',  # Missing quotes
            '{"key": undefined}',  # Undefined
            "{'key': 'value'}",  # Single quotes
            '{"key": "value",}',  # Trailing comma
            'NaN'
        ]
        
        for json_str in valid_json:
            assert validator.validate_json(json_str) is True
        
        for json_str in invalid_json:
            assert validator.validate_json(json_str) is False
    
    def test_validate_filename(self, validator):
        """Test filename validation."""
        valid_filenames = [
            "document.pdf",
            "image_001.jpg",
            "data-file.csv",
            "report_2024.docx"
        ]
        
        invalid_filenames = [
            "../etc/passwd",
            "file:name.txt",
            "file\x00.txt",  # Null byte
            "|pipe.txt",
            "<script>.js",
            "con.txt"  # Windows reserved name
        ]
        
        for filename in valid_filenames:
            assert validator.validate_filename(filename) is True
        
        for filename in invalid_filenames:
            assert validator.validate_filename(filename) is False
    
    def test_validate_input_length(self, validator):
        """Test input length validation."""
        # Valid lengths
        assert validator.validate_length("test", min_len=1, max_len=10) is True
        assert validator.validate_length("x" * 100, max_len=100) is True
        
        # Invalid lengths
        assert validator.validate_length("", min_len=1) is False
        assert validator.validate_length("x" * 101, max_len=100) is False
    
    def test_validate_alphanumeric(self, validator):
        """Test alphanumeric validation."""
        valid_inputs = ["abc123", "Test123", "999", "AAA"]
        invalid_inputs = ["abc-123", "test@123", "abc 123", "abc!"]
        
        for input_str in valid_inputs:
            assert validator.validate_alphanumeric(input_str) is True
        
        for input_str in invalid_inputs:
            assert validator.validate_alphanumeric(input_str) is False


class TestCredentialManager:
    """Test cases for CredentialManager class."""
    
    @pytest.fixture
    def cred_manager(self, temp_dir):
        """Create a test credential manager."""
        return CredentialManager(storage_path=temp_dir)
    
    def test_store_credential(self, cred_manager):
        """Test storing credentials securely."""
        cred_manager.store("api_key", "secret_key_123")
        
        # Should be stored encrypted
        stored = cred_manager.get("api_key")
        assert stored == "secret_key_123"
    
    def test_credential_encryption(self, cred_manager):
        """Test that credentials are encrypted in storage."""
        cred_manager.store("password", "my_password")
        
        # Check raw storage - should be encrypted
        raw_data = cred_manager._storage.get("password")
        assert raw_data != "my_password"
        assert isinstance(raw_data, bytes) or "encrypted" in str(raw_data)
    
    def test_delete_credential(self, cred_manager):
        """Test deleting credentials."""
        cred_manager.store("temp_key", "temp_value")
        assert cred_manager.get("temp_key") == "temp_value"
        
        cred_manager.delete("temp_key")
        assert cred_manager.get("temp_key") is None
    
    def test_list_credentials(self, cred_manager):
        """Test listing stored credentials."""
        cred_manager.store("key1", "value1")
        cred_manager.store("key2", "value2")
        cred_manager.store("key3", "value3")
        
        keys = cred_manager.list_keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys
    
    def test_credential_rotation(self, cred_manager):
        """Test credential rotation."""
        cred_manager.store("rotating_key", "old_value")
        
        # Rotate credential
        cred_manager.rotate("rotating_key", "new_value")
        
        assert cred_manager.get("rotating_key") == "new_value"
        # Old value should be in history (if implemented)
        history = cred_manager.get_history("rotating_key")
        if history:
            assert "old_value" in history


class TestEncryptionHandler:
    """Test cases for EncryptionHandler class."""
    
    @pytest.fixture
    def encryption(self):
        """Create a test encryption handler."""
        return EncryptionHandler(key="test_key_12345678")
    
    def test_encrypt_decrypt_string(self, encryption):
        """Test string encryption and decryption."""
        plaintext = "This is a secret message"
        
        encrypted = encryption.encrypt(plaintext)
        assert encrypted != plaintext
        
        decrypted = encryption.decrypt(encrypted)
        assert decrypted == plaintext
    
    def test_encrypt_decrypt_json(self, encryption):
        """Test JSON data encryption."""
        data = {"username": "admin", "password": "secret123"}
        
        encrypted = encryption.encrypt_json(data)
        assert isinstance(encrypted, (str, bytes))
        
        decrypted = encryption.decrypt_json(encrypted)
        assert decrypted == data
    
    def test_different_keys_produce_different_results(self):
        """Test that different keys produce different encrypted results."""
        handler1 = EncryptionHandler(key="key1")
        handler2 = EncryptionHandler(key="key2")
        
        plaintext = "same message"
        
        encrypted1 = handler1.encrypt(plaintext)
        encrypted2 = handler2.encrypt(plaintext)
        
        assert encrypted1 != encrypted2
    
    def test_wrong_key_fails_decryption(self):
        """Test that wrong key fails to decrypt."""
        handler1 = EncryptionHandler(key="correct_key")
        handler2 = EncryptionHandler(key="wrong_key")
        
        plaintext = "secret"
        encrypted = handler1.encrypt(plaintext)
        
        with pytest.raises(Exception):
            handler2.decrypt(encrypted)
    
    def test_encrypt_file(self, encryption, temp_dir):
        """Test file encryption."""
        file_path = temp_dir / "test.txt"
        file_path.write_text("File content to encrypt")
        
        encrypted_path = encryption.encrypt_file(file_path)
        assert encrypted_path.exists()
        
        # Encrypted file should be different
        assert encrypted_path.read_bytes() != file_path.read_bytes()
        
        # Decrypt file
        decrypted_path = encryption.decrypt_file(encrypted_path)
        assert decrypted_path.read_text() == "File content to encrypt"


class TestPasswordHashing:
    """Test cases for password hashing functions."""
    
    def test_hash_password(self):
        """Test password hashing."""
        password = "my_secure_password"
        
        hashed = hash_password(password)
        assert hashed != password
        assert len(hashed) > 0
    
    def test_verify_password(self):
        """Test password verification."""
        password = "test_password_123"
        wrong_password = "wrong_password"
        
        hashed = hash_password(password)
        
        # Correct password should verify
        assert verify_password(password, hashed) is True
        
        # Wrong password should not verify
        assert verify_password(wrong_password, hashed) is False
    
    def test_same_password_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        password = "same_password"
        
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        # Should have different salts
        assert hash1 != hash2
        
        # Both should verify correctly
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True
    
    def test_hash_strength(self):
        """Test hash strength and format."""
        password = "test"
        hashed = hash_password(password)
        
        # Should use strong hashing (bcrypt, scrypt, or argon2)
        assert len(hashed) >= 60  # Minimum for bcrypt
        
        # Should not be a simple hash
        assert hashed != hashlib.sha256(password.encode()).hexdigest()


class TestRateLimiter:
    """Test cases for RateLimiter class."""
    
    @pytest.fixture
    def rate_limiter(self):
        """Create a test rate limiter."""
        return RateLimiter(max_requests=5, window_seconds=60)
    
    def test_rate_limit_allows_requests(self, rate_limiter):
        """Test that rate limiter allows requests within limit."""
        client_id = "client1"
        
        for i in range(5):
            assert rate_limiter.check(client_id) is True
    
    def test_rate_limit_blocks_excess(self, rate_limiter):
        """Test that rate limiter blocks excess requests."""
        client_id = "client2"
        
        # Use up the limit
        for i in range(5):
            rate_limiter.check(client_id)
        
        # Next request should be blocked
        assert rate_limiter.check(client_id) is False
    
    def test_rate_limit_window_reset(self, rate_limiter):
        """Test that rate limit resets after window."""
        client_id = "client3"
        
        # Use up the limit
        for i in range(5):
            rate_limiter.check(client_id)
        
        # Should be blocked
        assert rate_limiter.check(client_id) is False
        
        # Simulate time passing
        import time
        with patch('time.time', return_value=time.time() + 61):
            # Should be allowed again
            assert rate_limiter.check(client_id) is True
    
    def test_different_clients_independent(self, rate_limiter):
        """Test that different clients have independent limits."""
        # Client 1 uses up limit
        for i in range(5):
            rate_limiter.check("client1")
        
        # Client 2 should still be allowed
        assert rate_limiter.check("client2") is True


class TestAccessControl:
    """Test cases for AccessControl class."""
    
    @pytest.fixture
    def access_control(self):
        """Create a test access control instance."""
        return AccessControl()
    
    def test_grant_permission(self, access_control):
        """Test granting permissions."""
        access_control.grant("user1", "read", "resource1")
        
        assert access_control.check("user1", "read", "resource1") is True
        assert access_control.check("user1", "write", "resource1") is False
    
    def test_revoke_permission(self, access_control):
        """Test revoking permissions."""
        access_control.grant("user2", "write", "resource2")
        assert access_control.check("user2", "write", "resource2") is True
        
        access_control.revoke("user2", "write", "resource2")
        assert access_control.check("user2", "write", "resource2") is False
    
    def test_role_based_access(self, access_control):
        """Test role-based access control."""
        # Define roles
        access_control.define_role("admin", ["read", "write", "delete"])
        access_control.define_role("user", ["read"])
        
        # Assign roles
        access_control.assign_role("alice", "admin")
        access_control.assign_role("bob", "user")
        
        # Check permissions
        assert access_control.check_role("alice", "delete") is True
        assert access_control.check_role("bob", "delete") is False
        assert access_control.check_role("bob", "read") is True
    
    def test_resource_ownership(self, access_control):
        """Test resource ownership validation."""
        access_control.set_owner("resource1", "owner1")
        
        assert access_control.is_owner("owner1", "resource1") is True
        assert access_control.is_owner("owner2", "resource1") is False


class TestAuditLogger:
    """Test cases for AuditLogger class."""
    
    @pytest.fixture
    def audit_logger(self, temp_dir):
        """Create a test audit logger."""
        return AuditLogger(log_path=temp_dir / "audit.log")
    
    def test_log_security_event(self, audit_logger):
        """Test logging security events."""
        audit_logger.log_event(
            event_type="login_attempt",
            user="testuser",
            success=True,
            ip_address="192.168.1.1"
        )
        
        events = audit_logger.get_events(event_type="login_attempt")
        assert len(events) > 0
        assert events[0]["user"] == "testuser"
        assert events[0]["success"] is True
    
    def test_log_failed_attempts(self, audit_logger):
        """Test logging failed security attempts."""
        for i in range(3):
            audit_logger.log_failed_attempt(
                user="attacker",
                action="login",
                reason="invalid_password"
            )
        
        failed = audit_logger.get_failed_attempts("attacker")
        assert len(failed) == 3
    
    def test_detect_suspicious_activity(self, audit_logger):
        """Test detection of suspicious activity patterns."""
        # Log many failed attempts
        for i in range(10):
            audit_logger.log_failed_attempt(
                user="suspicious",
                action="access",
                ip_address="10.0.0.1"
            )
        
        suspicious = audit_logger.detect_suspicious("suspicious")
        assert suspicious is True
    
    def test_audit_log_rotation(self, audit_logger, temp_dir):
        """Test audit log rotation."""
        # Log many events
        for i in range(1000):
            audit_logger.log_event(f"event_{i}", data={"index": i})
        
        # Check that log rotation occurred if configured
        log_files = list(temp_dir.glob("audit*.log*"))
        # Should have rotated logs if size exceeded
        assert len(log_files) >= 1


class TestSecurityCompliance:
    """Test cases for security compliance checks."""
    
    def test_owasp_top10_compliance(self, sanitizer, validator):
        """Test OWASP Top 10 compliance."""
        # A01:2021 - Broken Access Control
        # A02:2021 - Cryptographic Failures
        # A03:2021 - Injection
        injections = [
            "'; DROP TABLE users; --",
            "<script>alert('XSS')</script>",
            "../../etc/passwd"
        ]
        
        for injection in injections:
            assert sanitizer.sanitize_sql(injection) != injection
            assert sanitizer.sanitize_html(injection) != injection
            assert sanitizer.sanitize_path(injection) != injection
    
    def test_gdpr_compliance(self):
        """Test GDPR compliance features."""
        # Personal data should be encrypted
        encryption = EncryptionHandler(key="gdpr_key")
        personal_data = {
            "name": "John Doe",
            "email": "john@example.com",
            "ssn": "123-45-6789"
        }
        
        encrypted = encryption.encrypt_json(personal_data)
        assert encrypted != str(personal_data)
        
        # Should be able to delete personal data
        cred_manager = CredentialManager()
        cred_manager.store("user_data", personal_data)
        cred_manager.delete("user_data")
        assert cred_manager.get("user_data") is None
    
    def test_pci_dss_compliance(self):
        """Test PCI DSS compliance for payment data."""
        # Credit card numbers should be masked
        sanitizer = InputSanitizer()
        cc_numbers = [
            "4532-1234-5678-9010",
            "5500 0000 0000 0004",
            "3400 000000 00009"
        ]
        
        for cc in cc_numbers:
            masked = sanitizer.mask_credit_card(cc)
            assert masked != cc
            assert "****" in masked


class TestSecurityPerformance:
    """Performance tests for security operations."""
    
    @pytest.mark.performance
    def test_encryption_performance(self):
        """Test encryption/decryption performance."""
        encryption = EncryptionHandler(key="perf_test_key")
        data = "x" * 10000  # 10KB of data
        
        import time
        
        # Encryption performance
        start = time.perf_counter()
        for _ in range(100):
            encrypted = encryption.encrypt(data)
        end = time.perf_counter()
        
        avg_encrypt_time = (end - start) / 100
        assert avg_encrypt_time < 0.01  # Should encrypt in < 10ms
        
        # Decryption performance
        start = time.perf_counter()
        for _ in range(100):
            decrypted = encryption.decrypt(encrypted)
        end = time.perf_counter()
        
        avg_decrypt_time = (end - start) / 100
        assert avg_decrypt_time < 0.01  # Should decrypt in < 10ms
    
    @pytest.mark.performance
    def test_validation_performance(self):
        """Test input validation performance."""
        validator = SecurityValidator()
        
        import time
        
        inputs = ["test@example.com"] * 1000
        
        start = time.perf_counter()
        for input_str in inputs:
            validator.validate_email(input_str)
        end = time.perf_counter()
        
        avg_time = (end - start) / 1000
        assert avg_time < 0.001  # Should validate in < 1ms
    
    @pytest.mark.performance
    def test_sanitization_performance(self):
        """Test input sanitization performance."""
        sanitizer = InputSanitizer()
        
        malicious_input = "'; DROP TABLE users; --" * 100
        
        import time
        
        start = time.perf_counter()
        for _ in range(100):
            sanitized = sanitizer.sanitize_sql(malicious_input)
        end = time.perf_counter()
        
        avg_time = (end - start) / 100
        assert avg_time < 0.01  # Should sanitize in < 10ms