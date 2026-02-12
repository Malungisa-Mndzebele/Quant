"""Tests for credential management functionality"""

import os
import json
import pytest
from pathlib import Path
from hypothesis import given, strategies as st, settings as hyp_settings
from config.settings import Settings


@pytest.fixture
def settings():
    """Create a Settings instance for testing"""
    return Settings()


@pytest.fixture
def cleanup_credentials():
    """Clean up test credentials after each test"""
    yield
    # Clean up any test credential files
    creds_dir = Path('data/credentials')
    if creds_dir.exists():
        for file in creds_dir.glob('test_*.enc'):
            file.unlink()


def test_save_credentials(settings, cleanup_credentials):
    """Test saving credentials with encryption"""
    test_creds = {
        'api_key': 'test_api_key_12345',
        'secret_key': 'test_secret_key_67890'
    }
    
    # Save credentials
    result = settings.save_credentials(test_creds, 'test_broker')
    assert result is True
    
    # Verify file was created
    creds_file = Path('data/credentials/test_broker_credentials.enc')
    assert creds_file.exists()
    
    # Verify file has restrictive permissions (on Unix-like systems)
    if os.name != 'nt':  # Skip on Windows
        file_stat = creds_file.stat()
        assert oct(file_stat.st_mode)[-3:] == '600'
    
    # Verify content is encrypted (not plaintext)
    with open(creds_file, 'r') as f:
        content = json.load(f)
        # Encrypted values should not match original values
        assert content['api_key'] != test_creds['api_key']
        assert content['secret_key'] != test_creds['secret_key']


def test_load_credentials(settings, cleanup_credentials):
    """Test loading and decrypting credentials"""
    test_creds = {
        'api_key': 'test_api_key_12345',
        'secret_key': 'test_secret_key_67890'
    }
    
    # Save credentials first
    settings.save_credentials(test_creds, 'test_broker')
    
    # Load credentials
    loaded_creds = settings.load_credentials('test_broker')
    
    assert loaded_creds is not None
    assert loaded_creds['api_key'] == test_creds['api_key']
    assert loaded_creds['secret_key'] == test_creds['secret_key']


def test_load_nonexistent_credentials(settings):
    """Test loading credentials that don't exist"""
    loaded_creds = settings.load_credentials('nonexistent_type')
    assert loaded_creds is None


def test_delete_credentials(settings, cleanup_credentials):
    """Test secure deletion of credentials"""
    test_creds = {
        'api_key': 'test_api_key_12345',
        'secret_key': 'test_secret_key_67890'
    }
    
    # Save credentials first
    settings.save_credentials(test_creds, 'test_broker')
    creds_file = Path('data/credentials/test_broker_credentials.enc')
    assert creds_file.exists()
    
    # Delete credentials
    result = settings.delete_credentials('test_broker')
    assert result is True
    
    # Verify file was deleted
    assert not creds_file.exists()


def test_delete_nonexistent_credentials(settings):
    """Test deleting credentials that don't exist"""
    result = settings.delete_credentials('nonexistent_type')
    assert result is True  # Should return True even if file doesn't exist


def test_save_empty_credentials(settings, cleanup_credentials):
    """Test saving credentials with empty values"""
    test_creds = {
        'api_key': 'test_api_key',
        'secret_key': ''  # Empty value
    }
    
    result = settings.save_credentials(test_creds, 'test_broker')
    assert result is True
    
    # Load and verify only non-empty values were saved
    loaded_creds = settings.load_credentials('test_broker')
    assert loaded_creds is not None
    assert 'api_key' in loaded_creds
    assert 'secret_key' not in loaded_creds  # Empty values are not saved


def test_credential_encryption_decryption(settings):
    """Test basic encryption and decryption"""
    original = "my_secret_credential"
    
    # Encrypt
    encrypted = settings.encrypt_credential(original)
    assert encrypted != original.encode()
    
    # Decrypt
    decrypted = settings.decrypt_credential(encrypted)
    assert decrypted == original


def test_multiple_credential_types(settings, cleanup_credentials):
    """Test saving and loading different credential types"""
    broker_creds = {
        'api_key': 'broker_api_key',
        'secret_key': 'broker_secret_key'
    }
    
    news_creds = {
        'api_key': 'news_api_key'
    }
    
    # Save different credential types
    settings.save_credentials(broker_creds, 'test_broker')
    settings.save_credentials(news_creds, 'test_news')
    
    # Load and verify each type
    loaded_broker = settings.load_credentials('test_broker')
    loaded_news = settings.load_credentials('test_news')
    
    assert loaded_broker['api_key'] == broker_creds['api_key']
    assert loaded_broker['secret_key'] == broker_creds['secret_key']
    assert loaded_news['api_key'] == news_creds['api_key']
    
    # Clean up
    settings.delete_credentials('test_broker')
    settings.delete_credentials('test_news')


def test_credential_overwrite(settings, cleanup_credentials):
    """Test overwriting existing credentials"""
    original_creds = {
        'api_key': 'original_key',
        'secret_key': 'original_secret'
    }
    
    new_creds = {
        'api_key': 'new_key',
        'secret_key': 'new_secret'
    }
    
    # Save original credentials
    settings.save_credentials(original_creds, 'test_broker')
    
    # Overwrite with new credentials
    settings.save_credentials(new_creds, 'test_broker')
    
    # Load and verify new credentials
    loaded_creds = settings.load_credentials('test_broker')
    assert loaded_creds['api_key'] == new_creds['api_key']
    assert loaded_creds['secret_key'] == new_creds['secret_key']
    
    # Clean up
    settings.delete_credentials('test_broker')


# Property-Based Tests

@given(
    api_key=st.text(min_size=1, max_size=200),
    secret_key=st.text(min_size=1, max_size=200),
    credential_type=st.text(
        alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd'), min_codepoint=97, max_codepoint=122),
        min_size=1,
        max_size=50
    )
)
@hyp_settings(max_examples=100, deadline=None)
def test_property_credential_encryption(api_key, secret_key, credential_type):
    """
    Property 4: Credential encryption
    
    For any API credentials stored, they should be encrypted using AES-256 
    and never stored in plaintext.
    
    Validates: Requirements 4.1
    
    Feature: ai-trading-agent, Property 4: Credential encryption
    """
    settings = Settings()
    
    # Create test credentials
    test_creds = {
        'api_key': api_key,
        'secret_key': secret_key
    }
    
    # Use a test-specific credential type to avoid conflicts
    test_type = f'test_pbt_{credential_type}'
    
    try:
        # Save credentials
        result = settings.save_credentials(test_creds, test_type)
        assert result is True, "Credentials should be saved successfully"
        
        # Verify file was created
        creds_file = Path('data/credentials') / f'{test_type}_credentials.enc'
        assert creds_file.exists(), "Credential file should exist"
        
        # Read the raw file content
        with open(creds_file, 'r') as f:
            file_content = f.read()
            encrypted_data = json.loads(file_content)
        
        # Property 1: Credentials should NOT be stored in plaintext
        # The encrypted values should not match the original values
        assert encrypted_data['api_key'] != api_key, \
            "API key should not be stored in plaintext"
        assert encrypted_data['secret_key'] != secret_key, \
            "Secret key should not be stored in plaintext"
        
        # Property 2: Encrypted values should be different from original
        # (even if they look similar, they should be base64-encoded encrypted data)
        assert api_key not in file_content, \
            "Original API key should not appear in file content"
        assert secret_key not in file_content, \
            "Original secret key should not appear in file content"
        
        # Property 3: Encrypted data should be decodable (valid base64/Fernet format)
        # This is implicitly tested by the decrypt operation below
        
        # Property 4: Decryption should recover original values (round-trip property)
        loaded_creds = settings.load_credentials(test_type)
        assert loaded_creds is not None, "Credentials should be loadable"
        assert loaded_creds['api_key'] == api_key, \
            "Decrypted API key should match original"
        assert loaded_creds['secret_key'] == secret_key, \
            "Decrypted secret key should match original"
        
        # Property 5: Encryption should use Fernet (AES-256 in CBC mode)
        # Fernet tokens are always 128 bytes or more when base64-encoded
        # and start with 'gAAAAA' after base64 encoding
        encrypted_api_key = encrypted_data['api_key']
        # Fernet uses base64 encoding, so encrypted data should be longer than original
        # and contain only base64 characters
        assert len(encrypted_api_key) > len(api_key), \
            "Encrypted data should be longer than original (includes IV, timestamp, HMAC)"
        
        # Verify it's valid base64 (Fernet format)
        import base64
        try:
            base64.urlsafe_b64decode(encrypted_api_key)
        except Exception:
            pytest.fail("Encrypted data should be valid base64 (Fernet format)")
        
    finally:
        # Clean up test credentials
        settings.delete_credentials(test_type)

