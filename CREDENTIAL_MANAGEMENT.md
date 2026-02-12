# Credential Management

## Overview

The AI Trading Agent includes a secure credential management system that encrypts and stores sensitive API keys and credentials. This system uses Fernet (symmetric encryption with AES-256) to protect credentials at rest.

## Features

- **Encryption**: All credentials are encrypted using Fernet (AES-256) before storage
- **Secure Storage**: Encrypted credentials are stored in files with restrictive permissions (owner-only access)
- **Secure Deletion**: Credentials are overwritten with random data before deletion to prevent recovery
- **Multiple Credential Types**: Support for different credential types (broker, news, custom)
- **Environment Variable Fallback**: Credentials can still be loaded from environment variables

## Usage

### Saving Credentials

```python
from config.settings import Settings

settings = Settings()

# Save broker credentials
broker_creds = {
    'api_key': 'your_alpaca_api_key',
    'secret_key': 'your_alpaca_secret_key'
}
success = settings.save_credentials(broker_creds, 'broker')

# Save news API credentials
news_creds = {
    'api_key': 'your_news_api_key'
}
success = settings.save_credentials(news_creds, 'news')
```

### Loading Credentials

```python
# Load broker credentials
broker_creds = settings.load_credentials('broker')
if broker_creds:
    api_key = broker_creds['api_key']
    secret_key = broker_creds['secret_key']

# Load news credentials
news_creds = settings.load_credentials('news')
if news_creds:
    api_key = news_creds['api_key']
```

### Deleting Credentials

```python
# Delete broker credentials
success = settings.delete_credentials('broker')

# Delete news credentials
success = settings.delete_credentials('news')
```

### Updating Credentials

To update credentials, simply save new credentials with the same credential type:

```python
# Update broker credentials
new_broker_creds = {
    'api_key': 'new_alpaca_api_key',
    'secret_key': 'new_alpaca_secret_key'
}
settings.save_credentials(new_broker_creds, 'broker')
```

## Security Features

### Encryption

- **Algorithm**: Fernet (symmetric encryption)
- **Key Size**: 256-bit AES in CBC mode with HMAC authentication
- **Key Storage**: Encryption key is stored in `data/.encryption_key` with owner-only permissions
- **Key Generation**: Automatically generated on first use if not present

### File Permissions

On Unix-like systems (Linux, macOS), credential files are created with `0o600` permissions (owner read/write only). On Windows, the system's default file permissions apply.

### Secure Deletion

When deleting credentials, the system:
1. Overwrites the file with random data 3 times
2. Flushes the data to disk after each overwrite
3. Deletes the file

This prevents recovery of sensitive data using file recovery tools.

## File Structure

```
data/
├── .encryption_key              # Encryption key (auto-generated)
└── credentials/
    ├── broker_credentials.enc   # Encrypted broker credentials
    ├── news_credentials.enc     # Encrypted news credentials
    └── custom_credentials.enc   # Custom credential types
```

## Environment Variables

The system still supports loading credentials from environment variables as a fallback:

```bash
# .env.ai file
ALPACA_API_KEY=your_api_key
ALPACA_SECRET_KEY=your_secret_key
NEWS_API_KEY=your_news_api_key
```

Environment variables take precedence during Settings initialization, but stored credentials can be loaded explicitly using `load_credentials()`.

## Best Practices

1. **Never commit credentials**: Add `data/credentials/` and `data/.encryption_key` to `.gitignore`
2. **Backup encryption key**: If you lose the encryption key, you cannot decrypt stored credentials
3. **Use environment variables for CI/CD**: For automated deployments, use environment variables
4. **Rotate credentials regularly**: Update credentials periodically for security
5. **Delete unused credentials**: Remove credentials when no longer needed

## Example

See `examples/credential_management_example.py` for a complete working example.

## API Reference

### `save_credentials(credentials: Dict[str, str], credential_type: str = 'broker') -> bool`

Save credentials securely with encryption.

**Parameters:**
- `credentials`: Dictionary of credential key-value pairs
- `credential_type`: Type of credentials (e.g., 'broker', 'news', 'custom')

**Returns:**
- `True` if successful, `False` otherwise

### `load_credentials(credential_type: str = 'broker') -> Optional[Dict[str, str]]`

Load and decrypt credentials from secure storage.

**Parameters:**
- `credential_type`: Type of credentials to load

**Returns:**
- Dictionary of decrypted credentials, or `None` if not found or error occurs

### `delete_credentials(credential_type: str = 'broker') -> bool`

Securely delete stored credentials with secure overwrite.

**Parameters:**
- `credential_type`: Type of credentials to delete

**Returns:**
- `True` if successful, `False` otherwise

### `encrypt_credential(credential: str) -> bytes`

Encrypt a credential string.

**Parameters:**
- `credential`: Plain text credential

**Returns:**
- Encrypted credential bytes

### `decrypt_credential(encrypted: bytes) -> str`

Decrypt a credential.

**Parameters:**
- `encrypted`: Encrypted credential bytes

**Returns:**
- Plain text credential

## Requirements Validation

This implementation satisfies the following requirements:

- **Requirement 4.1**: Credentials are securely stored using AES-256 encryption via Fernet
- **Requirement 4.5**: Credentials can be securely deleted with overwrite before removal
- **Support for environment variables**: Existing environment variable loading is maintained

## Testing

Run the credential management tests:

```bash
pytest tests/test_credential_management.py -v
```

Run the example:

```bash
python examples/credential_management_example.py
```
