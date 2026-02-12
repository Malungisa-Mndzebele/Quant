"""
Example demonstrating secure credential management.

This example shows how to:
1. Save credentials securely with encryption
2. Load credentials from secure storage
3. Delete credentials securely
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings


def main():
    """Demonstrate credential management functionality"""
    
    # Initialize settings
    settings = Settings()
    
    print("=== Credential Management Example ===\n")
    
    # Example 1: Save broker credentials
    print("1. Saving broker credentials...")
    broker_creds = {
        'api_key': 'ALPACA_API_KEY_EXAMPLE',
        'secret_key': 'ALPACA_SECRET_KEY_EXAMPLE'
    }
    
    success = settings.save_credentials(broker_creds, 'broker')
    if success:
        print("   ✓ Broker credentials saved securely\n")
    else:
        print("   ✗ Failed to save broker credentials\n")
    
    # Example 2: Save news API credentials
    print("2. Saving news API credentials...")
    news_creds = {
        'api_key': 'NEWS_API_KEY_EXAMPLE'
    }
    
    success = settings.save_credentials(news_creds, 'news')
    if success:
        print("   ✓ News API credentials saved securely\n")
    else:
        print("   ✗ Failed to save news API credentials\n")
    
    # Example 3: Load broker credentials
    print("3. Loading broker credentials...")
    loaded_broker = settings.load_credentials('broker')
    if loaded_broker:
        print(f"   ✓ Loaded broker credentials:")
        print(f"     - API Key: {loaded_broker['api_key'][:10]}...")
        print(f"     - Secret Key: {loaded_broker['secret_key'][:10]}...\n")
    else:
        print("   ✗ Failed to load broker credentials\n")
    
    # Example 4: Load news API credentials
    print("4. Loading news API credentials...")
    loaded_news = settings.load_credentials('news')
    if loaded_news:
        print(f"   ✓ Loaded news API credentials:")
        print(f"     - API Key: {loaded_news['api_key'][:10]}...\n")
    else:
        print("   ✗ Failed to load news API credentials\n")
    
    # Example 5: Update credentials (overwrite)
    print("5. Updating broker credentials...")
    updated_broker_creds = {
        'api_key': 'NEW_ALPACA_API_KEY',
        'secret_key': 'NEW_ALPACA_SECRET_KEY'
    }
    
    success = settings.save_credentials(updated_broker_creds, 'broker')
    if success:
        print("   ✓ Broker credentials updated\n")
        
        # Verify update
        loaded_updated = settings.load_credentials('broker')
        if loaded_updated and loaded_updated['api_key'] == updated_broker_creds['api_key']:
            print("   ✓ Update verified\n")
    
    # Example 6: Delete credentials
    print("6. Deleting credentials...")
    
    # Delete broker credentials
    success = settings.delete_credentials('broker')
    if success:
        print("   ✓ Broker credentials deleted securely")
    
    # Delete news credentials
    success = settings.delete_credentials('news')
    if success:
        print("   ✓ News credentials deleted securely\n")
    
    # Verify deletion
    print("7. Verifying deletion...")
    loaded_broker = settings.load_credentials('broker')
    loaded_news = settings.load_credentials('news')
    
    if loaded_broker is None and loaded_news is None:
        print("   ✓ All credentials successfully deleted\n")
    else:
        print("   ✗ Some credentials still exist\n")
    
    print("=== Example Complete ===")
    print("\nKey Features:")
    print("• Credentials are encrypted using Fernet (AES-256)")
    print("• Files have restrictive permissions (owner only)")
    print("• Secure deletion overwrites data before removing files")
    print("• Supports multiple credential types (broker, news, custom)")
    print("• Environment variables are still supported as fallback")


if __name__ == '__main__':
    main()
