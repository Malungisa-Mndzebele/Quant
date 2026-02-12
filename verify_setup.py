"""Verify AI Trading Agent setup and configuration"""

import sys
from pathlib import Path

def check_directories():
    """Check if required directories exist"""
    required_dirs = [
        'ai',
        'ai/models',
        'config',
        'data',
        'data/models',
        'data/cache',
        'data/database',
    ]
    
    print("Checking directory structure...")
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_config_files():
    """Check if configuration files exist"""
    required_files = [
        'config/__init__.py',
        'config/settings.py',
        'config/logging_config.py',
        'ai_requirements.txt',
        '.env.ai.example',
    ]
    
    print("\nChecking configuration files...")
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_env_file():
    """Check if .env.ai file exists"""
    print("\nChecking environment configuration...")
    env_file = Path('.env.ai')
    
    if env_file.exists():
        print("  ✓ .env.ai exists")
        return True
    else:
        print("  ⚠ .env.ai not found")
        print("    Run: cp .env.ai.example .env.ai")
        print("    Then edit .env.ai with your API keys")
        return False

def check_imports():
    """Check if configuration modules can be imported"""
    print("\nChecking Python imports...")
    
    try:
        from config.logging_config import setup_logging
        print("  ✓ config.logging_config")
    except ImportError as e:
        print(f"  ✗ config.logging_config - {e}")
        return False
    
    try:
        from config.settings import settings
        print("  ✓ config.settings")
    except ImportError as e:
        print(f"  ✗ config.settings - {e}")
        return False
    
    return True

def check_settings_validation():
    """Validate settings configuration"""
    print("\nValidating settings...")
    
    try:
        from config.settings import settings
        is_valid, errors = settings.validate()
        
        if is_valid:
            print("  ✓ Configuration is valid")
        else:
            print("  ⚠ Configuration has warnings:")
            for error in errors:
                print(f"    - {error}")
        
        # Display current configuration
        print("\nCurrent configuration:")
        print(f"  Paper Trading: {settings.alpaca.paper_trading}")
        print(f"  Log Level: {settings.log_level}")
        print(f"  Max Position Size: {settings.risk.max_position_size * 100}%")
        print(f"  Daily Loss Limit: ${settings.risk.daily_loss_limit}")
        print(f"  Automated Trading: {settings.enable_automated_trading}")
        print(f"  Sentiment Analysis: {settings.enable_sentiment_analysis}")
        
        return True
    except Exception as e:
        print(f"  ✗ Error validating settings: {e}")
        return False

def main():
    """Run all setup checks"""
    print("=" * 60)
    print("AI Trading Agent - Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Directory Structure", check_directories),
        ("Configuration Files", check_config_files),
        ("Environment File", check_env_file),
        ("Python Imports", check_imports),
        ("Settings Validation", check_settings_validation),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Error during {name}: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    if all(results):
        print("✓ Setup verification complete - All checks passed!")
        print("\nNext steps:")
        print("  1. Install dependencies: pip install -r ai_requirements.txt")
        print("  2. Configure .env.ai with your API keys")
        print("  3. Run the application: streamlit run app.py")
        return 0
    else:
        print("⚠ Setup verification complete - Some checks failed")
        print("\nPlease fix the issues above before running the application.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
