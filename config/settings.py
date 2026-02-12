"""Application configuration and settings for AI Trading Agent"""

import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from cryptography.fernet import Fernet

# Load environment variables from .env.ai file
load_dotenv('.env.ai')


@dataclass
class AlpacaConfig:
    """Alpaca API configuration"""
    api_key: str
    secret_key: str
    paper_trading: bool = True
    base_url: Optional[str] = None
    
    def __post_init__(self):
        """Set base URL based on trading mode"""
        if self.base_url is None:
            if self.paper_trading:
                self.base_url = "https://paper-api.alpaca.markets"
            else:
                self.base_url = "https://api.alpaca.markets"


@dataclass
class NewsConfig:
    """News API configuration"""
    api_key: str
    max_articles: int = 50
    language: str = "en"


@dataclass
class RedisConfig:
    """Redis cache configuration"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    enabled: bool = False


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1  # 10% of portfolio
    max_portfolio_risk: float = 0.02  # 2% of portfolio
    daily_loss_limit: float = 1000.0  # $1000
    stop_loss_pct: float = 0.05  # 5%
    take_profit_pct: float = 0.10  # 10%
    max_open_positions: int = 10


@dataclass
class ModelConfig:
    """ML model configuration"""
    model_dir: Path
    cache_dir: Path
    lstm_lookback: int = 60  # days
    lstm_epochs: int = 50
    rf_n_estimators: int = 100
    rf_max_depth: int = 10
    min_confidence: float = 0.6  # Minimum confidence for signals


@dataclass
class DatabaseConfig:
    """Database configuration"""
    path: Path
    
    def __post_init__(self):
        """Ensure database directory exists"""
        self.path.parent.mkdir(parents=True, exist_ok=True)


class Settings:
    """Main application settings"""
    
    def __init__(self):
        """Initialize settings from environment variables"""
        # Alpaca configuration
        self.alpaca = AlpacaConfig(
            api_key=os.getenv('ALPACA_API_KEY', ''),
            secret_key=os.getenv('ALPACA_SECRET_KEY', ''),
            paper_trading=os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        )
        
        # News API configuration
        self.news = NewsConfig(
            api_key=os.getenv('NEWS_API_KEY', '')
        )
        
        # Redis configuration
        self.redis = RedisConfig(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', '6379')),
            db=int(os.getenv('REDIS_DB', '0')),
            enabled=os.getenv('REDIS_ENABLED', 'false').lower() == 'true'
        )
        
        # Risk management configuration
        self.risk = RiskConfig(
            max_position_size=float(os.getenv('MAX_POSITION_SIZE', '0.1')),
            max_portfolio_risk=float(os.getenv('MAX_PORTFOLIO_RISK', '0.02')),
            daily_loss_limit=float(os.getenv('DAILY_LOSS_LIMIT', '1000')),
            stop_loss_pct=float(os.getenv('STOP_LOSS_PCT', '0.05')),
            take_profit_pct=float(os.getenv('TAKE_PROFIT_PCT', '0.10')),
            max_open_positions=int(os.getenv('MAX_OPEN_POSITIONS', '10'))
        )
        
        # Model configuration
        model_dir = Path(os.getenv('MODEL_DIR', 'data/models'))
        cache_dir = Path(os.getenv('CACHE_DIR', 'data/cache'))
        model_dir.mkdir(parents=True, exist_ok=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = ModelConfig(
            model_dir=model_dir,
            cache_dir=cache_dir
        )
        
        # Database configuration
        db_path = Path(os.getenv('DATABASE_PATH', 'data/database/trading.db'))
        self.database = DatabaseConfig(path=db_path)
        
        # Application settings
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.cache_ttl_seconds = int(os.getenv('CACHE_TTL_SECONDS', '300'))
        
        # Convenience property for news API key
        self.news_api_key = self.news.api_key
        
        # Feature flags
        self.enable_automated_trading = os.getenv('ENABLE_AUTOMATED_TRADING', 'false').lower() == 'true'
        self.enable_sentiment_analysis = os.getenv('ENABLE_SENTIMENT_ANALYSIS', 'true').lower() == 'true'
        self.enable_paper_trading = os.getenv('ENABLE_PAPER_TRADING', 'true').lower() == 'true'
        
        # Encryption key for credentials (generate if not exists)
        self._encryption_key = self._get_or_create_encryption_key()
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for credential storage"""
        key_file = Path('data/.encryption_key')
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new key
            key = Fernet.generate_key()
            key_file.parent.mkdir(parents=True, exist_ok=True)
            with open(key_file, 'wb') as f:
                f.write(key)
            # Set restrictive permissions (owner only)
            os.chmod(key_file, 0o600)
            return key
    
    def encrypt_credential(self, credential: str) -> bytes:
        """
        Encrypt a credential string.
        
        Args:
            credential: Plain text credential
            
        Returns:
            Encrypted credential bytes
        """
        fernet = Fernet(self._encryption_key)
        return fernet.encrypt(credential.encode())
    
    def decrypt_credential(self, encrypted: bytes) -> str:
        """
        Decrypt a credential.
        
        Args:
            encrypted: Encrypted credential bytes
            
        Returns:
            Plain text credential
        """
        fernet = Fernet(self._encryption_key)
        return fernet.decrypt(encrypted).decode()
    
    def save_credentials(self, credentials: Dict[str, str], credential_type: str = 'broker') -> bool:
        """
        Save credentials securely with encryption.
        
        Args:
            credentials: Dictionary of credential key-value pairs
            credential_type: Type of credentials (e.g., 'broker', 'news', 'custom')
            
        Returns:
            True if successful, False otherwise
            
        Example:
            settings.save_credentials({
                'api_key': 'your_api_key',
                'secret_key': 'your_secret_key'
            }, 'broker')
        """
        try:
            # Create credentials directory if it doesn't exist
            creds_dir = Path('data/credentials')
            creds_dir.mkdir(parents=True, exist_ok=True)
            
            # Encrypt each credential
            encrypted_creds = {}
            for key, value in credentials.items():
                if value:  # Only encrypt non-empty values
                    encrypted_creds[key] = self.encrypt_credential(value).decode('utf-8')
            
            # Save to file
            creds_file = creds_dir / f'{credential_type}_credentials.enc'
            with open(creds_file, 'w') as f:
                json.dump(encrypted_creds, f)
            
            # Set restrictive permissions (owner only)
            os.chmod(creds_file, 0o600)
            
            return True
        except Exception as e:
            print(f"Error saving credentials: {e}")
            return False
    
    def load_credentials(self, credential_type: str = 'broker') -> Optional[Dict[str, str]]:
        """
        Load and decrypt credentials from secure storage.
        
        Args:
            credential_type: Type of credentials to load (e.g., 'broker', 'news', 'custom')
            
        Returns:
            Dictionary of decrypted credentials, or None if not found or error occurs
            
        Example:
            creds = settings.load_credentials('broker')
            if creds:
                api_key = creds.get('api_key')
                secret_key = creds.get('secret_key')
        """
        try:
            creds_file = Path('data/credentials') / f'{credential_type}_credentials.enc'
            
            if not creds_file.exists():
                return None
            
            # Load encrypted credentials
            with open(creds_file, 'r') as f:
                encrypted_creds = json.load(f)
            
            # Decrypt each credential
            decrypted_creds = {}
            for key, encrypted_value in encrypted_creds.items():
                decrypted_creds[key] = self.decrypt_credential(encrypted_value.encode('utf-8'))
            
            return decrypted_creds
        except Exception as e:
            print(f"Error loading credentials: {e}")
            return None
    
    def delete_credentials(self, credential_type: str = 'broker') -> bool:
        """
        Securely delete stored credentials with secure overwrite.
        
        This function overwrites the credential file with random data before deletion
        to prevent recovery of sensitive information.
        
        Args:
            credential_type: Type of credentials to delete (e.g., 'broker', 'news', 'custom')
            
        Returns:
            True if successful, False otherwise
            
        Example:
            settings.delete_credentials('broker')
        """
        try:
            creds_file = Path('data/credentials') / f'{credential_type}_credentials.enc'
            
            if not creds_file.exists():
                return True  # Already deleted
            
            # Secure overwrite: write random data multiple times
            file_size = creds_file.stat().st_size
            with open(creds_file, 'wb') as f:
                # Overwrite with random data 3 times
                for _ in range(3):
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete the file
            creds_file.unlink()
            
            return True
        except Exception as e:
            print(f"Error deleting credentials: {e}")
            return False
    
    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate configuration.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check required API keys
        if not self.alpaca.api_key:
            errors.append("ALPACA_API_KEY is not set")
        if not self.alpaca.secret_key:
            errors.append("ALPACA_SECRET_KEY is not set")
        
        # News API is optional but warn if not set
        if not self.news.api_key and self.enable_sentiment_analysis:
            errors.append("NEWS_API_KEY is not set (sentiment analysis will be disabled)")
        
        # Validate risk parameters
        if not 0 < self.risk.max_position_size <= 1:
            errors.append("MAX_POSITION_SIZE must be between 0 and 1")
        if not 0 < self.risk.max_portfolio_risk <= 1:
            errors.append("MAX_PORTFOLIO_RISK must be between 0 and 1")
        if self.risk.daily_loss_limit <= 0:
            errors.append("DAILY_LOSS_LIMIT must be positive")
        if not 0 < self.risk.stop_loss_pct < 1:
            errors.append("STOP_LOSS_PCT must be between 0 and 1")
        if not 0 < self.risk.take_profit_pct < 1:
            errors.append("TAKE_PROFIT_PCT must be between 0 and 1")
        if self.risk.max_open_positions <= 0:
            errors.append("MAX_OPEN_POSITIONS must be positive")
        
        return len(errors) == 0, errors


# Global settings instance
settings = Settings()
