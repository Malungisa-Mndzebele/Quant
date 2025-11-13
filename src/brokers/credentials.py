"""Credential management for brokerage connections."""

import os
import logging
from typing import Dict, Optional
from dataclasses import dataclass

from .errors import AuthenticationError


logger = logging.getLogger(__name__)


@dataclass
class BrokerageCredentials:
    """Container for brokerage credentials."""
    api_key: str
    api_secret: str
    account_id: Optional[str] = None
    additional_params: Optional[Dict[str, str]] = None
    
    def validate(self) -> None:
        """
        Validate that required credentials are present.
        
        Raises:
            AuthenticationError: If credentials are invalid
        """
        if not self.api_key or not self.api_key.strip():
            raise AuthenticationError("API key is required")
        
        if not self.api_secret or not self.api_secret.strip():
            raise AuthenticationError("API secret is required")
        
        logger.debug("Credentials validated successfully")


class CredentialManager:
    """Manages secure loading and validation of brokerage credentials."""
    
    @staticmethod
    def load_from_env(provider: str) -> BrokerageCredentials:
        """
        Load credentials from environment variables.
        
        Environment variables should be named:
        - {PROVIDER}_API_KEY
        - {PROVIDER}_API_SECRET
        - {PROVIDER}_ACCOUNT_ID (optional)
        
        Args:
            provider: Brokerage provider name (e.g., 'PUBLIC', 'MOOMOO')
            
        Returns:
            BrokerageCredentials object
            
        Raises:
            AuthenticationError: If required credentials are missing
        """
        provider_upper = provider.upper()
        
        logger.info(f"Loading credentials for {provider} from environment variables")
        
        api_key = os.getenv(f"{provider_upper}_API_KEY")
        api_secret = os.getenv(f"{provider_upper}_API_SECRET")
        account_id = os.getenv(f"{provider_upper}_ACCOUNT_ID")
        
        if not api_key:
            raise AuthenticationError(
                f"Missing environment variable: {provider_upper}_API_KEY"
            )
        
        if not api_secret:
            raise AuthenticationError(
                f"Missing environment variable: {provider_upper}_API_SECRET"
            )
        
        credentials = BrokerageCredentials(
            api_key=api_key,
            api_secret=api_secret,
            account_id=account_id
        )
        
        credentials.validate()
        
        logger.info(f"Successfully loaded credentials for {provider}")
        return credentials
    
    @staticmethod
    def load_from_dict(credentials_dict: Dict[str, str]) -> BrokerageCredentials:
        """
        Load credentials from a dictionary.
        
        Args:
            credentials_dict: Dictionary containing credentials
            
        Returns:
            BrokerageCredentials object
            
        Raises:
            AuthenticationError: If required credentials are missing
        """
        logger.info("Loading credentials from dictionary")
        
        api_key = credentials_dict.get('api_key')
        api_secret = credentials_dict.get('api_secret')
        account_id = credentials_dict.get('account_id')
        
        if not api_key:
            raise AuthenticationError("Missing credential: api_key")
        
        if not api_secret:
            raise AuthenticationError("Missing credential: api_secret")
        
        # Extract any additional parameters
        additional_params = {
            k: v for k, v in credentials_dict.items()
            if k not in ['api_key', 'api_secret', 'account_id']
        }
        
        credentials = BrokerageCredentials(
            api_key=api_key,
            api_secret=api_secret,
            account_id=account_id,
            additional_params=additional_params if additional_params else None
        )
        
        credentials.validate()
        
        logger.info("Successfully loaded credentials from dictionary")
        return credentials
    
    @staticmethod
    def mask_credential(credential: str, visible_chars: int = 4) -> str:
        """
        Mask a credential for safe logging.
        
        Args:
            credential: The credential to mask
            visible_chars: Number of characters to show at the end
            
        Returns:
            Masked credential string
        """
        if not credential or len(credential) <= visible_chars:
            return "****"
        
        return "*" * (len(credential) - visible_chars) + credential[-visible_chars:]


class ConnectionStatus:
    """Tracks the connection status of a brokerage adapter."""
    
    def __init__(self):
        """Initialize connection status."""
        self.is_connected = False
        self.is_authenticated = False
        self.last_error: Optional[str] = None
        self.connection_time: Optional[float] = None
        self.account_info: Optional[Dict] = None
    
    def set_connected(self, authenticated: bool = True) -> None:
        """
        Mark connection as established.
        
        Args:
            authenticated: Whether authentication was successful
        """
        import time
        self.is_connected = True
        self.is_authenticated = authenticated
        self.connection_time = time.time()
        self.last_error = None
        logger.info("Connection status: CONNECTED and AUTHENTICATED")
    
    def set_disconnected(self, error: Optional[str] = None) -> None:
        """
        Mark connection as disconnected.
        
        Args:
            error: Optional error message
        """
        self.is_connected = False
        self.is_authenticated = False
        self.last_error = error
        self.connection_time = None
        logger.warning(f"Connection status: DISCONNECTED - {error if error else 'No error'}")
    
    def set_account_info(self, account_info: Dict) -> None:
        """
        Store account information retrieved at startup.
        
        Args:
            account_info: Dictionary containing account details
        """
        self.account_info = account_info
        logger.info(f"Account info stored: Account ID {account_info.get('account_id', 'N/A')}")
    
    def is_ready(self) -> bool:
        """
        Check if connection is ready for trading.
        
        Returns:
            True if connected and authenticated
        """
        return self.is_connected and self.is_authenticated
    
    def get_status_summary(self) -> Dict[str, any]:
        """
        Get a summary of the connection status.
        
        Returns:
            Dictionary with status information
        """
        return {
            'connected': self.is_connected,
            'authenticated': self.is_authenticated,
            'ready': self.is_ready(),
            'last_error': self.last_error,
            'connection_time': self.connection_time,
            'has_account_info': self.account_info is not None
        }
