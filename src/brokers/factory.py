"""Factory for creating brokerage adapters."""

import logging
from typing import Dict, Type, Optional

from .base import BrokerageAdapter
from .simulated_adapter import SimulatedBrokerageAdapter
from .credentials import CredentialManager, BrokerageCredentials
from .errors import BrokerageError
from ..models.config import BrokerageConfig, TradingMode


logger = logging.getLogger(__name__)


class BrokerageFactory:
    """
    Factory for creating brokerage adapter instances.
    
    Supports pluggable adapter registration and mode switching with safety checks.
    """
    
    # Registry of available brokerage adapters
    _adapters: Dict[str, Type[BrokerageAdapter]] = {
        'simulated': SimulatedBrokerageAdapter,
        'simulation': SimulatedBrokerageAdapter,
        'paper': SimulatedBrokerageAdapter,
    }
    
    @classmethod
    def _register_default_adapters(cls):
        """Register default brokerage adapters."""
        try:
            from .robinhood_adapter import RobinhoodAdapter
            cls._adapters['robinhood'] = RobinhoodAdapter
        except ImportError:
            logger.debug("Robinhood adapter not available (robin_stocks not installed)")
        
        try:
            from .public_adapter import PublicBrokerageAdapter
            cls._adapters['public'] = PublicBrokerageAdapter
        except ImportError:
            logger.debug("Public adapter not available")
        
        try:
            from .moomoo_adapter import MoomooBrokerageAdapter
            cls._adapters['moomoo'] = MoomooBrokerageAdapter
        except ImportError:
            logger.debug("Moomoo adapter not available")
    
    @classmethod
    def register_adapter(cls, provider: str, adapter_class: Type[BrokerageAdapter]) -> None:
        """
        Register a new brokerage adapter.
        
        Args:
            provider: Provider name (e.g., 'public', 'moomoo')
            adapter_class: Adapter class to register
        """
        provider_lower = provider.lower()
        if provider_lower in cls._adapters:
            logger.warning(f"Overwriting existing adapter for provider: {provider}")
        
        cls._adapters[provider_lower] = adapter_class
        logger.info(f"Registered brokerage adapter: {provider} -> {adapter_class.__name__}")
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Get list of available brokerage providers.
        
        Returns:
            List of provider names
        """
        # Ensure default adapters are registered
        cls._register_default_adapters()
        return list(cls._adapters.keys())
    
    @classmethod
    def create_adapter(
        cls,
        config: BrokerageConfig,
        trading_mode: TradingMode,
        initial_capital: Optional[float] = None
    ) -> BrokerageAdapter:
        """
        Create a brokerage adapter based on configuration.
        
        Args:
            config: Brokerage configuration
            trading_mode: Trading mode (simulation or live)
            initial_capital: Initial capital for simulated accounts
            
        Returns:
            Configured brokerage adapter instance
            
        Raises:
            BrokerageError: If adapter creation fails
        """
        # Validate configuration
        cls._validate_config(config, trading_mode)
        
        provider_lower = config.provider.lower()
        
        # Check if provider is registered
        if provider_lower not in cls._adapters:
            raise BrokerageError(
                f"Unknown brokerage provider: {config.provider}. "
                f"Available providers: {', '.join(cls.get_available_providers())}"
            )
        
        adapter_class = cls._adapters[provider_lower]
        
        # Create adapter instance
        try:
            if provider_lower in ['simulated', 'simulation', 'paper']:
                # Create simulated adapter
                capital = initial_capital or 100000.0
                adapter = adapter_class(initial_capital=capital)
                logger.info(
                    f"Created simulated brokerage adapter with ${capital:,.2f} initial capital"
                )
            else:
                # Create real brokerage adapter
                # Load credentials
                credentials = cls._load_credentials(config)
                
                # Instantiate adapter (real adapters will need credentials in constructor)
                adapter = adapter_class(config=config, credentials=credentials)
                logger.info(f"Created {config.provider} brokerage adapter")
            
            return adapter
            
        except Exception as e:
            error_msg = f"Failed to create brokerage adapter for {config.provider}: {str(e)}"
            logger.error(error_msg)
            raise BrokerageError(error_msg)
    
    @classmethod
    def _validate_config(cls, config: BrokerageConfig, trading_mode: TradingMode) -> None:
        """
        Validate brokerage configuration with safety checks.
        
        Args:
            config: Brokerage configuration
            trading_mode: Trading mode
            
        Raises:
            BrokerageError: If configuration is invalid
        """
        # Safety check: prevent live trading with simulated broker
        if trading_mode == TradingMode.LIVE and config.is_simulated():
            raise BrokerageError(
                "Cannot use simulated brokerage in LIVE trading mode. "
                "Please configure a real brokerage provider or switch to SIMULATION mode."
            )
        
        # Safety check: warn if using real broker in simulation mode
        if trading_mode == TradingMode.SIMULATION and not config.is_simulated():
            logger.warning(
                f"Using real brokerage provider '{config.provider}' in SIMULATION mode. "
                f"Consider using 'simulated' provider for paper trading."
            )
        
        # Check credentials for real brokerages
        if config.requires_credentials():
            if not config.api_key and not config.api_secret:
                raise BrokerageError(
                    f"Brokerage provider '{config.provider}' requires API credentials. "
                    f"Please provide api_key and api_secret in configuration or environment variables."
                )
        
        logger.debug(f"Configuration validated for {config.provider} in {trading_mode.value} mode")
    
    @classmethod
    def _load_credentials(cls, config: BrokerageConfig) -> BrokerageCredentials:
        """
        Load credentials from configuration or environment.
        
        Args:
            config: Brokerage configuration
            
        Returns:
            BrokerageCredentials object
            
        Raises:
            BrokerageError: If credentials cannot be loaded
        """
        try:
            # Try loading from config first
            if config.api_key and config.api_secret:
                credentials_dict = {
                    'api_key': config.api_key,
                    'api_secret': config.api_secret,
                    'account_id': config.account_id,
                }
                # Add any additional parameters
                credentials_dict.update(config.additional_params)
                
                return CredentialManager.load_from_dict(credentials_dict)
            else:
                # Load from environment variables
                return CredentialManager.load_from_env(config.provider)
                
        except Exception as e:
            raise BrokerageError(f"Failed to load credentials: {str(e)}")
    
    @classmethod
    def validate_mode_switch(
        cls,
        current_mode: TradingMode,
        new_mode: TradingMode,
        config: BrokerageConfig
    ) -> tuple[bool, str]:
        """
        Validate if mode switching is safe.
        
        Args:
            current_mode: Current trading mode
            new_mode: Requested new mode
            config: Brokerage configuration
            
        Returns:
            Tuple of (is_valid, message)
        """
        # Switching to same mode is always valid
        if current_mode == new_mode:
            return True, "Already in requested mode"
        
        # Switching from simulation to live
        if current_mode == TradingMode.SIMULATION and new_mode == TradingMode.LIVE:
            if config.is_simulated():
                return False, (
                    "Cannot switch to LIVE mode with simulated brokerage. "
                    "Please configure a real brokerage provider first."
                )
            
            return True, (
                "WARNING: Switching to LIVE mode will execute real trades with real money. "
                "Ensure you have tested your strategies thoroughly in simulation mode."
            )
        
        # Switching from live to simulation
        if current_mode == TradingMode.LIVE and new_mode == TradingMode.SIMULATION:
            return True, (
                "Switching to SIMULATION mode. "
                "No real trades will be executed until you switch back to LIVE mode."
            )
        
        return True, "Mode switch validated"
