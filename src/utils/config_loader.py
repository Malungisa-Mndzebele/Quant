"""Configuration loading and validation utilities."""

import os
import re
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from src.models.config import (
    SystemConfig,
    TradingConfig,
    BrokerageConfig,
    DataConfig,
    RiskConfig,
    StrategyConfig,
    TradingMode
)


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


class ConfigLoader:
    """Loads and validates system configuration from YAML files."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self._env_pattern = re.compile(r'\$\{([^}]+)\}')
    
    def load(self) -> SystemConfig:
        """
        Load and validate configuration from file.
        
        Returns:
            SystemConfig: Validated system configuration
            
        Raises:
            ConfigurationError: If configuration is invalid or file not found
        """
        if not self.config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                raw_config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Failed to parse YAML configuration: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to read configuration file: {e}")
        
        if not raw_config:
            raise ConfigurationError("Configuration file is empty")
        
        # Substitute environment variables
        raw_config = self._substitute_env_vars(raw_config)
        
        # Validate and build configuration objects
        try:
            config = self._build_config(raw_config)
        except (ValueError, KeyError, TypeError) as e:
            raise ConfigurationError(f"Invalid configuration: {e}")
        
        # Perform additional validation
        self._validate_config(config)
        
        return config
    
    def _substitute_env_vars(self, obj: Any) -> Any:
        """
        Recursively substitute environment variables in configuration.
        
        Environment variables are specified as ${VAR_NAME}.
        
        Args:
            obj: Configuration object (dict, list, or primitive)
            
        Returns:
            Object with environment variables substituted
        """
        if isinstance(obj, dict):
            return {k: self._substitute_env_vars(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._substitute_env_vars(item) for item in obj]
        elif isinstance(obj, str):
            return self._substitute_env_var_string(obj)
        else:
            return obj
    
    def _substitute_env_var_string(self, value: str) -> str:
        """
        Substitute environment variables in a string.
        
        Args:
            value: String potentially containing ${VAR_NAME} patterns
            
        Returns:
            String with environment variables substituted
        """
        def replace_var(match):
            var_name = match.group(1)
            env_value = os.environ.get(var_name)
            if env_value is None:
                # Return empty string for missing env vars (will be validated later)
                return ""
            return env_value
        
        return self._env_pattern.sub(replace_var, value)
    
    def _build_config(self, raw_config: Dict[str, Any]) -> SystemConfig:
        """
        Build SystemConfig from raw configuration dictionary.
        
        Args:
            raw_config: Raw configuration dictionary from YAML
            
        Returns:
            SystemConfig: Validated configuration object
        """
        # Build trading config
        trading_data = raw_config.get('trading', {})
        trading_config = TradingConfig(
            mode=trading_data.get('mode', 'simulation'),
            initial_capital=float(trading_data.get('initial_capital', 100000)),
            symbols=trading_data.get('symbols', []),
            update_interval_seconds=trading_data.get('update_interval_seconds', 60),
            enable_logging=trading_data.get('enable_logging', True),
            log_level=trading_data.get('log_level', 'INFO')
        )
        
        # Build brokerage config
        brokerage_data = raw_config.get('brokerage', {})
        credentials = brokerage_data.get('credentials', {})
        brokerage_config = BrokerageConfig(
            provider=brokerage_data.get('provider', 'simulated'),
            api_key=credentials.get('api_key'),
            api_secret=credentials.get('api_secret'),
            account_id=brokerage_data.get('account_id'),
            base_url=brokerage_data.get('base_url'),
            timeout_seconds=brokerage_data.get('timeout_seconds', 30),
            max_retries=brokerage_data.get('max_retries', 3),
            additional_params=brokerage_data.get('additional_params', {})
        )
        
        # Build data config
        data_data = raw_config.get('data', {})
        data_config = DataConfig(
            provider=data_data.get('provider', 'yfinance'),
            api_key=data_data.get('api_key'),
            cache_enabled=data_data.get('cache_enabled', True),
            cache_dir=data_data.get('cache_dir', './data/cache'),
            cache_ttl_seconds=data_data.get('cache_ttl_seconds', 3600),
            max_retries=data_data.get('max_retries', 3)
        )
        
        # Build risk config
        risk_data = raw_config.get('risk', {})
        risk_config = RiskConfig(
            max_position_size_pct=float(risk_data.get('max_position_size_pct', 10.0)),
            max_daily_loss_pct=float(risk_data.get('max_daily_loss_pct', 5.0)),
            max_portfolio_leverage=float(risk_data.get('max_portfolio_leverage', 1.0)),
            allowed_symbols=risk_data.get('allowed_symbols'),
            max_positions=risk_data.get('max_positions'),
            max_order_value=risk_data.get('max_order_value')
        )
        
        # Build strategy configs
        strategies_data = raw_config.get('strategies', [])
        strategy_configs = []
        for strategy_data in strategies_data:
            strategy_config = StrategyConfig(
                name=strategy_data.get('name'),
                enabled=strategy_data.get('enabled', True),
                params=strategy_data.get('params', {})
            )
            strategy_configs.append(strategy_config)
        
        return SystemConfig(
            trading=trading_config,
            brokerage=brokerage_config,
            data=data_config,
            risk=risk_config,
            strategies=strategy_configs
        )
    
    def _validate_config(self, config: SystemConfig) -> None:
        """
        Perform additional validation on configuration.
        
        Args:
            config: System configuration to validate
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        # Validate brokerage credentials for live mode
        if config.is_live_mode():
            if config.brokerage.requires_credentials():
                if not config.brokerage.api_key:
                    raise ConfigurationError(
                        "Brokerage API key is required for live trading mode. "
                        "Set BROKERAGE_API_KEY environment variable."
                    )
                if not config.brokerage.api_secret:
                    raise ConfigurationError(
                        "Brokerage API secret is required for live trading mode. "
                        "Set BROKERAGE_API_SECRET environment variable."
                    )
        
        # Validate at least one strategy is enabled
        enabled_strategies = [s for s in config.strategies if s.enabled]
        if not enabled_strategies:
            raise ConfigurationError("At least one strategy must be enabled")
        
        # Validate symbols are provided if needed
        if config.trading.symbols:
            for symbol in config.trading.symbols:
                if not symbol or not isinstance(symbol, str):
                    raise ConfigurationError(f"Invalid symbol: {symbol}")
        
        # Validate data provider API key if required
        if config.data.provider.lower() == 'alphaavantage':
            if not config.data.api_key:
                raise ConfigurationError(
                    "Alpha Vantage API key is required. "
                    "Set ALPHA_VANTAGE_API_KEY environment variable or configure in data.api_key"
                )


def load_config(config_path: str = "config.yaml") -> SystemConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SystemConfig: Validated system configuration
    """
    loader = ConfigLoader(config_path)
    return loader.load()
