"""Configuration models."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class TradingMode(Enum):
    """Trading mode options."""
    SIMULATION = "simulation"
    LIVE = "live"


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_position_size_pct: float = 10.0  # % of portfolio value
    max_daily_loss_pct: float = 5.0  # % of portfolio value
    max_portfolio_leverage: float = 1.0
    allowed_symbols: Optional[List[str]] = None
    max_positions: Optional[int] = None
    max_order_value: Optional[float] = None

    def __post_init__(self):
        """Validate risk configuration."""
        if self.max_position_size_pct <= 0 or self.max_position_size_pct > 100:
            raise ValueError(f"max_position_size_pct must be between 0 and 100, got {self.max_position_size_pct}")
        
        if self.max_daily_loss_pct <= 0 or self.max_daily_loss_pct > 100:
            raise ValueError(f"max_daily_loss_pct must be between 0 and 100, got {self.max_daily_loss_pct}")
        
        if self.max_portfolio_leverage <= 0:
            raise ValueError(f"max_portfolio_leverage must be positive, got {self.max_portfolio_leverage}")
        
        if self.max_positions is not None and self.max_positions <= 0:
            raise ValueError(f"max_positions must be positive, got {self.max_positions}")
        
        if self.max_order_value is not None and self.max_order_value <= 0:
            raise ValueError(f"max_order_value must be positive, got {self.max_order_value}")


@dataclass
class TradingConfig:
    """Trading system configuration."""
    mode: TradingMode = TradingMode.SIMULATION
    initial_capital: float = 100000.0
    symbols: List[str] = field(default_factory=list)
    update_interval_seconds: int = 60
    enable_logging: bool = True
    log_level: str = "INFO"

    def __post_init__(self):
        """Validate trading configuration."""
        if isinstance(self.mode, str):
            self.mode = TradingMode(self.mode)
        
        if self.initial_capital <= 0:
            raise ValueError(f"initial_capital must be positive, got {self.initial_capital}")
        
        if self.update_interval_seconds <= 0:
            raise ValueError(f"update_interval_seconds must be positive, got {self.update_interval_seconds}")
        
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}, got {self.log_level}")


@dataclass
class BrokerageConfig:
    """Brokerage connection configuration."""
    provider: str  # e.g., "public", "moomoo", "simulated"
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    account_id: Optional[str] = None
    base_url: Optional[str] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    additional_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate brokerage configuration."""
        if not self.provider:
            raise ValueError("provider is required")
        
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be positive, got {self.timeout_seconds}")
        
        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {self.max_retries}")

    def is_simulated(self) -> bool:
        """Check if this is a simulated brokerage."""
        return self.provider.lower() in ["simulated", "simulation", "paper"]

    def requires_credentials(self) -> bool:
        """Check if this brokerage requires API credentials."""
        return not self.is_simulated()


@dataclass
class DataConfig:
    """Market data configuration."""
    provider: str = "yfinance"  # e.g., "yfinance", "alphaavantage"
    api_key: Optional[str] = None
    cache_enabled: bool = True
    cache_dir: str = "./data/cache"
    cache_ttl_seconds: int = 3600
    max_retries: int = 3

    def __post_init__(self):
        """Validate data configuration."""
        if not self.provider:
            raise ValueError("provider is required")
        
        if self.cache_ttl_seconds < 0:
            raise ValueError(f"cache_ttl_seconds cannot be negative, got {self.cache_ttl_seconds}")
        
        if self.max_retries < 0:
            raise ValueError(f"max_retries cannot be negative, got {self.max_retries}")


@dataclass
class StrategyConfig:
    """Strategy configuration."""
    name: str
    enabled: bool = True
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate strategy configuration."""
        if not self.name:
            raise ValueError("Strategy name is required")


@dataclass
class SystemConfig:
    """Complete system configuration."""
    trading: TradingConfig
    brokerage: BrokerageConfig
    data: DataConfig
    risk: RiskConfig
    strategies: List[StrategyConfig] = field(default_factory=list)

    def validate(self) -> bool:
        """Validate the entire configuration."""
        # All validation is done in __post_init__ of individual configs
        return True

    def is_live_mode(self) -> bool:
        """Check if system is in live trading mode."""
        return self.trading.mode == TradingMode.LIVE
