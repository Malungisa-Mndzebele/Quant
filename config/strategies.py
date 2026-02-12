"""
Trading strategy configuration and customization.

This module provides strategy parameter schemas, presets, and validation
for customizing AI trading behavior according to user preferences.
"""

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum


class TradingStyle(Enum):
    """Trading style presets"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class IndicatorType(Enum):
    """Available technical indicators"""
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    MACD = "macd"
    BOLLINGER_BANDS = "bollinger_bands"
    ATR = "atr"


@dataclass
class IndicatorConfig:
    """Configuration for a technical indicator"""
    indicator_type: str  # IndicatorType value
    enabled: bool = True
    weight: float = 1.0  # Weight in signal generation (0.0 to 1.0)
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate indicator configuration"""
        # Validate indicator type
        valid_types = [ind.value for ind in IndicatorType]
        if self.indicator_type not in valid_types:
            raise ValueError(f"Invalid indicator type: {self.indicator_type}")
        
        # Validate weight
        if not 0.0 <= self.weight <= 1.0:
            raise ValueError(f"Indicator weight must be between 0.0 and 1.0, got {self.weight}")


@dataclass
class StrategyParameters:
    """
    Trading strategy parameters.
    
    These parameters control how the AI generates trading signals and
    executes trades.
    """
    # Strategy identification
    name: str = "Default Strategy"
    style: str = TradingStyle.MODERATE.value
    
    # Signal generation parameters
    min_confidence: float = 0.6  # Minimum confidence to generate signal (0.0 to 1.0)
    signal_threshold: float = 0.7  # Threshold for strong signals (0.0 to 1.0)
    
    # Model weights (must sum to 1.0)
    lstm_weight: float = 0.4  # Weight for LSTM predictions
    rf_weight: float = 0.4  # Weight for Random Forest predictions
    sentiment_weight: float = 0.2  # Weight for sentiment analysis
    
    # Risk parameters
    max_position_size: float = 0.1  # Max % of portfolio per position
    max_portfolio_risk: float = 0.02  # Max % of portfolio at risk
    stop_loss_pct: float = 0.05  # Stop loss percentage
    take_profit_pct: float = 0.10  # Take profit percentage
    
    # Trading behavior
    max_trades_per_day: int = 10  # Maximum trades per day
    min_holding_period: int = 1  # Minimum holding period in days
    max_holding_period: int = 30  # Maximum holding period in days
    
    # Technical indicators
    indicators: List[IndicatorConfig] = field(default_factory=list)
    
    # Advanced parameters
    use_trailing_stop: bool = False  # Use trailing stop loss
    trailing_stop_pct: float = 0.03  # Trailing stop percentage
    rebalance_frequency: int = 7  # Days between portfolio rebalancing
    
    def __post_init__(self):
        """Initialize default indicators if none provided"""
        if not self.indicators:
            self.indicators = self._get_default_indicators()
    
    def _get_default_indicators(self) -> List[IndicatorConfig]:
        """Get default indicator configuration"""
        return [
            IndicatorConfig(
                indicator_type=IndicatorType.SMA.value,
                enabled=True,
                weight=0.8,
                parameters={"period": 20}
            ),
            IndicatorConfig(
                indicator_type=IndicatorType.EMA.value,
                enabled=True,
                weight=0.9,
                parameters={"period": 12}
            ),
            IndicatorConfig(
                indicator_type=IndicatorType.RSI.value,
                enabled=True,
                weight=1.0,
                parameters={"period": 14}
            ),
            IndicatorConfig(
                indicator_type=IndicatorType.MACD.value,
                enabled=True,
                weight=0.9,
                parameters={"fast_period": 12, "slow_period": 26, "signal_period": 9}
            ),
            IndicatorConfig(
                indicator_type=IndicatorType.BOLLINGER_BANDS.value,
                enabled=True,
                weight=0.7,
                parameters={"period": 20, "num_std": 2.0}
            ),
            IndicatorConfig(
                indicator_type=IndicatorType.ATR.value,
                enabled=True,
                weight=0.6,
                parameters={"period": 14}
            )
        ]
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate strategy parameters.
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Validate confidence thresholds
        if not 0.0 <= self.min_confidence <= 1.0:
            errors.append(f"min_confidence must be between 0.0 and 1.0, got {self.min_confidence}")
        if not 0.0 <= self.signal_threshold <= 1.0:
            errors.append(f"signal_threshold must be between 0.0 and 1.0, got {self.signal_threshold}")
        if self.min_confidence > self.signal_threshold:
            errors.append("min_confidence cannot be greater than signal_threshold")
        
        # Validate model weights
        total_weight = self.lstm_weight + self.rf_weight + self.sentiment_weight
        if not 0.99 <= total_weight <= 1.01:  # Allow small floating point error
            errors.append(f"Model weights must sum to 1.0, got {total_weight}")
        if not 0.0 <= self.lstm_weight <= 1.0:
            errors.append(f"lstm_weight must be between 0.0 and 1.0, got {self.lstm_weight}")
        if not 0.0 <= self.rf_weight <= 1.0:
            errors.append(f"rf_weight must be between 0.0 and 1.0, got {self.rf_weight}")
        if not 0.0 <= self.sentiment_weight <= 1.0:
            errors.append(f"sentiment_weight must be between 0.0 and 1.0, got {self.sentiment_weight}")
        
        # Validate risk parameters
        if not 0.0 < self.max_position_size <= 1.0:
            errors.append(f"max_position_size must be between 0.0 and 1.0, got {self.max_position_size}")
        if not 0.0 < self.max_portfolio_risk <= 1.0:
            errors.append(f"max_portfolio_risk must be between 0.0 and 1.0, got {self.max_portfolio_risk}")
        if not 0.0 < self.stop_loss_pct < 1.0:
            errors.append(f"stop_loss_pct must be between 0.0 and 1.0, got {self.stop_loss_pct}")
        if not 0.0 < self.take_profit_pct < 1.0:
            errors.append(f"take_profit_pct must be between 0.0 and 1.0, got {self.take_profit_pct}")
        
        # Validate trading behavior
        if self.max_trades_per_day <= 0:
            errors.append(f"max_trades_per_day must be positive, got {self.max_trades_per_day}")
        if self.min_holding_period <= 0:
            errors.append(f"min_holding_period must be positive, got {self.min_holding_period}")
        if self.max_holding_period <= 0:
            errors.append(f"max_holding_period must be positive, got {self.max_holding_period}")
        if self.min_holding_period > self.max_holding_period:
            errors.append("min_holding_period cannot be greater than max_holding_period")
        
        # Validate trailing stop
        if self.use_trailing_stop:
            if not 0.0 < self.trailing_stop_pct < 1.0:
                errors.append(f"trailing_stop_pct must be between 0.0 and 1.0, got {self.trailing_stop_pct}")
        
        # Validate rebalance frequency
        if self.rebalance_frequency <= 0:
            errors.append(f"rebalance_frequency must be positive, got {self.rebalance_frequency}")
        
        # Validate indicators
        for i, indicator in enumerate(self.indicators):
            try:
                # Validate indicator weight
                if not 0.0 <= indicator.weight <= 1.0:
                    errors.append(f"Indicator {i} weight must be between 0.0 and 1.0")
            except Exception as e:
                errors.append(f"Indicator {i} validation error: {str(e)}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert strategy to dictionary"""
        data = asdict(self)
        # Convert indicator configs to dicts
        data['indicators'] = [asdict(ind) for ind in self.indicators]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyParameters':
        """Create strategy from dictionary"""
        # Convert indicator dicts to IndicatorConfig objects
        if 'indicators' in data:
            data['indicators'] = [
                IndicatorConfig(**ind) if isinstance(ind, dict) else ind
                for ind in data['indicators']
            ]
        return cls(**data)


class StrategyPresets:
    """Predefined strategy presets for different trading styles"""
    
    @staticmethod
    def conservative() -> StrategyParameters:
        """
        Conservative strategy preset.
        
        Characteristics:
        - High confidence threshold
        - Lower position sizes
        - Tighter stop losses
        - Fewer trades
        - Focus on capital preservation
        """
        return StrategyParameters(
            name="Conservative Strategy",
            style=TradingStyle.CONSERVATIVE.value,
            min_confidence=0.75,
            signal_threshold=0.85,
            lstm_weight=0.3,
            rf_weight=0.5,
            sentiment_weight=0.2,
            max_position_size=0.05,  # 5% per position
            max_portfolio_risk=0.01,  # 1% portfolio risk
            stop_loss_pct=0.03,  # 3% stop loss
            take_profit_pct=0.08,  # 8% take profit
            max_trades_per_day=5,
            min_holding_period=3,
            max_holding_period=60,
            use_trailing_stop=True,
            trailing_stop_pct=0.02,
            rebalance_frequency=14
        )
    
    @staticmethod
    def moderate() -> StrategyParameters:
        """
        Moderate strategy preset.
        
        Characteristics:
        - Balanced confidence threshold
        - Moderate position sizes
        - Standard stop losses
        - Moderate trading frequency
        - Balance between growth and safety
        """
        return StrategyParameters(
            name="Moderate Strategy",
            style=TradingStyle.MODERATE.value,
            min_confidence=0.6,
            signal_threshold=0.7,
            lstm_weight=0.4,
            rf_weight=0.4,
            sentiment_weight=0.2,
            max_position_size=0.1,  # 10% per position
            max_portfolio_risk=0.02,  # 2% portfolio risk
            stop_loss_pct=0.05,  # 5% stop loss
            take_profit_pct=0.10,  # 10% take profit
            max_trades_per_day=10,
            min_holding_period=1,
            max_holding_period=30,
            use_trailing_stop=False,
            trailing_stop_pct=0.03,
            rebalance_frequency=7
        )
    
    @staticmethod
    def aggressive() -> StrategyParameters:
        """
        Aggressive strategy preset.
        
        Characteristics:
        - Lower confidence threshold
        - Larger position sizes
        - Wider stop losses
        - More frequent trading
        - Focus on maximizing returns
        """
        return StrategyParameters(
            name="Aggressive Strategy",
            style=TradingStyle.AGGRESSIVE.value,
            min_confidence=0.5,
            signal_threshold=0.6,
            lstm_weight=0.5,
            rf_weight=0.3,
            sentiment_weight=0.2,
            max_position_size=0.15,  # 15% per position
            max_portfolio_risk=0.03,  # 3% portfolio risk
            stop_loss_pct=0.08,  # 8% stop loss
            take_profit_pct=0.15,  # 15% take profit
            max_trades_per_day=20,
            min_holding_period=1,  # At least 1 day
            max_holding_period=14,
            use_trailing_stop=False,
            trailing_stop_pct=0.05,
            rebalance_frequency=3
        )
    
    @staticmethod
    def get_preset(style: TradingStyle) -> StrategyParameters:
        """
        Get strategy preset by style.
        
        Args:
            style: Trading style enum
            
        Returns:
            Strategy parameters for the specified style
        """
        if style == TradingStyle.CONSERVATIVE:
            return StrategyPresets.conservative()
        elif style == TradingStyle.MODERATE:
            return StrategyPresets.moderate()
        elif style == TradingStyle.AGGRESSIVE:
            return StrategyPresets.aggressive()
        else:
            return StrategyPresets.moderate()  # Default to moderate


class StrategyManager:
    """Manager for saving, loading, and managing trading strategies"""
    
    def __init__(self, strategies_dir: Path = Path("data/strategies")):
        """
        Initialize strategy manager.
        
        Args:
            strategies_dir: Directory for storing strategy files
        """
        self.strategies_dir = strategies_dir
        self.strategies_dir.mkdir(parents=True, exist_ok=True)
    
    def save_strategy(self, strategy: StrategyParameters, filename: Optional[str] = None) -> bool:
        """
        Save strategy to file.
        
        Args:
            strategy: Strategy parameters to save
            filename: Optional filename (defaults to strategy name)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate strategy before saving
            is_valid, errors = strategy.validate()
            if not is_valid:
                print(f"Strategy validation failed: {errors}")
                return False
            
            # Generate filename from strategy name if not provided
            if filename is None:
                # Sanitize strategy name for filename
                filename = strategy.name.lower().replace(" ", "_") + ".json"
            
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.strategies_dir / filename
            
            # Convert to dict and save
            with open(filepath, 'w') as f:
                json.dump(strategy.to_dict(), f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error saving strategy: {e}")
            return False
    
    def load_strategy(self, filename: str) -> Optional[StrategyParameters]:
        """
        Load strategy from file.
        
        Args:
            filename: Name of strategy file
            
        Returns:
            Strategy parameters, or None if error occurs
        """
        try:
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.strategies_dir / filename
            
            if not filepath.exists():
                print(f"Strategy file not found: {filepath}")
                return None
            
            # Load and parse
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            strategy = StrategyParameters.from_dict(data)
            
            # Validate loaded strategy
            is_valid, errors = strategy.validate()
            if not is_valid:
                print(f"Loaded strategy validation failed: {errors}")
                return None
            
            return strategy
        except Exception as e:
            print(f"Error loading strategy: {e}")
            return None
    
    def list_strategies(self) -> List[str]:
        """
        List all saved strategies.
        
        Returns:
            List of strategy filenames (without .json extension)
        """
        try:
            strategy_files = list(self.strategies_dir.glob("*.json"))
            return [f.stem for f in strategy_files]
        except Exception as e:
            print(f"Error listing strategies: {e}")
            return []
    
    def delete_strategy(self, filename: str) -> bool:
        """
        Delete a saved strategy.
        
        Args:
            filename: Name of strategy file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure .json extension
            if not filename.endswith('.json'):
                filename += '.json'
            
            filepath = self.strategies_dir / filename
            
            if not filepath.exists():
                print(f"Strategy file not found: {filepath}")
                return False
            
            filepath.unlink()
            return True
        except Exception as e:
            print(f"Error deleting strategy: {e}")
            return False
    
    def get_preset_strategies(self) -> Dict[str, StrategyParameters]:
        """
        Get all preset strategies.
        
        Returns:
            Dictionary mapping style names to strategy parameters
        """
        return {
            "conservative": StrategyPresets.conservative(),
            "moderate": StrategyPresets.moderate(),
            "aggressive": StrategyPresets.aggressive()
        }


# Global strategy manager instance
strategy_manager = StrategyManager()


# Convenience functions
def get_strategy_preset(style: str) -> StrategyParameters:
    """
    Get a strategy preset by name.
    
    Args:
        style: Strategy style name ('conservative', 'moderate', 'aggressive')
        
    Returns:
        Strategy parameters
    """
    style_enum = TradingStyle(style.lower())
    return StrategyPresets.get_preset(style_enum)


def create_custom_strategy(
    name: str,
    base_style: str = "moderate",
    **overrides
) -> StrategyParameters:
    """
    Create a custom strategy based on a preset with overrides.
    
    Args:
        name: Name for the custom strategy
        base_style: Base preset to start from ('conservative', 'moderate', 'aggressive')
        **overrides: Parameter overrides
        
    Returns:
        Custom strategy parameters
        
    Example:
        strategy = create_custom_strategy(
            "My Strategy",
            base_style="moderate",
            min_confidence=0.7,
            max_position_size=0.08
        )
    """
    # Get base preset
    base_strategy = get_strategy_preset(base_style)
    
    # Convert to dict and apply overrides
    strategy_dict = base_strategy.to_dict()
    strategy_dict['name'] = name
    strategy_dict['style'] = TradingStyle.CUSTOM.value
    strategy_dict.update(overrides)
    
    # Auto-adjust signal_threshold if min_confidence is overridden
    # and signal_threshold would be invalid
    if 'min_confidence' in overrides and 'signal_threshold' not in overrides:
        min_conf = overrides['min_confidence']
        if min_conf > strategy_dict['signal_threshold']:
            # Set signal_threshold slightly higher than min_confidence
            strategy_dict['signal_threshold'] = min(min_conf + 0.1, 1.0)
    
    # Create and validate
    strategy = StrategyParameters.from_dict(strategy_dict)
    is_valid, errors = strategy.validate()
    if not is_valid:
        raise ValueError(f"Invalid strategy parameters: {errors}")
    
    return strategy
