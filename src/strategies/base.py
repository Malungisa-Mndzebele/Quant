"""Abstract base class for trading strategies."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime

from ..models.signal import Signal
from ..models.market_data import MarketData
from ..models.order import Order


class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str, **params):
        """
        Initialize strategy with name and parameters.
        
        Args:
            name: Strategy name
            **params: Strategy-specific parameters
        """
        self.name = name
        self.params = params
        self._state: Dict[str, Any] = {}
    
    @abstractmethod
    def on_data(self, market_data: MarketData) -> None:
        """
        Called when new market data arrives.
        
        Args:
            market_data: Market data object containing price information
        """
        pass
    
    @abstractmethod
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on current market conditions.
        
        Returns:
            List of Signal objects
        """
        pass
    
    def on_order_filled(self, order: Order) -> None:
        """
        Callback when an order is executed.
        
        Args:
            order: Order object that was filled
        """
        pass
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """
        Get strategy state value.
        
        Args:
            key: State key
            default: Default value if key not found
            
        Returns:
            State value or default
        """
        return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """
        Set strategy state value.
        
        Args:
            key: State key
            value: State value
        """
        self._state[key] = value
    
    def reset_state(self) -> None:
        """Reset strategy state."""
        self._state.clear()


class StrategyManager:
    """Manages multiple trading strategies."""
    
    def __init__(self):
        """Initialize strategy manager."""
        self._strategies: List[Strategy] = []
    
    def register_strategy(self, strategy: Strategy) -> None:
        """
        Register a strategy for execution.
        
        Args:
            strategy: Strategy instance to register
        """
        if not isinstance(strategy, Strategy):
            raise TypeError(f"Expected Strategy instance, got {type(strategy)}")
        
        # Check for duplicate names
        if any(s.name == strategy.name for s in self._strategies):
            raise ValueError(f"Strategy with name '{strategy.name}' already registered")
        
        self._strategies.append(strategy)
    
    def unregister_strategy(self, strategy_name: str) -> bool:
        """
        Unregister a strategy by name.
        
        Args:
            strategy_name: Name of strategy to unregister
            
        Returns:
            True if strategy was found and removed, False otherwise
        """
        for i, strategy in enumerate(self._strategies):
            if strategy.name == strategy_name:
                self._strategies.pop(i)
                return True
        return False
    
    def get_strategy(self, strategy_name: str) -> Strategy:
        """
        Get a strategy by name.
        
        Args:
            strategy_name: Name of strategy to retrieve
            
        Returns:
            Strategy instance
            
        Raises:
            ValueError: If strategy not found
        """
        for strategy in self._strategies:
            if strategy.name == strategy_name:
                return strategy
        raise ValueError(f"Strategy '{strategy_name}' not found")
    
    def run_strategies(self, market_data: MarketData) -> None:
        """
        Execute all registered strategies with new market data.
        
        Args:
            market_data: Market data to process
        """
        for strategy in self._strategies:
            try:
                strategy.on_data(market_data)
            except Exception as e:
                # Log error but continue with other strategies
                print(f"Error in strategy '{strategy.name}' on_data: {e}")
    
    def get_signals(self) -> List[Signal]:
        """
        Collect signals from all registered strategies.
        
        Returns:
            List of all signals from all strategies
        """
        all_signals = []
        for strategy in self._strategies:
            try:
                signals = strategy.generate_signals()
                if signals:
                    all_signals.extend(signals)
            except Exception as e:
                # Log error but continue with other strategies
                print(f"Error in strategy '{strategy.name}' generate_signals: {e}")
        return all_signals
    
    def notify_order_filled(self, order: Order) -> None:
        """
        Notify all strategies that an order was filled.
        
        Args:
            order: Order that was filled
        """
        for strategy in self._strategies:
            try:
                strategy.on_order_filled(order)
            except Exception as e:
                # Log error but continue with other strategies
                print(f"Error in strategy '{strategy.name}' on_order_filled: {e}")
    
    @property
    def strategies(self) -> List[Strategy]:
        """Get list of registered strategies."""
        return self._strategies.copy()
    
    def clear(self) -> None:
        """Remove all registered strategies."""
        self._strategies.clear()
