"""Moving Average Crossover trading strategy."""

from typing import List, Dict, Deque
from collections import deque

from .base import Strategy
from ..models.signal import Signal, SignalAction, OrderType
from ..models.market_data import MarketData
from ..models.order import Order


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover strategy.
    
    Generates BUY signals when fast MA crosses above slow MA.
    Generates SELL signals when fast MA crosses below slow MA.
    """
    
    def __init__(
        self,
        name: str = "MovingAverageCrossover",
        fast_period: int = 10,
        slow_period: int = 30,
        quantity: int = 100
    ):
        """
        Initialize Moving Average Crossover strategy.
        
        Args:
            name: Strategy name
            fast_period: Period for fast moving average
            slow_period: Period for slow moving average
            quantity: Number of shares to trade per signal
        """
        super().__init__(name, fast_period=fast_period, slow_period=slow_period, quantity=quantity)
        
        if fast_period >= slow_period:
            raise ValueError(f"Fast period ({fast_period}) must be less than slow period ({slow_period})")
        
        if fast_period < 2:
            raise ValueError(f"Fast period must be at least 2, got {fast_period}")
        
        if quantity <= 0:
            raise ValueError(f"Quantity must be positive, got {quantity}")
        
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.quantity = quantity
        
        # Price history for each symbol
        self._price_history: Dict[str, Deque[float]] = {}
        
        # Track current position for each symbol to avoid duplicate signals
        self._positions: Dict[str, int] = {}
    
    def on_data(self, market_data: MarketData) -> None:
        """
        Process new market data and update price history.
        
        Args:
            market_data: Market data containing price information
        """
        symbol = market_data.symbol
        
        # Initialize price history for new symbols
        if symbol not in self._price_history:
            # Need slow_period + 1 to calculate previous MA for crossover detection
            self._price_history[symbol] = deque(maxlen=self.slow_period + 1)
            self._positions[symbol] = 0
        
        # Add new price to history
        self._price_history[symbol].append(market_data.close)
    
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on moving average crossover.
        
        Returns:
            List of Signal objects
        """
        signals = []
        
        for symbol, prices in self._price_history.items():
            # Need enough data for crossover detection (slow_period + 1 for previous MA)
            if len(prices) < self.slow_period + 1:
                continue
            
            # Calculate current moving averages
            fast_ma = self._calculate_ma(prices, self.fast_period)
            slow_ma = self._calculate_ma(prices, self.slow_period)
            
            # Calculate previous MAs
            prev_prices = list(prices)[:-1]
            prev_fast_ma = self._calculate_ma(prev_prices, self.fast_period)
            prev_slow_ma = self._calculate_ma(prev_prices, self.slow_period)
            
            current_position = self._positions[symbol]
            
            # Detect crossover
            # BUY: fast MA crosses above slow MA (and we don't have a position)
            if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma and current_position == 0:
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    quantity=self.quantity,
                    order_type=OrderType.MARKET
                )
                signals.append(signal)
                # Update position tracking (will be confirmed in on_order_filled)
                self._positions[symbol] = self.quantity
            
            # SELL: fast MA crosses below slow MA (and we have a position)
            elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma and current_position > 0:
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    quantity=current_position,
                    order_type=OrderType.MARKET
                )
                signals.append(signal)
                # Update position tracking (will be confirmed in on_order_filled)
                self._positions[symbol] = 0
        
        return signals
    
    def on_order_filled(self, order: Order) -> None:
        """
        Update position tracking when order is filled.
        
        Args:
            order: Order that was filled
        """
        symbol = order.symbol
        
        if symbol not in self._positions:
            self._positions[symbol] = 0
        
        # Update position based on order action
        if order.action.value == "BUY":
            self._positions[symbol] += order.filled_quantity
        elif order.action.value == "SELL":
            self._positions[symbol] -= order.filled_quantity
            # Ensure position doesn't go negative
            if self._positions[symbol] < 0:
                self._positions[symbol] = 0
    
    def _calculate_ma(self, prices: List[float], period: int) -> float:
        """
        Calculate simple moving average.
        
        Args:
            prices: List of prices
            period: Number of periods for MA
            
        Returns:
            Moving average value
        """
        if len(prices) < period:
            raise ValueError(f"Not enough data for MA calculation: need {period}, have {len(prices)}")
        
        # Use last 'period' prices
        recent_prices = list(prices)[-period:]
        return sum(recent_prices) / period
    
    def get_current_position(self, symbol: str) -> int:
        """
        Get current position for a symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Current position quantity
        """
        return self._positions.get(symbol, 0)
    
    def reset_positions(self) -> None:
        """Reset all position tracking."""
        self._positions.clear()
    
    def reset_history(self) -> None:
        """Reset all price history."""
        self._price_history.clear()
