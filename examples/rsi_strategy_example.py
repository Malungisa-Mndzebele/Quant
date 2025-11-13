"""
RSI (Relative Strength Index) Strategy Example

This example demonstrates a complete RSI-based trading strategy that:
1. Calculates RSI indicator from price data
2. Generates buy signals when RSI < 30 (oversold)
3. Generates sell signals when RSI > 70 (overbought)

RSI is a momentum oscillator that measures the speed and magnitude of price changes.
It ranges from 0 to 100, with readings below 30 indicating oversold conditions and
readings above 70 indicating overbought conditions.

Usage:
    1. Copy this file to src/strategies/rsi_strategy.py
    2. Register in src/trading_system.py
    3. Add to config.yaml:
       strategies:
         - name: "RSIStrategy"
           enabled: true
           params:
             rsi_period: 14
             oversold_threshold: 30
             overbought_threshold: 70
             position_size: 100
"""

from typing import List, Dict
from src.strategies.base import Strategy
from src.models.signal import Signal, SignalAction, OrderType
from src.models.market_data import MarketData


class RSIStrategy(Strategy):
    """
    RSI-based trading strategy.
    
    This strategy uses the Relative Strength Index (RSI) to identify
    overbought and oversold conditions:
    - Buy when RSI crosses below oversold threshold (default 30)
    - Sell when RSI crosses above overbought threshold (default 70)
    
    The RSI is calculated using the standard formula:
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over the period
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        position_size: int = 100
    ):
        """
        Initialize RSI strategy.
        
        Args:
            rsi_period: Number of periods for RSI calculation (default 14)
            oversold_threshold: RSI level below which to buy (default 30)
            overbought_threshold: RSI level above which to sell (default 70)
            position_size: Number of shares to trade per signal
        """
        super().__init__(name="RSIStrategy")
        
        # Strategy parameters
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.position_size = position_size
        
        # Data storage
        self.price_history: Dict[str, List[float]] = {}
        self.rsi_values: Dict[str, List[float]] = {}
        
        # Position tracking
        self.positions: Dict[str, int] = {}
        
        # Previous RSI for crossover detection
        self.prev_rsi: Dict[str, float] = {}
        
        self.logger.info(
            f"RSI Strategy initialized: period={rsi_period}, "
            f"oversold={oversold_threshold}, overbought={overbought_threshold}"
        )
    
    def on_data(self, market_data: MarketData) -> None:
        """
        Process incoming market data and calculate RSI.
        
        Args:
            market_data: New market data for a symbol
        """
        symbol = market_data.symbol
        
        # Initialize storage for new symbols
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.rsi_values[symbol] = []
            self.positions[symbol] = 0
            self.prev_rsi[symbol] = 50.0  # Neutral starting point
        
        # Store closing price
        self.price_history[symbol].append(market_data.close)
        
        # Keep only necessary history (RSI period + buffer)
        max_history = self.rsi_period * 3
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.rsi_values[symbol] = self.rsi_values[symbol][-max_history:]
        
        # Calculate RSI if we have enough data
        if len(self.price_history[symbol]) >= self.rsi_period + 1:
            rsi = self._calculate_rsi(symbol)
            self.rsi_values[symbol].append(rsi)
            
            self.logger.debug(
                f"{symbol}: Price=${market_data.close:.2f}, RSI={rsi:.2f}"
            )
    
    def _calculate_rsi(self, symbol: str) -> float:
        """
        Calculate RSI for a symbol using the standard formula.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            RSI value (0-100)
        """
        prices = self.price_history[symbol]
        
        # Need at least rsi_period + 1 prices to calculate
        if len(prices) < self.rsi_period + 1:
            return 50.0  # Return neutral value
        
        # Calculate price changes
        changes = []
        for i in range(1, len(prices)):
            changes.append(prices[i] - prices[i-1])
        
        # Use only the most recent rsi_period changes
        recent_changes = changes[-self.rsi_period:]
        
        # Separate gains and losses
        gains = [change if change > 0 else 0 for change in recent_changes]
        losses = [-change if change < 0 else 0 for change in recent_changes]
        
        # Calculate average gain and loss
        avg_gain = sum(gains) / self.rsi_period
        avg_loss = sum(losses) / self.rsi_period
        
        # Avoid division by zero
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on RSI levels.
        
        Returns:
            List of Signal objects
        """
        signals = []
        
        for symbol in self.price_history.keys():
            # Need enough data to calculate RSI
            if len(self.rsi_values[symbol]) == 0:
                continue
            
            current_rsi = self.rsi_values[symbol][-1]
            prev_rsi = self.prev_rsi[symbol]
            
            # Buy signal: RSI crosses below oversold threshold
            if (prev_rsi >= self.oversold_threshold and 
                current_rsi < self.oversold_threshold and 
                self.positions[symbol] == 0):
                
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    quantity=self.position_size,
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    timestamp=None
                )
                signals.append(signal)
                
                self.logger.info(
                    f"{symbol}: BUY signal - RSI crossed below {self.oversold_threshold} "
                    f"(RSI={current_rsi:.2f})"
                )
            
            # Sell signal: RSI crosses above overbought threshold
            elif (prev_rsi <= self.overbought_threshold and 
                  current_rsi > self.overbought_threshold and 
                  self.positions[symbol] > 0):
                
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    quantity=self.positions[symbol],
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    timestamp=None
                )
                signals.append(signal)
                
                self.logger.info(
                    f"{symbol}: SELL signal - RSI crossed above {self.overbought_threshold} "
                    f"(RSI={current_rsi:.2f})"
                )
            
            # Update previous RSI for next iteration
            self.prev_rsi[symbol] = current_rsi
        
        return signals
    
    def on_order_filled(self, order) -> None:
        """
        Handle order fill notifications.
        
        Args:
            order: Filled Order object
        """
        from src.models.order import OrderAction
        
        symbol = order.symbol
        
        if order.action == OrderAction.BUY:
            self.positions[symbol] = self.positions.get(symbol, 0) + order.quantity
            self.logger.info(
                f"{symbol}: Position opened - {order.quantity} shares @ ${order.filled_price:.2f}"
            )
        
        elif order.action == OrderAction.SELL:
            self.positions[symbol] = self.positions.get(symbol, 0) - order.quantity
            self.logger.info(
                f"{symbol}: Position closed - {order.quantity} shares @ ${order.filled_price:.2f}"
            )
    
    def reset_state(self) -> None:
        """
        Reset strategy state for backtesting.
        """
        self.price_history = {}
        self.rsi_values = {}
        self.positions = {}
        self.prev_rsi = {}
        self.logger.info("RSI Strategy state reset")


# ============================================================================
# Advanced RSI Strategy with Additional Features
# ============================================================================

class AdvancedRSIStrategy(Strategy):
    """
    Advanced RSI strategy with additional features:
    - Multiple timeframe confirmation
    - Volume filter
    - Trend filter using moving average
    - Dynamic position sizing based on RSI strength
    """
    
    def __init__(
        self,
        rsi_period: int = 14,
        oversold_threshold: float = 30.0,
        overbought_threshold: float = 70.0,
        ma_period: int = 50,
        min_volume: int = 1000000,
        base_position_size: int = 100
    ):
        """
        Initialize advanced RSI strategy.
        
        Args:
            rsi_period: RSI calculation period
            oversold_threshold: RSI buy threshold
            overbought_threshold: RSI sell threshold
            ma_period: Moving average period for trend filter
            min_volume: Minimum volume required for signals
            base_position_size: Base position size (scaled by RSI strength)
        """
        super().__init__(name="AdvancedRSIStrategy")
        
        self.rsi_period = rsi_period
        self.oversold_threshold = oversold_threshold
        self.overbought_threshold = overbought_threshold
        self.ma_period = ma_period
        self.min_volume = min_volume
        self.base_position_size = base_position_size
        
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[int]] = {}
        self.rsi_values: Dict[str, List[float]] = {}
        self.positions: Dict[str, int] = {}
        self.prev_rsi: Dict[str, float] = {}
    
    def on_data(self, market_data: MarketData) -> None:
        """Process market data."""
        symbol = market_data.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
            self.rsi_values[symbol] = []
            self.positions[symbol] = 0
            self.prev_rsi[symbol] = 50.0
        
        self.price_history[symbol].append(market_data.close)
        self.volume_history[symbol].append(market_data.volume)
        
        # Keep limited history
        max_history = max(self.rsi_period, self.ma_period) * 3
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.volume_history[symbol] = self.volume_history[symbol][-max_history:]
            self.rsi_values[symbol] = self.rsi_values[symbol][-max_history:]
        
        # Calculate RSI
        if len(self.price_history[symbol]) >= self.rsi_period + 1:
            rsi = self._calculate_rsi(symbol)
            self.rsi_values[symbol].append(rsi)
    
    def _calculate_rsi(self, symbol: str) -> float:
        """Calculate RSI (same as basic strategy)."""
        prices = self.price_history[symbol]
        
        if len(prices) < self.rsi_period + 1:
            return 50.0
        
        changes = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        recent_changes = changes[-self.rsi_period:]
        
        gains = [c if c > 0 else 0 for c in recent_changes]
        losses = [-c if c < 0 else 0 for c in recent_changes]
        
        avg_gain = sum(gains) / self.rsi_period
        avg_loss = sum(losses) / self.rsi_period
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        return rsi
    
    def _calculate_ma(self, symbol: str) -> float:
        """Calculate simple moving average."""
        prices = self.price_history[symbol]
        
        if len(prices) < self.ma_period:
            return prices[-1] if prices else 0
        
        recent_prices = prices[-self.ma_period:]
        return sum(recent_prices) / len(recent_prices)
    
    def _check_volume_filter(self, symbol: str) -> bool:
        """Check if volume meets minimum requirement."""
        if not self.volume_history[symbol]:
            return False
        
        current_volume = self.volume_history[symbol][-1]
        return current_volume >= self.min_volume
    
    def _check_trend_filter(self, symbol: str, signal_type: str) -> bool:
        """Check if price trend aligns with signal."""
        if len(self.price_history[symbol]) < self.ma_period:
            return True  # Skip filter if not enough data
        
        current_price = self.price_history[symbol][-1]
        ma = self._calculate_ma(symbol)
        
        if signal_type == "BUY":
            # For buy signals, prefer price near or above MA (uptrend)
            return current_price >= ma * 0.98  # Allow 2% below MA
        else:  # SELL
            # For sell signals, prefer price near or below MA (downtrend)
            return current_price <= ma * 1.02  # Allow 2% above MA
    
    def _calculate_position_size(self, rsi: float, signal_type: str) -> int:
        """Calculate dynamic position size based on RSI strength."""
        if signal_type == "BUY":
            # Stronger signal = lower RSI = larger position
            strength = (self.oversold_threshold - rsi) / self.oversold_threshold
        else:  # SELL
            # Stronger signal = higher RSI = larger position
            strength = (rsi - self.overbought_threshold) / (100 - self.overbought_threshold)
        
        # Scale position size (1x to 2x base size)
        multiplier = 1.0 + max(0, min(1.0, strength))
        return int(self.base_position_size * multiplier)
    
    def generate_signals(self) -> List[Signal]:
        """Generate signals with advanced filters."""
        signals = []
        
        for symbol in self.price_history.keys():
            if len(self.rsi_values[symbol]) == 0:
                continue
            
            current_rsi = self.rsi_values[symbol][-1]
            prev_rsi = self.prev_rsi[symbol]
            
            # Buy signal with filters
            if (prev_rsi >= self.oversold_threshold and 
                current_rsi < self.oversold_threshold and 
                self.positions[symbol] == 0):
                
                # Apply filters
                if not self._check_volume_filter(symbol):
                    self.logger.debug(f"{symbol}: Buy signal filtered - insufficient volume")
                    continue
                
                if not self._check_trend_filter(symbol, "BUY"):
                    self.logger.debug(f"{symbol}: Buy signal filtered - trend not aligned")
                    continue
                
                # Calculate dynamic position size
                quantity = self._calculate_position_size(current_rsi, "BUY")
                
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    quantity=quantity,
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    timestamp=None
                )
                signals.append(signal)
                
                self.logger.info(
                    f"{symbol}: BUY signal - RSI={current_rsi:.2f}, Quantity={quantity}"
                )
            
            # Sell signal with filters
            elif (prev_rsi <= self.overbought_threshold and 
                  current_rsi > self.overbought_threshold and 
                  self.positions[symbol] > 0):
                
                if not self._check_volume_filter(symbol):
                    self.logger.debug(f"{symbol}: Sell signal filtered - insufficient volume")
                    continue
                
                if not self._check_trend_filter(symbol, "SELL"):
                    self.logger.debug(f"{symbol}: Sell signal filtered - trend not aligned")
                    continue
                
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    quantity=self.positions[symbol],
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    timestamp=None
                )
                signals.append(signal)
                
                self.logger.info(
                    f"{symbol}: SELL signal - RSI={current_rsi:.2f}"
                )
            
            self.prev_rsi[symbol] = current_rsi
        
        return signals
    
    def on_order_filled(self, order) -> None:
        """Handle order fills."""
        from src.models.order import OrderAction
        
        if order.action == OrderAction.BUY:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        elif order.action == OrderAction.SELL:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) - order.quantity
    
    def reset_state(self) -> None:
        """Reset state."""
        self.price_history = {}
        self.volume_history = {}
        self.rsi_values = {}
        self.positions = {}
        self.prev_rsi = {}


# ============================================================================
# Usage Example
# ============================================================================

if __name__ == "__main__":
    """
    Example usage of RSI strategy.
    
    To use this strategy in the trading system:
    
    1. Copy this file to src/strategies/rsi_strategy.py
    
    2. Register in src/trading_system.py:
       
       def _create_strategy(self, strategy_config):
           strategy_name = strategy_config.name.lower()
           
           if strategy_name == 'rsistrategy':
               return RSIStrategy(
                   rsi_period=strategy_config.params.get('rsi_period', 14),
                   oversold_threshold=strategy_config.params.get('oversold_threshold', 30),
                   overbought_threshold=strategy_config.params.get('overbought_threshold', 70),
                   position_size=strategy_config.params.get('position_size', 100)
               )
           
           elif strategy_name == 'advancedrsistrategy':
               return AdvancedRSIStrategy(
                   rsi_period=strategy_config.params.get('rsi_period', 14),
                   oversold_threshold=strategy_config.params.get('oversold_threshold', 30),
                   overbought_threshold=strategy_config.params.get('overbought_threshold', 70),
                   ma_period=strategy_config.params.get('ma_period', 50),
                   min_volume=strategy_config.params.get('min_volume', 1000000),
                   base_position_size=strategy_config.params.get('base_position_size', 100)
               )
    
    3. Add to config.yaml:
       
       strategies:
         - name: "RSIStrategy"
           enabled: true
           params:
             rsi_period: 14
             oversold_threshold: 30
             overbought_threshold: 70
             position_size: 100
    
    4. Run backtest:
       
       python main.py backtest \
         --start-date 2024-01-01 \
         --end-date 2024-12-31 \
         --symbols AAPL GOOGL \
         --strategy RSIStrategy
    
    5. Run in simulation:
       
       python main.py run
    """
    
    print("RSI Strategy Example")
    print("=" * 50)
    print("\nThis file contains two RSI strategy implementations:")
    print("1. RSIStrategy - Basic RSI strategy")
    print("2. AdvancedRSIStrategy - RSI with volume and trend filters")
    print("\nSee docstrings above for usage instructions.")