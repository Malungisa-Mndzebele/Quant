"""
Custom Strategy Template

This template provides a starting point for creating your own trading strategies.
Copy this file and modify it to implement your trading logic.
"""

from typing import List, Dict
from src.strategies.base import Strategy
from src.models.signal import Signal, SignalAction, OrderType
from src.models.market_data import MarketData


class CustomStrategyTemplate(Strategy):
    """
    Template for creating custom trading strategies.
    
    This strategy demonstrates the basic structure and methods you need to implement.
    Replace the logic in each method with your own trading algorithm.
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        threshold: float = 0.02,
        position_size: int = 100
    ):
        """
        Initialize the strategy with parameters.
        
        Args:
            lookback_period: Number of periods to look back for analysis
            threshold: Threshold for signal generation (e.g., 2% price change)
            position_size: Number of shares to trade per signal
        """
        super().__init__(name="CustomStrategyTemplate")
        
        # Strategy parameters
        self.lookback_period = lookback_period
        self.threshold = threshold
        self.position_size = position_size
        
        # Data storage
        self.price_history: Dict[str, List[float]] = {}
        self.volume_history: Dict[str, List[int]] = {}
        
        # Position tracking
        self.current_positions: Dict[str, int] = {}
    
    def on_data(self, market_data: MarketData) -> None:
        """
        Process incoming market data.
        
        This method is called every time new market data arrives.
        Use it to update your indicators, store historical data, etc.
        
        Args:
            market_data: New market data for a symbol
        """
        symbol = market_data.symbol
        
        # Initialize storage for new symbols
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.volume_history[symbol] = []
            self.current_positions[symbol] = 0
        
        # Store price and volume data
        self.price_history[symbol].append(market_data.close)
        self.volume_history[symbol].append(market_data.volume)
        
        # Keep only recent data (memory management)
        max_history = self.lookback_period * 2
        if len(self.price_history[symbol]) > max_history:
            self.price_history[symbol] = self.price_history[symbol][-max_history:]
            self.volume_history[symbol] = self.volume_history[symbol][-max_history:]
    
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on your strategy logic.
        
        This method is called after on_data() to generate trading signals.
        Implement your trading logic here.
        
        Returns:
            List of Signal objects (BUY, SELL, or HOLD)
        """
        signals = []
        
        for symbol in self.price_history.keys():
            # Skip if not enough data
            if len(self.price_history[symbol]) < self.lookback_period:
                continue
            
            # Get recent prices
            recent_prices = self.price_history[symbol][-self.lookback_period:]
            current_price = recent_prices[-1]
            
            # Example Strategy Logic: Simple Momentum
            # Calculate price change over lookback period
            price_change = (current_price - recent_prices[0]) / recent_prices[0]
            
            # Generate BUY signal if price increased above threshold
            if price_change > self.threshold and self.current_positions[symbol] == 0:
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    quantity=self.position_size,
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    timestamp=None  # Will be set by system
                )
                signals.append(signal)
                self.logger.info(
                    f"{symbol}: BUY signal - Price change: {price_change:.2%}"
                )
            
            # Generate SELL signal if price decreased below threshold
            elif price_change < -self.threshold and self.current_positions[symbol] > 0:
                signal = Signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    quantity=self.current_positions[symbol],
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    timestamp=None
                )
                signals.append(signal)
                self.logger.info(
                    f"{symbol}: SELL signal - Price change: {price_change:.2%}"
                )
        
        return signals
    
    def on_order_filled(self, order) -> None:
        """
        Handle order fill notifications.
        
        This method is called when an order is filled.
        Use it to update your strategy state, position tracking, etc.
        
        Args:
            order: Filled Order object
        """
        from src.models.order import OrderAction
        
        symbol = order.symbol
        
        # Update position tracking
        if order.action == OrderAction.BUY:
            self.current_positions[symbol] = self.current_positions.get(symbol, 0) + order.quantity
            self.logger.info(
                f"{symbol}: Position opened - {order.quantity} shares @ ${order.filled_price:.2f}"
            )
        
        elif order.action == OrderAction.SELL:
            self.current_positions[symbol] = self.current_positions.get(symbol, 0) - order.quantity
            self.logger.info(
                f"{symbol}: Position closed - {order.quantity} shares @ ${order.filled_price:.2f}"
            )
    
    def reset_state(self) -> None:
        """
        Reset strategy state.
        
        This method is called before backtesting to ensure clean state.
        Clear all stored data and reset position tracking.
        """
        self.price_history = {}
        self.volume_history = {}
        self.current_positions = {}
        self.logger.info("Strategy state reset")


# ============================================================================
# Advanced Example: Mean Reversion Strategy
# ============================================================================

class MeanReversionStrategy(Strategy):
    """
    Mean reversion strategy that trades when price deviates from moving average.
    
    This strategy:
    1. Calculates a simple moving average
    2. Measures standard deviation of prices
    3. Buys when price is below MA - (threshold * std dev)
    4. Sells when price is above MA + (threshold * std dev)
    """
    
    def __init__(
        self,
        ma_period: int = 20,
        std_dev_threshold: float = 2.0,
        position_size: int = 100
    ):
        super().__init__(name="MeanReversion")
        self.ma_period = ma_period
        self.std_dev_threshold = std_dev_threshold
        self.position_size = position_size
        self.price_history: Dict[str, List[float]] = {}
        self.positions: Dict[str, int] = {}
    
    def on_data(self, market_data: MarketData) -> None:
        symbol = market_data.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            self.positions[symbol] = 0
        
        self.price_history[symbol].append(market_data.close)
        
        # Keep only necessary history
        if len(self.price_history[symbol]) > self.ma_period * 2:
            self.price_history[symbol] = self.price_history[symbol][-self.ma_period * 2:]
    
    def generate_signals(self) -> List[Signal]:
        signals = []
        
        for symbol, prices in self.price_history.items():
            if len(prices) < self.ma_period:
                continue
            
            # Calculate indicators
            recent_prices = prices[-self.ma_period:]
            ma = sum(recent_prices) / len(recent_prices)
            
            # Calculate standard deviation
            variance = sum((p - ma) ** 2 for p in recent_prices) / len(recent_prices)
            std_dev = variance ** 0.5
            
            current_price = prices[-1]
            
            # Calculate z-score (how many std devs from mean)
            z_score = (current_price - ma) / std_dev if std_dev > 0 else 0
            
            # Buy when price is significantly below mean
            if z_score < -self.std_dev_threshold and self.positions[symbol] == 0:
                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.BUY,
                    quantity=self.position_size,
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    timestamp=None
                ))
                self.logger.info(
                    f"{symbol}: BUY - Price ${current_price:.2f} is {abs(z_score):.2f} "
                    f"std devs below MA ${ma:.2f}"
                )
            
            # Sell when price is significantly above mean
            elif z_score > self.std_dev_threshold and self.positions[symbol] > 0:
                signals.append(Signal(
                    symbol=symbol,
                    action=SignalAction.SELL,
                    quantity=self.positions[symbol],
                    order_type=OrderType.MARKET,
                    limit_price=None,
                    timestamp=None
                ))
                self.logger.info(
                    f"{symbol}: SELL - Price ${current_price:.2f} is {z_score:.2f} "
                    f"std devs above MA ${ma:.2f}"
                )
        
        return signals
    
    def on_order_filled(self, order) -> None:
        from src.models.order import OrderAction
        
        if order.action == OrderAction.BUY:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) + order.quantity
        elif order.action == OrderAction.SELL:
            self.positions[order.symbol] = self.positions.get(order.symbol, 0) - order.quantity
    
    def reset_state(self) -> None:
        self.price_history = {}
        self.positions = {}


# ============================================================================
# Usage Instructions
# ============================================================================

"""
To use your custom strategy:

1. Copy this template and rename it (e.g., my_strategy.py)

2. Implement your logic in the three main methods:
   - on_data(): Process incoming market data
   - generate_signals(): Generate BUY/SELL signals
   - on_order_filled(): Update state when orders fill

3. Register your strategy in src/trading_system.py:
   
   def _create_strategy(self, strategy_config):
       if strategy_name == 'mycustomstrategy':
           return MyCustomStrategy(
               param1=strategy_config.params.get('param1', default_value)
           )

4. Add to config.yaml:
   
   strategies:
     - name: "MyCustomStrategy"
       enabled: true
       params:
         param1: value1
         param2: value2

5. Test with backtest:
   
   python main.py backtest \
     --start-date 2024-01-01 \
     --end-date 2024-12-31 \
     --strategy MyCustomStrategy

6. Run in simulation:
   
   python main.py run

Tips:
- Start simple and add complexity gradually
- Test thoroughly with backtests before live trading
- Use logging (self.logger.info/debug) for debugging
- Consider transaction costs in your logic
- Implement proper position sizing
- Add stop-loss logic for risk management
"""
