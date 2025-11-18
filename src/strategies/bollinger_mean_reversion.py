"""
Bollinger Bands Mean Reversion Strategy

This strategy identifies overbought and oversold conditions using Bollinger Bands
and RSI, then trades on mean reversion principles.

Strategy Logic:
- BUY when price touches lower Bollinger Band AND RSI < 30 (oversold)
- SELL when price touches upper Bollinger Band AND RSI > 70 (overbought)
- Exit positions when price crosses middle band (mean)

Parameters:
- bb_period: Bollinger Bands period (default: 20)
- bb_std: Standard deviations for bands (default: 2.0)
- rsi_period: RSI calculation period (default: 14)
- rsi_oversold: RSI oversold threshold (default: 30)
- rsi_overbought: RSI overbought threshold (default: 70)
"""

import logging
from typing import List, Dict
from collections import deque
import statistics

from src.strategies.base import Strategy
from src.models.signal import Signal, SignalAction
from src.models.market_data import MarketData

logger = logging.getLogger(__name__)


class BollingerMeanReversionStrategy(Strategy):
    """
    Mean reversion strategy using Bollinger Bands and RSI.
    
    Buys when price is oversold (touches lower band + low RSI)
    Sells when price is overbought (touches upper band + high RSI)
    """
    
    def __init__(
        self,
        symbols: List[str],
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        quantity: int = 100
    ):
        """
        Initialize Bollinger Mean Reversion strategy.
        
        Args:
            symbols: List of symbols to trade
            bb_period: Period for Bollinger Bands calculation
            bb_std: Number of standard deviations for bands
            rsi_period: Period for RSI calculation
            rsi_oversold: RSI threshold for oversold condition
            rsi_overbought: RSI threshold for overbought condition
            quantity: Number of shares to trade per signal
        """
        super().__init__(name="BollingerMeanReversion")
        
        # Validate parameters
        if bb_period < 2:
            raise ValueError("Bollinger Bands period must be at least 2")
        if bb_std <= 0:
            raise ValueError("Standard deviation must be positive")
        if rsi_period < 2:
            raise ValueError("RSI period must be at least 2")
        if not (0 < rsi_oversold < rsi_overbought < 100):
            raise ValueError("RSI thresholds must be: 0 < oversold < overbought < 100")
        if quantity <= 0:
            raise ValueError("Quantity must be positive")
        
        self.symbols = symbols
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.quantity = quantity
        
        # Price history for each symbol
        self.price_history: Dict[str, deque] = {
            symbol: deque(maxlen=max(bb_period, rsi_period) + 1)
            for symbol in symbols
        }
        
        # Track positions
        self.positions: Dict[str, bool] = {symbol: False for symbol in symbols}
        
        logger.info(
            f"Initialized {self.name} strategy: "
            f"BB({bb_period}, {bb_std}), RSI({rsi_period}), "
            f"Thresholds({rsi_oversold}/{rsi_overbought})"
        )
    
    def on_data(self, market_data: MarketData) -> None:
        """
        Process new market data.
        
        Args:
            market_data: New market data point
        """
        symbol = market_data.symbol
        
        if symbol not in self.price_history:
            return
        
        # Store closing price
        self.price_history[symbol].append(market_data.close)
    
    def _calculate_sma(self, prices: List[float], period: int) -> float:
        """Calculate Simple Moving Average."""
        if len(prices) < period:
            return None
        return statistics.mean(prices[-period:])
    
    def _calculate_std(self, prices: List[float], period: int) -> float:
        """Calculate Standard Deviation."""
        if len(prices) < period:
            return None
        return statistics.stdev(prices[-period:])
    
    def _calculate_bollinger_bands(self, prices: List[float]) -> tuple:
        """
        Calculate Bollinger Bands.
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band) or (None, None, None)
        """
        if len(prices) < self.bb_period:
            return None, None, None
        
        middle = self._calculate_sma(prices, self.bb_period)
        std = self._calculate_std(prices, self.bb_period)
        
        if middle is None or std is None:
            return None, None, None
        
        upper = middle + (self.bb_std * std)
        lower = middle - (self.bb_std * std)
        
        return upper, middle, lower
    
    def _calculate_rsi(self, prices: List[float]) -> float:
        """
        Calculate Relative Strength Index (RSI).
        
        Returns:
            RSI value (0-100) or None if insufficient data
        """
        if len(prices) < self.rsi_period + 1:
            return None
        
        # Calculate price changes
        changes = [prices[i] - prices[i-1] for i in range(-self.rsi_period, 0)]
        
        # Separate gains and losses
        gains = [max(change, 0) for change in changes]
        losses = [abs(min(change, 0)) for change in changes]
        
        # Calculate average gain and loss
        avg_gain = statistics.mean(gains)
        avg_loss = statistics.mean(losses)
        
        # Avoid division by zero
        if avg_loss == 0:
            return 100.0
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self) -> List[Signal]:
        """
        Generate trading signals based on Bollinger Bands and RSI.
        
        Returns:
            List of trading signals
        """
        signals = []
        
        for symbol in self.symbols:
            prices = list(self.price_history[symbol])
            
            # Need enough data
            if len(prices) < max(self.bb_period, self.rsi_period + 1):
                continue
            
            # Calculate indicators
            upper_band, middle_band, lower_band = self._calculate_bollinger_bands(prices)
            rsi = self._calculate_rsi(prices)
            
            if None in [upper_band, middle_band, lower_band, rsi]:
                continue
            
            current_price = prices[-1]
            has_position = self.positions[symbol]
            
            # BUY Signal: Price at/below lower band AND RSI oversold
            if not has_position:
                if current_price <= lower_band and rsi < self.rsi_oversold:
                    signal = Signal(
                        symbol=symbol,
                        action=SignalAction.BUY,
                        quantity=self.quantity,
                        order_type="MARKET",
                        reason=f"Mean reversion BUY: Price={current_price:.2f}, "
                               f"Lower Band={lower_band:.2f}, RSI={rsi:.1f}"
                    )
                    signals.append(signal)
                    logger.info(f"Generated BUY signal for {symbol}: {signal.reason}")
            
            # SELL Signal: Price at/above upper band AND RSI overbought
            elif has_position:
                # Exit on overbought
                if current_price >= upper_band and rsi > self.rsi_overbought:
                    signal = Signal(
                        symbol=symbol,
                        action=SignalAction.SELL,
                        quantity=self.quantity,
                        order_type="MARKET",
                        reason=f"Mean reversion SELL: Price={current_price:.2f}, "
                               f"Upper Band={upper_band:.2f}, RSI={rsi:.1f}"
                    )
                    signals.append(signal)
                    logger.info(f"Generated SELL signal for {symbol}: {signal.reason}")
                
                # Exit on mean reversion (price crosses middle band from below)
                elif len(prices) >= 2:
                    prev_price = prices[-2]
                    if prev_price < middle_band and current_price >= middle_band:
                        signal = Signal(
                            symbol=symbol,
                            action=SignalAction.SELL,
                            quantity=self.quantity,
                            order_type="MARKET",
                            reason=f"Mean reversion exit: Price crossed middle band "
                                   f"({current_price:.2f} >= {middle_band:.2f})"
                        )
                        signals.append(signal)
                        logger.info(f"Generated SELL signal for {symbol}: {signal.reason}")
        
        return signals
    
    def on_order_filled(self, order) -> None:
        """
        Update strategy state when order is filled.
        
        Args:
            order: Filled order object
        """
        symbol = order.symbol
        
        if symbol not in self.positions:
            return
        
        if order.action == "BUY":
            self.positions[symbol] = True
            logger.info(f"Position opened for {symbol}")
        elif order.action == "SELL":
            self.positions[symbol] = False
            logger.info(f"Position closed for {symbol}")
    
    def reset_state(self) -> None:
        """Reset strategy state for backtesting."""
        for symbol in self.symbols:
            self.price_history[symbol].clear()
            self.positions[symbol] = False
        
        logger.debug(f"{self.name} strategy state reset")
    
    def get_strategy_info(self) -> Dict:
        """Get strategy configuration and current state."""
        return {
            'name': self.name,
            'symbols': self.symbols,
            'parameters': {
                'bb_period': self.bb_period,
                'bb_std': self.bb_std,
                'rsi_period': self.rsi_period,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought,
                'quantity': self.quantity
            },
            'positions': self.positions.copy()
        }
