"""Market data models."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MarketData:
    """Represents OHLCV market data for a symbol."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

    def __post_init__(self):
        """Validate market data."""
        if self.high < self.low:
            raise ValueError(f"High price ({self.high}) cannot be less than low price ({self.low})")
        if self.open < 0 or self.high < 0 or self.low < 0 or self.close < 0:
            raise ValueError("Prices cannot be negative")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")


@dataclass
class Quote:
    """Represents a real-time price quote."""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    timestamp: datetime

    def __post_init__(self):
        """Validate quote data."""
        if self.price < 0 or self.bid < 0 or self.ask < 0:
            raise ValueError("Prices cannot be negative")
        if self.ask < self.bid:
            raise ValueError(f"Ask price ({self.ask}) cannot be less than bid price ({self.bid})")
        if self.volume < 0:
            raise ValueError("Volume cannot be negative")
