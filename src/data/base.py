"""Abstract base class for market data providers."""
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Optional
import pandas as pd
from src.utils.exceptions import DataError


class Quote:
    """Represents a real-time price quote."""
    
    def __init__(
        self,
        symbol: str,
        price: float,
        bid: float,
        ask: float,
        volume: int,
        timestamp: datetime = None
    ):
        self.symbol = symbol
        self.price = price
        self.bid = bid
        self.ask = ask
        self.volume = volume
        self.timestamp = timestamp or datetime.now()
    
    def __repr__(self):
        return f"Quote({self.symbol}: ${self.price}, Vol: {self.volume})"


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """
        Retrieve current price quote for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            Quote object with current price information
            
        Raises:
            DataError: If quote cannot be retrieved
        """
        pass
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        """
        Retrieve historical OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date for historical data
            end_date: End date for historical data
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            
        Raises:
            DataError: If historical data cannot be retrieved
        """
        pass
