"""YFinance market data provider implementation."""
import logging
from datetime import date, datetime
import pandas as pd
import yfinance as yf

from src.data.base import MarketDataProvider, Quote
from src.utils.exceptions import DataError

logger = logging.getLogger(__name__)


class YFinanceProvider(MarketDataProvider):
    """Market data provider using Yahoo Finance API."""
    
    def __init__(self):
        """Initialize YFinance provider."""
        logger.info("Initialized YFinanceProvider")
    
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
        try:
            logger.debug(f"Fetching quote for {symbol}")
            ticker = yf.Ticker(symbol)
            
            # Get current market data
            info = ticker.info
            
            # Check if we got valid data
            if not info or ('regularMarketPrice' not in info and 'currentPrice' not in info):
                raise DataError(f"No quote data available for symbol: {symbol}")
            
            # Extract quote information
            price = info.get('regularMarketPrice', info.get('currentPrice', 0))
            bid = info.get('bid', price)
            ask = info.get('ask', price)
            volume = info.get('regularMarketVolume', info.get('volume', 0))
            
            if price == 0:
                raise DataError(f"Invalid price data for symbol: {symbol}")
            
            quote = Quote(
                symbol=symbol,
                price=float(price),
                bid=float(bid) if bid else float(price),
                ask=float(ask) if ask else float(price),
                volume=int(volume) if volume else 0,
                timestamp=datetime.now()
            )
            
            logger.debug(f"Retrieved quote for {symbol}: ${quote.price}")
            return quote
            
        except DataError:
            raise
        except Exception as e:
            error_msg = f"Failed to retrieve quote for {symbol}: {str(e)}"
            logger.error(error_msg)
            raise DataError(error_msg) from e
    
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
        try:
            logger.debug(f"Fetching historical data for {symbol} from {start_date} to {end_date}")
            
            # Validate date range
            if start_date > end_date:
                raise DataError(f"Start date ({start_date}) cannot be after end date ({end_date})")
            
            # Download historical data
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            # Check if we got valid data
            if df is None or df.empty:
                raise DataError(f"No historical data available for {symbol} between {start_date} and {end_date}")
            
            # Ensure we have the required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise DataError(f"Missing required columns in historical data: {missing_columns}")
            
            # Select only the required columns
            df = df[required_columns].copy()
            
            # Remove any rows with NaN values
            df = df.dropna()
            
            if df.empty:
                raise DataError(f"No valid historical data for {symbol} after cleaning")
            
            logger.info(f"Retrieved {len(df)} rows of historical data for {symbol}")
            return df
            
        except DataError:
            raise
        except Exception as e:
            error_msg = f"Failed to retrieve historical data for {symbol}: {str(e)}"
            logger.error(error_msg)
            raise DataError(error_msg) from e
