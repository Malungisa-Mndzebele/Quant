"""Market data service for fetching real-time and historical market data from Alpaca API."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, AsyncIterator, Callable, Set
from dataclasses import dataclass
from functools import lru_cache
from enum import Enum

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import (
    StockLatestQuoteRequest, StockBarsRequest,
    CryptoLatestQuoteRequest, CryptoBarsRequest
)
from alpaca.data.timeframe import TimeFrame
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetCalendarRequest

from config.settings import settings

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Supported asset classes"""
    STOCK = "stock"
    CRYPTO = "crypto"
    FOREX = "forex"


@dataclass
class Quote:
    """Real-time quote data"""
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    timestamp: datetime
    asset_class: AssetClass = AssetClass.STOCK
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price between bid and ask"""
        return (self.bid_price + self.ask_price) / 2


@dataclass
class MarketStatus:
    """Market status information"""
    is_open: bool
    next_open: Optional[datetime]
    next_close: Optional[datetime]
    current_time: datetime


class MarketDataService:
    """
    Service for fetching market data from Alpaca API.
    
    Provides real-time quotes, historical bars, and market status information
    with built-in caching and error handling.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
        paper: bool = True
    ):
        """
        Initialize market data service.
        
        Args:
            api_key: Alpaca API key (defaults to settings)
            api_secret: Alpaca API secret (defaults to settings)
            paper: Whether to use paper trading endpoint
        """
        self.api_key = api_key or settings.alpaca.api_key
        self.api_secret = api_secret or settings.alpaca.secret_key
        self.paper = paper
        
        # Validate credentials
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials are required")
        
        # Initialize clients for different asset classes
        try:
            # Stock data client
            self.stock_data_client = StockHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
            
            # Crypto data client
            self.crypto_data_client = CryptoHistoricalDataClient(
                api_key=self.api_key,
                secret_key=self.api_secret
            )
            
            # Trading client
            self.trading_client = TradingClient(
                api_key=self.api_key,
                secret_key=self.api_secret,
                paper=self.paper
            )
            
            # Legacy reference for backward compatibility
            self.data_client = self.stock_data_client
            
            logger.info(f"Initialized Alpaca API connection (paper={self.paper})")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca API: {e}")
            raise
        
        # WebSocket stream clients (initialized lazily)
        self._stock_stream_client: Optional[StockDataStream] = None
        self._crypto_stream_client: Optional[CryptoDataStream] = None
        self._stream_subscriptions: Dict[AssetClass, Set[str]] = {
            AssetClass.STOCK: set(),
            AssetClass.CRYPTO: set()
        }
        self._stream_callbacks: Dict[str, List[Callable]] = {}
        self._stream_connected: Dict[AssetClass, bool] = {
            AssetClass.STOCK: False,
            AssetClass.CRYPTO: False
        }
        self._stream_reconnect_attempts: int = 0
        self._max_reconnect_attempts: int = 5
        self._reconnect_delay: float = 2.0
        
        # Cache for frequently accessed data
        self._quote_cache: Dict[str, tuple[Quote, float]] = {}
        self._bars_cache: Dict[str, tuple[pd.DataFrame, float]] = {}
        self._market_status_cache: Optional[tuple[MarketStatus, float]] = None
        
        # Cache TTL in seconds
        self.cache_ttl = settings.cache_ttl_seconds
        
        # Rate limiting
        self._last_request_time = 0
        self._min_request_interval = 0.2  # 200ms between requests
        
        # Retry configuration
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds
    
    def _rate_limit(self):
        """Enforce rate limiting between API requests"""
        current_time = time.time()
        time_since_last = current_time - self._last_request_time
        
        if time_since_last < self._min_request_interval:
            sleep_time = self._min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self._last_request_time = time.time()
    
    def _is_cache_valid(self, cached_time: float) -> bool:
        """Check if cached data is still valid"""
        return time.time() - cached_time < self.cache_ttl
    
    def _retry_on_failure(self, func, *args, **kwargs):
        """
        Retry a function call with exponential backoff on failure.
        
        Args:
            func: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self._max_retries):
            try:
                self._rate_limit()
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                logger.warning(
                    f"API request failed (attempt {attempt + 1}/{self._max_retries}): {e}"
                )
                
                if attempt < self._max_retries - 1:
                    # Exponential backoff
                    delay = self._retry_delay * (2 ** attempt)
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
        
        # All retries failed
        logger.error(f"All retry attempts failed: {last_exception}")
        raise last_exception
    
    def detect_asset_class(self, symbol: str) -> AssetClass:
        """
        Detect asset class from symbol format.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            AssetClass enum value
            
        Examples:
            'AAPL' -> AssetClass.STOCK
            'BTC/USD' -> AssetClass.CRYPTO
            'BTCUSD' -> AssetClass.CRYPTO
            'EUR/USD' -> AssetClass.FOREX
        """
        symbol = symbol.upper()
        
        # Crypto patterns
        crypto_pairs = ['BTC', 'ETH', 'LTC', 'BCH', 'XRP', 'DOGE', 'SHIB', 'AVAX', 'MATIC']
        
        # Check for crypto pairs (e.g., BTC/USD, BTCUSD)
        if '/' in symbol:
            base = symbol.split('/')[0]
            if base in crypto_pairs:
                return AssetClass.CRYPTO
            # Forex pairs (e.g., EUR/USD)
            if len(base) == 3 and base.isalpha():
                return AssetClass.FOREX
        else:
            # Check if symbol starts with crypto ticker
            for crypto in crypto_pairs:
                if symbol.startswith(crypto):
                    return AssetClass.CRYPTO
        
        # Default to stock
        return AssetClass.STOCK
    
    def get_latest_quote(
        self, 
        symbol: str, 
        use_cache: bool = True,
        asset_class: Optional[AssetClass] = None
    ) -> Quote:
        """
        Get the latest quote for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC/USD')
            use_cache: Whether to use cached data if available
            asset_class: Asset class (auto-detected if not provided)
            
        Returns:
            Quote object with latest bid/ask prices
            
        Raises:
            ValueError: If symbol is invalid
            Exception: If API request fails
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.upper()
        
        # Detect asset class if not provided
        if asset_class is None:
            asset_class = self.detect_asset_class(symbol)
        
        # Check cache
        cache_key = f"{asset_class.value}:{symbol}"
        if use_cache and cache_key in self._quote_cache:
            cached_quote, cached_time = self._quote_cache[cache_key]
            if self._is_cache_valid(cached_time):
                logger.debug(f"Using cached quote for {symbol} ({asset_class.value})")
                return cached_quote
        
        # Fetch from API based on asset class
        logger.info(f"Fetching latest quote for {symbol} ({asset_class.value})")
        
        def fetch_stock_quote():
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            response = self.stock_data_client.get_stock_latest_quote(request)
            
            if symbol not in response:
                raise ValueError(f"No quote data available for {symbol}")
            
            quote_data = response[symbol]
            
            return Quote(
                symbol=symbol,
                bid_price=float(quote_data.bid_price),
                ask_price=float(quote_data.ask_price),
                bid_size=int(quote_data.bid_size),
                ask_size=int(quote_data.ask_size),
                timestamp=quote_data.timestamp,
                asset_class=AssetClass.STOCK
            )
        
        def fetch_crypto_quote():
            request = CryptoLatestQuoteRequest(symbol_or_symbols=symbol)
            response = self.crypto_data_client.get_crypto_latest_quote(request)
            
            if symbol not in response:
                raise ValueError(f"No quote data available for {symbol}")
            
            quote_data = response[symbol]
            
            return Quote(
                symbol=symbol,
                bid_price=float(quote_data.bid_price),
                ask_price=float(quote_data.ask_price),
                bid_size=int(quote_data.bid_size),
                ask_size=int(quote_data.ask_size),
                timestamp=quote_data.timestamp,
                asset_class=AssetClass.CRYPTO
            )
        
        try:
            if asset_class == AssetClass.STOCK:
                quote = self._retry_on_failure(fetch_stock_quote)
            elif asset_class == AssetClass.CRYPTO:
                quote = self._retry_on_failure(fetch_crypto_quote)
            elif asset_class == AssetClass.FOREX:
                # Forex not yet supported by Alpaca data API
                raise NotImplementedError("Forex quotes not yet supported")
            else:
                raise ValueError(f"Unsupported asset class: {asset_class}")
            
            # Update cache
            self._quote_cache[cache_key] = (quote, time.time())
            
            return quote
        except Exception as e:
            logger.error(f"Failed to fetch quote for {symbol}: {e}")
            raise
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str = '1Day',
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
        use_cache: bool = True,
        asset_class: Optional[AssetClass] = None
    ) -> pd.DataFrame:
        """
        Get historical OHLCV bars for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC/USD')
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1Hour', '1Day')
            start: Start datetime (defaults to 30 days ago)
            end: End datetime (defaults to now)
            limit: Maximum number of bars to return
            use_cache: Whether to use cached data if available
            asset_class: Asset class (auto-detected if not provided)
            
        Returns:
            DataFrame with columns: open, high, low, close, volume, timestamp
            
        Raises:
            ValueError: If parameters are invalid
            Exception: If API request fails
        """
        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string")
        
        symbol = symbol.upper()
        
        # Detect asset class if not provided
        if asset_class is None:
            asset_class = self.detect_asset_class(symbol)
        
        # Set default date range
        if end is None:
            end = datetime.now()
        if start is None:
            start = end - timedelta(days=30)
        
        # Validate date range
        if start >= end:
            raise ValueError("Start date must be before end date")
        
        # Create cache key
        cache_key = f"{asset_class.value}:{symbol}_{timeframe}_{start.isoformat()}_{end.isoformat()}_{limit}"
        
        # Check cache
        if use_cache and cache_key in self._bars_cache:
            cached_bars, cached_time = self._bars_cache[cache_key]
            if self._is_cache_valid(cached_time):
                logger.debug(f"Using cached bars for {symbol}")
                return cached_bars.copy()
        
        # Map timeframe string to TimeFrame enum
        timeframe_map = {
            '1Min': TimeFrame.Minute,
            '5Min': TimeFrame(5, 'Min'),
            '15Min': TimeFrame(15, 'Min'),
            '1Hour': TimeFrame.Hour,
            '1Day': TimeFrame.Day,
        }
        
        if timeframe not in timeframe_map:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Must be one of {list(timeframe_map.keys())}"
            )
        
        # Fetch from API
        logger.info(f"Fetching bars for {symbol} ({asset_class.value}, {timeframe}, {start} to {end})")
        
        def fetch_stock_bars():
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe_map[timeframe],
                start=start,
                end=end,
                limit=limit
            )
            response = self.stock_data_client.get_stock_bars(request)
            
            if symbol not in response:
                raise ValueError(f"No bar data available for {symbol}")
            
            bars = response[symbol]
            
            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': int(bar.volume)
                })
            
            df = pd.DataFrame(data)
            
            if df.empty:
                raise ValueError(f"No bar data returned for {symbol}")
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
        
        def fetch_crypto_bars():
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe_map[timeframe],
                start=start,
                end=end,
                limit=limit
            )
            response = self.crypto_data_client.get_crypto_bars(request)
            
            if symbol not in response:
                raise ValueError(f"No bar data available for {symbol}")
            
            bars = response[symbol]
            
            # Convert to DataFrame
            data = []
            for bar in bars:
                data.append({
                    'timestamp': bar.timestamp,
                    'open': float(bar.open),
                    'high': float(bar.high),
                    'low': float(bar.low),
                    'close': float(bar.close),
                    'volume': float(bar.volume),  # Crypto volume can be fractional
                    'trade_count': int(bar.trade_count) if hasattr(bar, 'trade_count') else 0,
                    'vwap': float(bar.vwap) if hasattr(bar, 'vwap') else None
                })
            
            df = pd.DataFrame(data)
            
            if df.empty:
                raise ValueError(f"No bar data returned for {symbol}")
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            return df
        
        try:
            if asset_class == AssetClass.STOCK:
                bars_df = self._retry_on_failure(fetch_stock_bars)
            elif asset_class == AssetClass.CRYPTO:
                bars_df = self._retry_on_failure(fetch_crypto_bars)
            elif asset_class == AssetClass.FOREX:
                raise NotImplementedError("Forex bars not yet supported")
            else:
                raise ValueError(f"Unsupported asset class: {asset_class}")
            
            # Update cache
            self._bars_cache[cache_key] = (bars_df.copy(), time.time())
            
            logger.info(f"Fetched {len(bars_df)} bars for {symbol}")
            return bars_df
        except Exception as e:
            logger.error(f"Failed to fetch bars for {symbol}: {e}")
            raise
    
    def get_market_status(self, use_cache: bool = True) -> MarketStatus:
        """
        Check if the market is currently open.
        
        Args:
            use_cache: Whether to use cached status if available
            
        Returns:
            MarketStatus object with current market state
            
        Raises:
            Exception: If API request fails
        """
        # Check cache
        if use_cache and self._market_status_cache is not None:
            cached_status, cached_time = self._market_status_cache
            if self._is_cache_valid(cached_time):
                logger.debug("Using cached market status")
                return cached_status
        
        # Fetch from API
        logger.info("Fetching market status")
        
        def fetch_status():
            # Get calendar for today and tomorrow
            today = datetime.now().date()
            tomorrow = today + timedelta(days=1)
            
            request = GetCalendarRequest(
                start=today,
                end=tomorrow
            )
            calendar = self.trading_client.get_calendar(request)
            
            current_time = datetime.now()
            
            # Check if market is open today
            is_open = False
            next_open = None
            next_close = None
            
            if calendar:
                today_session = calendar[0]
                
                # Convert to datetime for comparison
                market_open = datetime.combine(
                    today_session.date,
                    today_session.open
                )
                market_close = datetime.combine(
                    today_session.date,
                    today_session.close
                )
                
                # Check if currently within trading hours
                is_open = market_open <= current_time <= market_close
                
                if current_time < market_open:
                    next_open = market_open
                    next_close = market_close
                elif current_time > market_close:
                    # Market closed for today, next open is tomorrow (if available)
                    if len(calendar) > 1:
                        next_session = calendar[1]
                        next_open = datetime.combine(
                            next_session.date,
                            next_session.open
                        )
                        next_close = datetime.combine(
                            next_session.date,
                            next_session.close
                        )
                else:
                    # Market is open, next close is today
                    next_close = market_close
            
            return MarketStatus(
                is_open=is_open,
                next_open=next_open,
                next_close=next_close,
                current_time=current_time
            )
        
        try:
            status = self._retry_on_failure(fetch_status)
            
            # Update cache
            self._market_status_cache = (status, time.time())
            
            logger.info(f"Market status: {'OPEN' if status.is_open else 'CLOSED'}")
            return status
        except Exception as e:
            logger.error(f"Failed to fetch market status: {e}")
            raise
    
    def _initialize_stream_client(self, asset_class: AssetClass = AssetClass.STOCK):
        """
        Initialize the WebSocket stream client if not already initialized.
        
        Args:
            asset_class: Asset class to initialize stream for
        """
        if asset_class == AssetClass.STOCK:
            if self._stock_stream_client is None:
                self._stock_stream_client = StockDataStream(
                    api_key=self.api_key,
                    secret_key=self.api_secret
                )
                logger.info("Initialized stock WebSocket stream client")
        elif asset_class == AssetClass.CRYPTO:
            if self._crypto_stream_client is None:
                self._crypto_stream_client = CryptoDataStream(
                    api_key=self.api_key,
                    secret_key=self.api_secret
                )
                logger.info("Initialized crypto WebSocket stream client")
        else:
            raise ValueError(f"Unsupported asset class for streaming: {asset_class}")
    
    async def _handle_quote_update(self, quote_data, asset_class: AssetClass = AssetClass.STOCK):
        """
        Internal handler for quote updates from WebSocket.
        
        Args:
            quote_data: Quote data from Alpaca WebSocket
            asset_class: Asset class of the quote
        """
        try:
            symbol = quote_data.symbol
            
            # Create Quote object
            quote = Quote(
                symbol=symbol,
                bid_price=float(quote_data.bid_price),
                ask_price=float(quote_data.ask_price),
                bid_size=int(quote_data.bid_size),
                ask_size=int(quote_data.ask_size),
                timestamp=quote_data.timestamp,
                asset_class=asset_class
            )
            
            # Update cache
            cache_key = f"{asset_class.value}:{symbol}"
            self._quote_cache[cache_key] = (quote, time.time())
            
            # Call registered callbacks
            if symbol in self._stream_callbacks:
                for callback in self._stream_callbacks[symbol]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(quote)
                        else:
                            callback(quote)
                    except Exception as e:
                        logger.error(f"Error in quote callback for {symbol}: {e}")
            
            logger.debug(f"Processed quote update for {symbol} ({asset_class.value}): {quote.mid_price}")
        except Exception as e:
            logger.error(f"Error handling quote update: {e}")
    
    async def _connect_stream(self):
        """
        Establish WebSocket connection with reconnection logic.
        
        Raises:
            Exception: If connection fails after max retries
        """
        self._initialize_stream_client()
        
        while self._stream_reconnect_attempts < self._max_reconnect_attempts:
            try:
                logger.info(
                    f"Connecting to WebSocket stream "
                    f"(attempt {self._stream_reconnect_attempts + 1}/{self._max_reconnect_attempts})"
                )
                
                # Subscribe to quotes for all tracked symbols
                if self._stream_subscriptions:
                    for symbol in self._stream_subscriptions:
                        self._stream_client.subscribe_quotes(
                            self._handle_quote_update,
                            symbol
                        )
                    logger.info(f"Subscribed to {len(self._stream_subscriptions)} symbols")
                
                # Run the stream
                self._stream_connected = True
                self._stream_reconnect_attempts = 0
                logger.info("WebSocket stream connected successfully")
                
                await self._stream_client._run_forever()
                
            except Exception as e:
                self._stream_connected = False
                self._stream_reconnect_attempts += 1
                logger.error(
                    f"WebSocket connection failed: {e} "
                    f"(attempt {self._stream_reconnect_attempts}/{self._max_reconnect_attempts})"
                )
                
                if self._stream_reconnect_attempts < self._max_reconnect_attempts:
                    # Exponential backoff
                    delay = self._reconnect_delay * (2 ** (self._stream_reconnect_attempts - 1))
                    logger.info(f"Reconnecting in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max reconnection attempts reached")
                    raise
    
    async def stream_quotes(
        self,
        symbols: List[str],
        callback: Optional[Callable[[Quote], None]] = None
    ) -> AsyncIterator[Quote]:
        """
        Stream real-time quotes for given symbols via WebSocket.
        
        This method establishes a WebSocket connection to Alpaca and streams
        live quote updates. It handles connection lifecycle including automatic
        reconnection on failures.
        
        Args:
            symbols: List of stock symbols to stream (e.g., ['AAPL', 'GOOGL'])
            callback: Optional callback function to call on each quote update.
                     Can be sync or async function.
        
        Yields:
            Quote objects as they are received from the stream
            
        Raises:
            ValueError: If symbols list is empty or invalid
            Exception: If connection fails after max retries
            
        Example:
            ```python
            async for quote in service.stream_quotes(['AAPL', 'GOOGL']):
                print(f"{quote.symbol}: ${quote.mid_price}")
            ```
        """
        if not symbols or not isinstance(symbols, list):
            raise ValueError("Symbols must be a non-empty list")
        
        # Normalize symbols
        symbols = [s.upper() for s in symbols if s]
        
        if not symbols:
            raise ValueError("No valid symbols provided")
        
        logger.info(f"Starting quote stream for {len(symbols)} symbols: {symbols}")
        
        # Initialize stream client
        self._initialize_stream_client()
        
        # Create a queue for this stream
        quote_queue: asyncio.Queue = asyncio.Queue()
        
        async def queue_callback(quote: Quote):
            """Internal callback to put quotes in the queue"""
            await quote_queue.put(quote)
            
            # Also call user callback if provided
            if callback:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(quote)
                    else:
                        callback(quote)
                except Exception as e:
                    logger.error(f"Error in user callback: {e}")
        
        # Register callbacks for each symbol
        for symbol in symbols:
            if symbol not in self._stream_callbacks:
                self._stream_callbacks[symbol] = []
            self._stream_callbacks[symbol].append(queue_callback)
            self._stream_subscriptions.add(symbol)
        
        # Start connection task if not already connected
        connection_task = None
        if not self._stream_connected:
            connection_task = asyncio.create_task(self._connect_stream())
        
        try:
            # Subscribe to quotes
            for symbol in symbols:
                self._stream_client.subscribe_quotes(
                    self._handle_quote_update,
                    symbol
                )
            
            logger.info(f"Subscribed to quotes for: {symbols}")
            
            # Yield quotes from the queue
            while True:
                try:
                    # Wait for quote with timeout to allow checking connection status
                    quote = await asyncio.wait_for(quote_queue.get(), timeout=1.0)
                    yield quote
                except asyncio.TimeoutError:
                    # Check if connection is still alive
                    if connection_task and connection_task.done():
                        # Connection task finished, check for exceptions
                        try:
                            connection_task.result()
                        except Exception as e:
                            logger.error(f"Connection task failed: {e}")
                            raise
                    continue
                    
        except asyncio.CancelledError:
            logger.info("Quote stream cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in quote stream: {e}")
            raise
        finally:
            # Cleanup: unsubscribe and remove callbacks
            for symbol in symbols:
                if symbol in self._stream_callbacks:
                    try:
                        self._stream_callbacks[symbol].remove(queue_callback)
                        if not self._stream_callbacks[symbol]:
                            del self._stream_callbacks[symbol]
                            self._stream_subscriptions.discard(symbol)
                    except (ValueError, KeyError):
                        pass
            
            logger.info(f"Cleaned up stream for symbols: {symbols}")
    
    async def disconnect_stream(self):
        """
        Disconnect the WebSocket stream and cleanup resources.
        
        This should be called when streaming is no longer needed to properly
        close the connection and release resources.
        """
        if self._stream_client and self._stream_connected:
            try:
                logger.info("Disconnecting WebSocket stream")
                await self._stream_client.stop_ws()
                self._stream_connected = False
                self._stream_subscriptions.clear()
                self._stream_callbacks.clear()
                logger.info("WebSocket stream disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting stream: {e}")
                raise
    
    def get_stream_status(self) -> Dict[str, Any]:
        """
        Get current WebSocket stream status.
        
        Returns:
            Dictionary with stream status information
        """
        return {
            'stock_connected': self._stream_connected.get(AssetClass.STOCK, False),
            'crypto_connected': self._stream_connected.get(AssetClass.CRYPTO, False),
            'stock_subscriptions': list(self._stream_subscriptions.get(AssetClass.STOCK, set())),
            'crypto_subscriptions': list(self._stream_subscriptions.get(AssetClass.CRYPTO, set())),
            'active_callbacks': sum(len(cbs) for cbs in self._stream_callbacks.values()),
            'reconnect_attempts': self._stream_reconnect_attempts
        }
    
    def clear_cache(self, symbol: Optional[str] = None, asset_class: Optional[AssetClass] = None):
        """
        Clear cached data.
        
        Args:
            symbol: If provided, clear cache only for this symbol.
                   If None, clear all cache.
            asset_class: If provided with symbol, clear cache for specific asset class.
                        If None, auto-detect or clear all.
        """
        if symbol:
            symbol = symbol.upper()
            
            # Detect asset class if not provided
            if asset_class is None:
                asset_class = self.detect_asset_class(symbol)
            
            # Clear quote cache
            cache_key = f"{asset_class.value}:{symbol}"
            if cache_key in self._quote_cache:
                del self._quote_cache[cache_key]
                logger.debug(f"Cleared quote cache for {symbol} ({asset_class.value})")
            
            # Clear bars cache (all entries containing this symbol and asset class)
            keys_to_delete = [
                key for key in self._bars_cache.keys()
                if key.startswith(f"{asset_class.value}:{symbol}")
            ]
            for key in keys_to_delete:
                del self._bars_cache[key]
            
            if keys_to_delete:
                logger.debug(f"Cleared {len(keys_to_delete)} bar cache entries for {symbol}")
        else:
            # Clear all cache
            self._quote_cache.clear()
            self._bars_cache.clear()
            self._market_status_cache = None
            logger.info("Cleared all cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            'quote_cache_size': len(self._quote_cache),
            'bars_cache_size': len(self._bars_cache),
            'market_status_cached': self._market_status_cache is not None,
            'cache_ttl_seconds': self.cache_ttl
        }
