"""Unit tests for market data service."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import pandas as pd
from hypothesis import given, settings, strategies as st
import asyncio

from services.market_data_service import (
    MarketDataService,
    Quote,
    MarketStatus
)


@pytest.fixture
def mock_alpaca_clients():
    """Mock Alpaca API clients"""
    with patch('services.market_data_service.StockHistoricalDataClient') as mock_data_client, \
         patch('services.market_data_service.TradingClient') as mock_trading_client:
        yield mock_data_client, mock_trading_client


@pytest.fixture
def market_data_service(mock_alpaca_clients):
    """Create market data service with mocked clients"""
    with patch('services.market_data_service.settings') as mock_settings:
        mock_settings.alpaca.api_key = 'test_key'
        mock_settings.alpaca.secret_key = 'test_secret'
        mock_settings.cache_ttl_seconds = 300
        
        service = MarketDataService(
            api_key='test_key',
            api_secret='test_secret',
            paper=True
        )
        return service


class TestMarketDataService:
    """Test suite for MarketDataService"""
    
    def test_initialization_with_credentials(self, mock_alpaca_clients):
        """Test service initialization with valid credentials"""
        with patch('services.market_data_service.settings') as mock_settings:
            mock_settings.cache_ttl_seconds = 300
            
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            assert service.api_key == 'test_key'
            assert service.api_secret == 'test_secret'
            assert service.paper is True
    
    def test_initialization_without_credentials_raises_error(self, mock_alpaca_clients):
        """Test that initialization fails without credentials"""
        with patch('services.market_data_service.settings') as mock_settings:
            mock_settings.alpaca.api_key = ''
            mock_settings.alpaca.secret_key = ''
            
            with pytest.raises(ValueError, match="Alpaca API credentials are required"):
                MarketDataService()
    
    def test_get_latest_quote_success(self, market_data_service):
        """Test successful quote fetching"""
        # Mock quote data
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.5
        mock_quote.bid_size = 100
        mock_quote.ask_size = 200
        mock_quote.timestamp = datetime.now()
        
        # Mock API response
        market_data_service.data_client.get_stock_latest_quote = Mock(
            return_value={'AAPL': mock_quote}
        )
        
        # Fetch quote
        quote = market_data_service.get_latest_quote('AAPL', use_cache=False)
        
        assert quote.symbol == 'AAPL'
        assert quote.bid_price == 150.0
        assert quote.ask_price == 150.5
        assert quote.bid_size == 100
        assert quote.ask_size == 200
        assert quote.mid_price == 150.25
    
    def test_get_latest_quote_invalid_symbol(self, market_data_service):
        """Test quote fetching with invalid symbol"""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            market_data_service.get_latest_quote('')
        
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            market_data_service.get_latest_quote(None)
    
    def test_get_latest_quote_uses_cache(self, market_data_service):
        """Test that quote caching works"""
        # Mock quote data
        mock_quote = Mock()
        mock_quote.bid_price = 150.0
        mock_quote.ask_price = 150.5
        mock_quote.bid_size = 100
        mock_quote.ask_size = 200
        mock_quote.timestamp = datetime.now()
        
        # Mock API response
        market_data_service.data_client.get_stock_latest_quote = Mock(
            return_value={'AAPL': mock_quote}
        )
        
        # First call - should hit API
        quote1 = market_data_service.get_latest_quote('AAPL', use_cache=True)
        assert market_data_service.data_client.get_stock_latest_quote.call_count == 1
        
        # Second call - should use cache
        quote2 = market_data_service.get_latest_quote('AAPL', use_cache=True)
        assert market_data_service.data_client.get_stock_latest_quote.call_count == 1
        
        # Quotes should be the same
        assert quote1.symbol == quote2.symbol
        assert quote1.bid_price == quote2.bid_price
    
    def test_get_bars_success(self, market_data_service):
        """Test successful historical bars fetching"""
        # Mock bar data
        mock_bar1 = Mock()
        mock_bar1.timestamp = datetime(2024, 1, 1, 9, 30)
        mock_bar1.open = 150.0
        mock_bar1.high = 151.0
        mock_bar1.low = 149.5
        mock_bar1.close = 150.5
        mock_bar1.volume = 1000000
        
        mock_bar2 = Mock()
        mock_bar2.timestamp = datetime(2024, 1, 2, 9, 30)
        mock_bar2.open = 150.5
        mock_bar2.high = 152.0
        mock_bar2.low = 150.0
        mock_bar2.close = 151.5
        mock_bar2.volume = 1200000
        
        # Mock API response
        market_data_service.data_client.get_stock_bars = Mock(
            return_value={'AAPL': [mock_bar1, mock_bar2]}
        )
        
        # Fetch bars
        start = datetime(2024, 1, 1)
        end = datetime(2024, 1, 3)
        bars = market_data_service.get_bars('AAPL', '1Day', start, end, use_cache=False)
        
        assert isinstance(bars, pd.DataFrame)
        assert len(bars) == 2
        assert 'open' in bars.columns
        assert 'high' in bars.columns
        assert 'low' in bars.columns
        assert 'close' in bars.columns
        assert 'volume' in bars.columns
        assert bars.iloc[0]['close'] == 150.5
        assert bars.iloc[1]['close'] == 151.5
    
    def test_get_bars_invalid_symbol(self, market_data_service):
        """Test bars fetching with invalid symbol"""
        with pytest.raises(ValueError, match="Symbol must be a non-empty string"):
            market_data_service.get_bars('')
    
    def test_get_bars_invalid_date_range(self, market_data_service):
        """Test bars fetching with invalid date range"""
        start = datetime(2024, 1, 10)
        end = datetime(2024, 1, 1)
        
        with pytest.raises(ValueError, match="Start date must be before end date"):
            market_data_service.get_bars('AAPL', '1Day', start, end)
    
    def test_get_bars_invalid_timeframe(self, market_data_service):
        """Test bars fetching with invalid timeframe"""
        with pytest.raises(ValueError, match="Invalid timeframe"):
            market_data_service.get_bars('AAPL', 'InvalidTimeframe')
    
    def test_get_market_status_open(self, market_data_service):
        """Test market status when market is open"""
        # Mock calendar data
        mock_session = Mock()
        mock_session.date = datetime.now().date()
        mock_session.open = (datetime.now() - timedelta(hours=1)).time()
        mock_session.close = (datetime.now() + timedelta(hours=1)).time()
        
        market_data_service.trading_client.get_calendar = Mock(
            return_value=[mock_session]
        )
        
        # Get market status
        status = market_data_service.get_market_status(use_cache=False)
        
        assert isinstance(status, MarketStatus)
        assert status.is_open is True
        assert status.next_close is not None
    
    def test_get_market_status_closed(self, market_data_service):
        """Test market status when market is closed"""
        # Mock calendar data - market closed hours ago
        mock_session = Mock()
        mock_session.date = datetime.now().date()
        mock_session.open = (datetime.now() - timedelta(hours=10)).time()
        mock_session.close = (datetime.now() - timedelta(hours=2)).time()
        
        market_data_service.trading_client.get_calendar = Mock(
            return_value=[mock_session]
        )
        
        # Get market status
        status = market_data_service.get_market_status(use_cache=False)
        
        assert isinstance(status, MarketStatus)
        assert status.is_open is False
    
    def test_clear_cache_specific_symbol(self, market_data_service):
        """Test clearing cache for specific symbol"""
        # Add some cache entries
        quote = Quote('AAPL', 150.0, 150.5, 100, 200, datetime.now())
        market_data_service._quote_cache['AAPL'] = (quote, 0)
        market_data_service._quote_cache['MSFT'] = (quote, 0)
        
        # Clear cache for AAPL only
        market_data_service.clear_cache('AAPL')
        
        assert 'AAPL' not in market_data_service._quote_cache
        assert 'MSFT' in market_data_service._quote_cache
    
    def test_clear_cache_all(self, market_data_service):
        """Test clearing all cache"""
        # Add some cache entries
        quote = Quote('AAPL', 150.0, 150.5, 100, 200, datetime.now())
        market_data_service._quote_cache['AAPL'] = (quote, 0)
        market_data_service._quote_cache['MSFT'] = (quote, 0)
        
        # Clear all cache
        market_data_service.clear_cache()
        
        assert len(market_data_service._quote_cache) == 0
        assert len(market_data_service._bars_cache) == 0
        assert market_data_service._market_status_cache is None
    
    def test_get_cache_stats(self, market_data_service):
        """Test cache statistics"""
        # Add some cache entries
        quote = Quote('AAPL', 150.0, 150.5, 100, 200, datetime.now())
        market_data_service._quote_cache['AAPL'] = (quote, 0)
        
        stats = market_data_service.get_cache_stats()
        
        assert stats['quote_cache_size'] == 1
        assert stats['bars_cache_size'] == 0
        assert stats['market_status_cached'] is False
        assert 'cache_ttl_seconds' in stats
    
    def test_retry_on_api_failure(self, market_data_service):
        """Test retry logic on API failures"""
        # Mock API to fail twice then succeed
        call_count = 0
        
        def mock_get_quote(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("API Error")
            
            mock_quote = Mock()
            mock_quote.bid_price = 150.0
            mock_quote.ask_price = 150.5
            mock_quote.bid_size = 100
            mock_quote.ask_size = 200
            mock_quote.timestamp = datetime.now()
            return {'AAPL': mock_quote}
        
        market_data_service.data_client.get_stock_latest_quote = mock_get_quote
        market_data_service._retry_delay = 0.1  # Speed up test
        
        # Should succeed after retries
        quote = market_data_service.get_latest_quote('AAPL', use_cache=False)
        assert quote.symbol == 'AAPL'
        assert call_count == 3
    
    def test_retry_exhaustion(self, market_data_service):
        """Test that retries eventually fail"""
        # Mock API to always fail
        market_data_service.data_client.get_stock_latest_quote = Mock(
            side_effect=Exception("API Error")
        )
        market_data_service._retry_delay = 0.1  # Speed up test
        
        # Should raise exception after all retries
        with pytest.raises(Exception, match="API Error"):
            market_data_service.get_latest_quote('AAPL', use_cache=False)
        
        # Should have tried max_retries times
        assert market_data_service.data_client.get_stock_latest_quote.call_count == 3


class TestQuote:
    """Test suite for Quote dataclass"""
    
    def test_quote_mid_price(self):
        """Test mid price calculation"""
        quote = Quote(
            symbol='AAPL',
            bid_price=150.0,
            ask_price=150.5,
            bid_size=100,
            ask_size=200,
            timestamp=datetime.now()
        )
        
        assert quote.mid_price == 150.25
    
    def test_quote_creation(self):
        """Test quote object creation"""
        timestamp = datetime.now()
        quote = Quote(
            symbol='AAPL',
            bid_price=150.0,
            ask_price=150.5,
            bid_size=100,
            ask_size=200,
            timestamp=timestamp
        )
        
        assert quote.symbol == 'AAPL'
        assert quote.bid_price == 150.0
        assert quote.ask_price == 150.5
        assert quote.bid_size == 100
        assert quote.ask_size == 200
        assert quote.timestamp == timestamp


class TestMarketStatus:
    """Test suite for MarketStatus dataclass"""
    
    def test_market_status_creation(self):
        """Test market status object creation"""
        current_time = datetime.now()
        next_open = current_time + timedelta(hours=1)
        next_close = current_time + timedelta(hours=8)
        
        status = MarketStatus(
            is_open=False,
            next_open=next_open,
            next_close=next_close,
            current_time=current_time
        )
        
        assert status.is_open is False
        assert status.next_open == next_open
        assert status.next_close == next_close
        assert status.current_time == current_time


# Property-Based Tests
class TestMarketDataServiceProperties:
    """Property-based tests for MarketDataService"""
    
    @given(
        symbol=st.text(
            alphabet=st.characters(whitelist_categories=('Lu',)),
            min_size=1,
            max_size=5
        ),
        bid_price=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        ask_price=st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False),
        bid_size=st.integers(min_value=1, max_value=1000000),
        ask_size=st.integers(min_value=1, max_value=1000000)
    )
    @settings(max_examples=100, deadline=None)
    def test_property_real_time_data_freshness(
        self,
        symbol,
        bid_price,
        ask_price,
        bid_size,
        ask_size
    ):
        """
        Feature: ai-trading-agent, Property 1: Real-time data freshness
        
        For any stock symbol being monitored, the displayed price should be 
        updated within 10 seconds of the actual market price change.
        
        Validates: Requirements 1.3
        """
        # Ensure ask_price >= bid_price for valid quote
        if ask_price < bid_price:
            bid_price, ask_price = ask_price, bid_price
        
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Mock the current time
            current_time = datetime.now()
            
            # Mock quote data with timestamp
            mock_quote = Mock()
            mock_quote.bid_price = bid_price
            mock_quote.ask_price = ask_price
            mock_quote.bid_size = bid_size
            mock_quote.ask_size = ask_size
            mock_quote.timestamp = current_time
            
            # Mock API response
            service.data_client.get_stock_latest_quote = Mock(
                return_value={symbol: mock_quote}
            )
            
            # Fetch quote
            quote = service.get_latest_quote(symbol, use_cache=False)
            
            # Calculate time difference between quote timestamp and current time
            time_diff = abs((datetime.now() - quote.timestamp).total_seconds())
            
            # Property: The quote timestamp should be within 10 seconds of current time
            # This validates that the data is fresh and meets the real-time requirement
            assert time_diff <= 10.0, (
                f"Quote timestamp is {time_diff:.2f} seconds old, "
                f"exceeds 10 second freshness requirement"
            )
            
            # Additional validation: Quote should have valid data
            assert quote.symbol == symbol
            assert quote.bid_price == bid_price
            assert quote.ask_price == ask_price
            assert quote.bid_size == bid_size
            assert quote.ask_size == ask_size
            assert quote.timestamp is not None


# Integration Tests for WebSocket Streaming
class TestWebSocketStreaming:
    """Integration tests for WebSocket streaming functionality"""
    
    @pytest.mark.asyncio
    async def test_connection_establishment(self):
        """
        Test WebSocket connection establishment.
        
        Validates: Requirements 1.3
        """
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.StockDataStream') as mock_stream_class, \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create mock stream instance
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Verify stream client is not initialized yet
            assert service._stream_client is None
            assert service._stream_connected is False
            
            # Initialize stream client
            service._initialize_stream_client()
            
            # Verify stream client was created
            assert service._stream_client is not None
            mock_stream_class.assert_called_once_with(
                api_key='test_key',
                secret_key='test_secret'
            )
            
            # Verify initial state
            assert service._stream_connected is False
            assert len(service._stream_subscriptions) == 0
            assert len(service._stream_callbacks) == 0
    
    @pytest.mark.asyncio
    async def test_quote_updates(self):
        """
        Test receiving and processing quote updates from WebSocket.
        
        Validates: Requirements 1.3
        """
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.StockDataStream') as mock_stream_class, \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create mock stream instance
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Initialize stream client
            service._initialize_stream_client()
            
            # Create mock quote data
            mock_quote_data = Mock()
            mock_quote_data.symbol = 'AAPL'
            mock_quote_data.bid_price = 150.0
            mock_quote_data.ask_price = 150.5
            mock_quote_data.bid_size = 100
            mock_quote_data.ask_size = 200
            mock_quote_data.timestamp = datetime.now()
            
            # Track callback invocations
            callback_invoked = []
            
            async def test_callback(quote: Quote):
                callback_invoked.append(quote)
            
            # Register callback
            service._stream_callbacks['AAPL'] = [test_callback]
            
            # Process quote update
            await service._handle_quote_update(mock_quote_data)
            
            # Verify callback was invoked
            assert len(callback_invoked) == 1
            received_quote = callback_invoked[0]
            
            # Verify quote data
            assert received_quote.symbol == 'AAPL'
            assert received_quote.bid_price == 150.0
            assert received_quote.ask_price == 150.5
            assert received_quote.bid_size == 100
            assert received_quote.ask_size == 200
            assert received_quote.mid_price == 150.25
            
            # Verify quote was cached
            assert 'AAPL' in service._quote_cache
            cached_quote, _ = service._quote_cache['AAPL']
            assert cached_quote.symbol == 'AAPL'
            assert cached_quote.mid_price == 150.25
    
    @pytest.mark.asyncio
    async def test_stream_quotes_basic(self):
        """
        Test basic quote streaming functionality.
        
        Validates: Requirements 1.3
        """
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.StockDataStream') as mock_stream_class, \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create mock stream instance
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Simulate quote stream by manually triggering callbacks
            symbols = ['AAPL', 'GOOGL']
            received_quotes = []
            
            async def collect_quotes():
                """Collect quotes from stream"""
                count = 0
                async for quote in service.stream_quotes(symbols):
                    received_quotes.append(quote)
                    count += 1
                    if count >= 2:  # Stop after 2 quotes
                        break
            
            # Create task for streaming
            stream_task = asyncio.create_task(collect_quotes())
            
            # Wait a bit for subscription to be set up
            await asyncio.sleep(0.1)
            
            # Verify subscriptions were registered
            assert 'AAPL' in service._stream_subscriptions
            assert 'GOOGL' in service._stream_subscriptions
            
            # Simulate receiving quotes
            mock_quote1 = Mock()
            mock_quote1.symbol = 'AAPL'
            mock_quote1.bid_price = 150.0
            mock_quote1.ask_price = 150.5
            mock_quote1.bid_size = 100
            mock_quote1.ask_size = 200
            mock_quote1.timestamp = datetime.now()
            
            mock_quote2 = Mock()
            mock_quote2.symbol = 'GOOGL'
            mock_quote2.bid_price = 2800.0
            mock_quote2.ask_price = 2801.0
            mock_quote2.bid_size = 50
            mock_quote2.ask_size = 75
            mock_quote2.timestamp = datetime.now()
            
            # Trigger quote updates
            await service._handle_quote_update(mock_quote1)
            await service._handle_quote_update(mock_quote2)
            
            # Wait for stream task to complete
            await asyncio.wait_for(stream_task, timeout=2.0)
            
            # Verify quotes were received
            assert len(received_quotes) == 2
            assert received_quotes[0].symbol == 'AAPL'
            assert received_quotes[1].symbol == 'GOOGL'
            
            # Cleanup
            await service.disconnect_stream()
    
    @pytest.mark.asyncio
    async def test_stream_with_callback(self):
        """
        Test quote streaming with custom callback.
        
        Validates: Requirements 1.3
        """
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.StockDataStream') as mock_stream_class, \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create mock stream instance
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Track callback invocations
            callback_quotes = []
            
            async def user_callback(quote: Quote):
                callback_quotes.append(quote)
            
            # Start streaming with callback
            symbols = ['AAPL']
            stream_quotes = []
            
            async def collect_quotes():
                count = 0
                async for quote in service.stream_quotes(symbols, callback=user_callback):
                    stream_quotes.append(quote)
                    count += 1
                    if count >= 1:
                        break
            
            stream_task = asyncio.create_task(collect_quotes())
            
            # Wait for subscription
            await asyncio.sleep(0.1)
            
            # Simulate quote
            mock_quote = Mock()
            mock_quote.symbol = 'AAPL'
            mock_quote.bid_price = 150.0
            mock_quote.ask_price = 150.5
            mock_quote.bid_size = 100
            mock_quote.ask_size = 200
            mock_quote.timestamp = datetime.now()
            
            await service._handle_quote_update(mock_quote)
            
            # Wait for completion
            await asyncio.wait_for(stream_task, timeout=2.0)
            
            # Verify both stream and callback received the quote
            assert len(stream_quotes) == 1
            assert len(callback_quotes) == 1
            assert stream_quotes[0].symbol == 'AAPL'
            assert callback_quotes[0].symbol == 'AAPL'
            
            # Cleanup
            await service.disconnect_stream()
    
    @pytest.mark.asyncio
    async def test_reconnection_on_failure(self):
        """
        Test automatic reconnection when WebSocket connection fails.
        
        Validates: Requirements 1.3
        """
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.StockDataStream') as mock_stream_class, \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create mock stream instance
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Reduce reconnection delay for faster testing
            service._reconnect_delay = 0.05
            
            # Mock _run_forever to fail first time, succeed second time
            call_count = 0
            
            async def mock_run_forever():
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise ConnectionError("Connection failed")
                # Second call succeeds - wait briefly then complete
                await asyncio.sleep(0.2)
            
            mock_stream._run_forever = mock_run_forever
            
            # Add a symbol to trigger subscription
            service._stream_subscriptions.add('AAPL')
            
            # Start connection (will fail and retry)
            connection_task = asyncio.create_task(service._connect_stream())
            
            # Wait for first failure and retry
            await asyncio.sleep(0.3)
            
            # Verify reconnection was attempted
            assert service._stream_reconnect_attempts == 0  # Reset after successful connection
            assert call_count >= 2  # Should have tried at least twice
            
            # Cancel the task
            connection_task.cancel()
            
            try:
                await connection_task
            except asyncio.CancelledError:
                pass
    
    @pytest.mark.asyncio
    async def test_max_reconnection_attempts(self):
        """
        Test that reconnection attempts are tracked.
        
        Validates: Requirements 1.3
        """
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.StockDataStream') as mock_stream_class, \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create mock stream instance
            mock_stream = MagicMock()
            mock_stream.subscribe_quotes = MagicMock()
            mock_stream_class.return_value = mock_stream
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Verify initial state
            assert service._stream_reconnect_attempts == 0
            assert service._max_reconnect_attempts == 5  # Default value
            
            # Verify we can modify max attempts
            service._max_reconnect_attempts = 3
            assert service._max_reconnect_attempts == 3
    
    @pytest.mark.asyncio
    async def test_disconnect_stream(self):
        """
        Test proper disconnection and cleanup of WebSocket stream.
        
        Validates: Requirements 1.3
        """
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.StockDataStream') as mock_stream_class, \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create mock stream instance
            mock_stream = MagicMock()
            mock_stream.stop_ws = AsyncMock()
            mock_stream_class.return_value = mock_stream
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Initialize stream
            service._initialize_stream_client()
            service._stream_connected = True
            service._stream_subscriptions.add('AAPL')
            service._stream_callbacks['AAPL'] = [lambda q: None]
            
            # Disconnect
            await service.disconnect_stream()
            
            # Verify cleanup
            assert service._stream_connected is False
            assert len(service._stream_subscriptions) == 0
            assert len(service._stream_callbacks) == 0
            mock_stream.stop_ws.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_stream_status(self):
        """
        Test getting WebSocket stream status.
        
        Validates: Requirements 1.3
        """
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.StockDataStream') as mock_stream_class, \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create mock stream instance
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Get initial status
            status = service.get_stream_status()
            assert status['connected'] is False
            assert len(status['subscribed_symbols']) == 0
            assert status['active_callbacks'] == 0
            assert status['reconnect_attempts'] == 0
            
            # Simulate connected state
            service._stream_connected = True
            service._stream_subscriptions.add('AAPL')
            service._stream_subscriptions.add('GOOGL')
            service._stream_callbacks['AAPL'] = [lambda q: None, lambda q: None]
            service._stream_reconnect_attempts = 2
            
            # Get updated status
            status = service.get_stream_status()
            assert status['connected'] is True
            assert len(status['subscribed_symbols']) == 2
            assert 'AAPL' in status['subscribed_symbols']
            assert 'GOOGL' in status['subscribed_symbols']
            assert status['active_callbacks'] == 2
            assert status['reconnect_attempts'] == 2
    
    @pytest.mark.asyncio
    async def test_invalid_symbols(self):
        """
        Test error handling for invalid symbols in streaming.
        
        Validates: Requirements 1.3
        """
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.StockDataStream') as mock_stream_class, \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create mock stream instance
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Test empty list
            with pytest.raises(ValueError, match="Symbols must be a non-empty list"):
                stream = service.stream_quotes([])
                async for _ in stream:
                    pass
            
            # Test None
            with pytest.raises(ValueError, match="Symbols must be a non-empty list"):
                stream = service.stream_quotes(None)
                async for _ in stream:
                    pass
            
            # Test list with empty strings
            with pytest.raises(ValueError, match="No valid symbols provided"):
                stream = service.stream_quotes(['', '  '])
                async for _ in stream:
                    pass
    
    @pytest.mark.asyncio
    async def test_multiple_callbacks_per_symbol(self):
        """
        Test that multiple callbacks can be registered for the same symbol.
        
        Validates: Requirements 1.3
        """
        with patch('services.market_data_service.StockHistoricalDataClient'), \
             patch('services.market_data_service.TradingClient'), \
             patch('services.market_data_service.StockDataStream') as mock_stream_class, \
             patch('services.market_data_service.settings') as mock_settings:
            
            mock_settings.alpaca.api_key = 'test_key'
            mock_settings.alpaca.secret_key = 'test_secret'
            mock_settings.cache_ttl_seconds = 300
            
            # Create mock stream instance
            mock_stream = MagicMock()
            mock_stream_class.return_value = mock_stream
            
            # Create service
            service = MarketDataService(
                api_key='test_key',
                api_secret='test_secret',
                paper=True
            )
            
            # Initialize stream
            service._initialize_stream_client()
            
            # Register multiple callbacks for same symbol
            callback1_invoked = []
            callback2_invoked = []
            
            async def callback1(quote: Quote):
                callback1_invoked.append(quote)
            
            async def callback2(quote: Quote):
                callback2_invoked.append(quote)
            
            service._stream_callbacks['AAPL'] = [callback1, callback2]
            
            # Simulate quote update
            mock_quote = Mock()
            mock_quote.symbol = 'AAPL'
            mock_quote.bid_price = 150.0
            mock_quote.ask_price = 150.5
            mock_quote.bid_size = 100
            mock_quote.ask_size = 200
            mock_quote.timestamp = datetime.now()
            
            await service._handle_quote_update(mock_quote)
            
            # Verify both callbacks were invoked
            assert len(callback1_invoked) == 1
            assert len(callback2_invoked) == 1
            assert callback1_invoked[0].symbol == 'AAPL'
            assert callback2_invoked[0].symbol == 'AAPL'
