"""Unit tests for data feed components."""
import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import tempfile
import shutil
from pathlib import Path

from src.data import YFinanceProvider, DataCache, Quote
from src.utils.exceptions import DataError


class TestYFinanceProvider:
    """Test YFinance provider implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.provider = YFinanceProvider()
    
    @patch('src.data.yfinance_provider.yf.Ticker')
    def test_get_quote_success(self, mock_ticker):
        """Test successful quote retrieval."""
        # Mock yfinance response
        mock_info = {
            'regularMarketPrice': 150.0,
            'bid': 149.9,
            'ask': 150.1,
            'regularMarketVolume': 1000000
        }
        mock_ticker.return_value.info = mock_info
        
        # Get quote
        quote = self.provider.get_quote('AAPL')
        
        # Assertions
        assert isinstance(quote, Quote)
        assert quote.symbol == 'AAPL'
        assert quote.price == 150.0
        assert quote.bid == 149.9
        assert quote.ask == 150.1
        assert quote.volume == 1000000
        assert isinstance(quote.timestamp, datetime)
    
    @patch('src.data.yfinance_provider.yf.Ticker')
    def test_get_quote_with_current_price(self, mock_ticker):
        """Test quote retrieval using currentPrice fallback."""
        # Mock yfinance response with currentPrice instead of regularMarketPrice
        mock_info = {
            'currentPrice': 150.0,
            'bid': 149.9,
            'ask': 150.1,
            'volume': 1000000
        }
        mock_ticker.return_value.info = mock_info
        
        # Get quote
        quote = self.provider.get_quote('AAPL')
        
        # Assertions
        assert quote.price == 150.0
    
    @patch('src.data.yfinance_provider.yf.Ticker')
    def test_get_quote_invalid_symbol(self, mock_ticker):
        """Test quote retrieval with invalid symbol."""
        # Mock empty response
        mock_ticker.return_value.info = {}
        
        # Should raise DataError
        with pytest.raises(DataError) as exc_info:
            self.provider.get_quote('INVALID')
        
        assert 'No quote data available' in str(exc_info.value)
    
    @patch('src.data.yfinance_provider.yf.Ticker')
    def test_get_quote_api_failure(self, mock_ticker):
        """Test quote retrieval with API failure."""
        # Mock API exception
        mock_ticker.side_effect = Exception('API connection failed')
        
        # Should raise DataError
        with pytest.raises(DataError) as exc_info:
            self.provider.get_quote('AAPL')
        
        assert 'Failed to retrieve quote' in str(exc_info.value)
    
    @patch('src.data.yfinance_provider.yf.Ticker')
    def test_get_historical_data_success(self, mock_ticker):
        """Test successful historical data retrieval."""
        # Create mock historical data
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        mock_data = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0, 151.5, 153.0],
            'High': [151.0, 152.0, 153.0, 152.5, 154.0],
            'Low': [149.0, 150.0, 151.0, 150.5, 152.0],
            'Close': [150.5, 151.5, 152.5, 151.0, 153.5],
            'Volume': [1000000, 1100000, 1200000, 1050000, 1150000]
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = mock_data
        
        # Get historical data
        start = date(2024, 1, 1)
        end = date(2024, 1, 5)
        df = self.provider.get_historical_data('AAPL', start, end)
        
        # Assertions
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert list(df.columns) == ['Open', 'High', 'Low', 'Close', 'Volume']
        assert df['Close'].iloc[0] == 150.5
    
    @patch('src.data.yfinance_provider.yf.Ticker')
    def test_get_historical_data_invalid_date_range(self, mock_ticker):
        """Test historical data with invalid date range."""
        # Start date after end date
        start = date(2024, 1, 10)
        end = date(2024, 1, 1)
        
        # Should raise DataError
        with pytest.raises(DataError) as exc_info:
            self.provider.get_historical_data('AAPL', start, end)
        
        assert 'Start date' in str(exc_info.value)
        assert 'cannot be after end date' in str(exc_info.value)
    
    @patch('src.data.yfinance_provider.yf.Ticker')
    def test_get_historical_data_no_data(self, mock_ticker):
        """Test historical data retrieval with no data available."""
        # Mock empty DataFrame
        mock_ticker.return_value.history.return_value = pd.DataFrame()
        
        # Should raise DataError
        start = date(2024, 1, 1)
        end = date(2024, 1, 5)
        
        with pytest.raises(DataError) as exc_info:
            self.provider.get_historical_data('AAPL', start, end)
        
        assert 'No historical data available' in str(exc_info.value)
    
    @patch('src.data.yfinance_provider.yf.Ticker')
    def test_get_historical_data_missing_columns(self, mock_ticker):
        """Test historical data with missing required columns."""
        # Create DataFrame with missing columns
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        mock_data = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0, 151.5, 153.0],
            'Close': [150.5, 151.5, 152.5, 151.0, 153.5]
        }, index=dates)
        
        mock_ticker.return_value.history.return_value = mock_data
        
        # Should raise DataError
        start = date(2024, 1, 1)
        end = date(2024, 1, 5)
        
        with pytest.raises(DataError) as exc_info:
            self.provider.get_historical_data('AAPL', start, end)
        
        assert 'Missing required columns' in str(exc_info.value)


class TestDataCache:
    """Test data caching mechanism."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create temporary cache directory
        self.temp_dir = tempfile.mkdtemp()
        self.cache = DataCache(cache_dir=self.temp_dir, max_age_days=1)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_cache_initialization(self):
        """Test cache initialization creates directory."""
        assert Path(self.temp_dir).exists()
        assert self.cache.cache_dir == Path(self.temp_dir)
        assert self.cache.max_age_days == 1
    
    def test_cache_and_retrieve_data(self):
        """Test caching and retrieving data."""
        # Create test data
        dates = pd.date_range(start='2024-01-01', end='2024-01-05', freq='D')
        test_data = pd.DataFrame({
            'Open': [150.0, 151.0, 152.0, 151.5, 153.0],
            'High': [151.0, 152.0, 153.0, 152.5, 154.0],
            'Low': [149.0, 150.0, 151.0, 150.5, 152.0],
            'Close': [150.5, 151.5, 152.5, 151.0, 153.5],
            'Volume': [1000000, 1100000, 1200000, 1050000, 1150000]
        }, index=dates)
        
        # Cache data
        start = date(2024, 1, 1)
        end = date(2024, 1, 5)
        self.cache.cache_data('AAPL', test_data, start, end)
        
        # Retrieve cached data
        cached_data = self.cache.get_cached_data('AAPL', start, end)
        
        # Assertions
        assert cached_data is not None
        assert isinstance(cached_data, pd.DataFrame)
        assert len(cached_data) == len(test_data)
        pd.testing.assert_frame_equal(cached_data, test_data)
    
    def test_get_cached_data_not_found(self):
        """Test retrieving non-existent cached data."""
        start = date(2024, 1, 1)
        end = date(2024, 1, 5)
        
        # Should return None
        cached_data = self.cache.get_cached_data('AAPL', start, end)
        assert cached_data is None
    
    def test_cache_expiration(self):
        """Test cache expiration based on age."""
        # Create cache with 0 day max age (immediate expiration)
        cache = DataCache(cache_dir=self.temp_dir, max_age_days=0)
        
        # Create and cache test data
        test_data = pd.DataFrame({'Close': [150.0]})
        start = date(2024, 1, 1)
        end = date(2024, 1, 5)
        cache.cache_data('AAPL', test_data, start, end)
        
        # Should return None due to expiration
        cached_data = cache.get_cached_data('AAPL', start, end)
        assert cached_data is None
    
    def test_clear_cache_specific_symbol(self):
        """Test clearing cache for specific symbol."""
        # Cache data for multiple symbols
        test_data = pd.DataFrame({'Close': [150.0]})
        start = date(2024, 1, 1)
        end = date(2024, 1, 5)
        
        self.cache.cache_data('AAPL', test_data, start, end)
        self.cache.cache_data('GOOGL', test_data, start, end)
        
        # Clear cache for AAPL only
        count = self.cache.clear_cache('AAPL')
        
        # Assertions
        assert count == 1
        assert self.cache.get_cached_data('AAPL', start, end) is None
        assert self.cache.get_cached_data('GOOGL', start, end) is not None
    
    def test_clear_all_cache(self):
        """Test clearing all cached data."""
        # Cache data for multiple symbols
        test_data = pd.DataFrame({'Close': [150.0]})
        start = date(2024, 1, 1)
        end = date(2024, 1, 5)
        
        self.cache.cache_data('AAPL', test_data, start, end)
        self.cache.cache_data('GOOGL', test_data, start, end)
        
        # Clear all cache
        count = self.cache.clear_cache()
        
        # Assertions
        assert count == 2
        assert self.cache.get_cached_data('AAPL', start, end) is None
        assert self.cache.get_cached_data('GOOGL', start, end) is None
    
    def test_get_cache_info(self):
        """Test getting cache information."""
        # Cache some data
        test_data = pd.DataFrame({'Close': [150.0]})
        start = date(2024, 1, 1)
        end = date(2024, 1, 5)
        
        self.cache.cache_data('AAPL', test_data, start, end)
        self.cache.cache_data('GOOGL', test_data, start, end)
        
        # Get cache info
        info = self.cache.get_cache_info()
        
        # Assertions
        assert info['total_files'] == 2
        assert info['valid_files'] == 2
        assert info['expired_files'] == 0
        assert info['total_size_bytes'] > 0
        assert info['cache_dir'] == str(self.cache.cache_dir)
    
    def test_cache_data_error_handling(self):
        """Test error handling when caching fails."""
        # Create a cache and then make the directory read-only to cause write failure
        cache = DataCache(cache_dir=self.temp_dir)
        
        # Mock the open function to raise an exception
        test_data = pd.DataFrame({'Close': [150.0]})
        start = date(2024, 1, 1)
        end = date(2024, 1, 5)
        
        # Patch the open function to simulate a write failure
        with patch('builtins.open', side_effect=PermissionError('Permission denied')):
            with pytest.raises(DataError) as exc_info:
                cache.cache_data('AAPL', test_data, start, end)
            
            assert 'Failed to cache data' in str(exc_info.value)
