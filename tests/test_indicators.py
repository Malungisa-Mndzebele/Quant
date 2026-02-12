"""
Unit tests for technical indicators.

Tests each indicator with known inputs/outputs and edge cases including
insufficient data and NaN values.
"""

import pytest
import numpy as np
import pandas as pd
from utils.indicators import sma, ema, rsi, macd, bollinger_bands, atr


class TestSMA:
    """Tests for Simple Moving Average."""
    
    def test_sma_with_series(self):
        """Test SMA calculation with pandas Series."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = sma(data, period=3)
        
        # First two values should be NaN
        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        
        # Check known values
        assert result.iloc[2] == 2.0  # (1+2+3)/3
        assert result.iloc[3] == 3.0  # (2+3+4)/3
        assert result.iloc[9] == 9.0  # (8+9+10)/3
    
    def test_sma_with_array(self):
        """Test SMA calculation with numpy array."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = sma(data, period=3)
        
        # First two values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # Check known values
        assert result[2] == 2.0
        assert result[3] == 3.0
        assert result[9] == 9.0
    
    def test_sma_insufficient_data(self):
        """Test SMA with insufficient data points."""
        data = pd.Series([1, 2])
        result = sma(data, period=5)
        
        # All values should be NaN when data < period
        assert result.isna().all()
    
    def test_sma_with_nan_values(self):
        """Test SMA handles NaN values in input."""
        data = pd.Series([1, 2, np.nan, 4, 5])
        result = sma(data, period=3)
        
        # Result should contain NaN where input has NaN
        assert pd.isna(result.iloc[2])
    
    def test_sma_single_period(self):
        """Test SMA with period=1 returns original data."""
        data = pd.Series([1, 2, 3, 4, 5])
        result = sma(data, period=1)
        
        # Values should match, but dtype may differ (float vs int)
        assert (result == data).all()


class TestEMA:
    """Tests for Exponential Moving Average."""
    
    def test_ema_with_series(self):
        """Test EMA calculation with pandas Series."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = ema(data, period=3)
        
        # First values should exist (EMA starts immediately)
        assert not pd.isna(result.iloc[0])
        
        # EMA should be more responsive than SMA
        # Last value should be closer to recent prices
        assert result.iloc[9] > 8.0
    
    def test_ema_with_array(self):
        """Test EMA calculation with numpy array."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        result = ema(data, period=3)
        
        # First period-1 values should be NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # After period, values should exist
        assert not np.isnan(result[2])
    
    def test_ema_insufficient_data(self):
        """Test EMA with insufficient data points."""
        data = pd.Series([1, 2])
        result = ema(data, period=5)
        
        # Should still calculate with available data
        assert not result.isna().all()
    
    def test_ema_with_nan_values(self):
        """Test EMA handles NaN values in input."""
        data = pd.Series([1, 2, np.nan, 4, 5])
        result = ema(data, period=3)
        
        # EMA with pandas ewm() interpolates through NaN values
        # So we just check that result is computed
        assert not result.isna().all()


class TestRSI:
    """Tests for Relative Strength Index."""
    
    def test_rsi_with_series(self):
        """Test RSI calculation with pandas Series."""
        # Create data with clear uptrend
        data = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        result = rsi(data, period=14)
        
        # RSI should be high (>70) for strong uptrend
        assert result.iloc[-1] > 70
        
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()
    
    def test_rsi_with_array(self):
        """Test RSI calculation with numpy array."""
        data = np.array([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype=float)
        result = rsi(data, period=14)
        
        # RSI should be between 0 and 100
        valid_values = result[~np.isnan(result)]
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()
    
    def test_rsi_downtrend(self):
        """Test RSI with downtrend data."""
        # Create data with clear downtrend
        data = pd.Series([24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10])
        result = rsi(data, period=14)
        
        # RSI should be low (<30) for strong downtrend
        assert result.iloc[-1] < 30
    
    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data points."""
        data = pd.Series([1, 2, 3, 4, 5])
        result = rsi(data, period=14)
        
        # All values should be NaN when data < period
        assert result.isna().all()
    
    def test_rsi_with_nan_values(self):
        """Test RSI handles NaN values in input."""
        data = pd.Series([10, 11, np.nan, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
        result = rsi(data, period=14)
        
        # Result should handle NaN appropriately
        assert pd.isna(result.iloc[2])
    
    def test_rsi_no_change(self):
        """Test RSI with flat prices (no change)."""
        data = pd.Series([10] * 20)
        result = rsi(data, period=14)
        
        # RSI should be NaN or 50 when no price movement
        valid_values = result.dropna()
        if len(valid_values) > 0:
            # When there's no movement, RSI is undefined or 50
            assert valid_values.isna().all() or (valid_values == 50).all() or valid_values.isna().all()


class TestMACD:
    """Tests for Moving Average Convergence Divergence."""
    
    def test_macd_with_series(self):
        """Test MACD calculation with pandas Series."""
        data = pd.Series(range(1, 51))  # Uptrend
        macd_line, signal_line, histogram = macd(data)
        
        # All outputs should be same length as input
        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)
        
        # Histogram should equal MACD - Signal
        valid_idx = ~pd.isna(histogram)
        pd.testing.assert_series_equal(
            histogram[valid_idx],
            (macd_line - signal_line)[valid_idx]
        )
    
    def test_macd_with_array(self):
        """Test MACD calculation with numpy array."""
        data = np.arange(1, 51, dtype=float)
        macd_line, signal_line, histogram = macd(data)
        
        # All outputs should be same length as input
        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)
    
    def test_macd_custom_periods(self):
        """Test MACD with custom periods."""
        data = pd.Series(range(1, 51))
        macd_line, signal_line, histogram = macd(data, fast_period=5, slow_period=10, signal_period=5)
        
        # Should still produce valid output
        assert not macd_line.isna().all()
        assert not signal_line.isna().all()
    
    def test_macd_insufficient_data(self):
        """Test MACD with insufficient data points."""
        data = pd.Series([1, 2, 3, 4, 5])
        macd_line, signal_line, histogram = macd(data)
        
        # MACD can still compute with limited data, just less reliable
        # Check that outputs are generated
        assert len(macd_line) == len(data)
        assert len(signal_line) == len(data)
        assert len(histogram) == len(data)
    
    def test_macd_with_nan_values(self):
        """Test MACD handles NaN values in input."""
        data = pd.Series(list(range(1, 26)) + [np.nan] + list(range(27, 51)))
        macd_line, signal_line, histogram = macd(data)
        
        # MACD with pandas ewm() interpolates through NaN values
        # Check that computation completes
        assert len(macd_line) == len(data)
        assert not macd_line.isna().all()


class TestBollingerBands:
    """Tests for Bollinger Bands."""
    
    def test_bollinger_bands_with_series(self):
        """Test Bollinger Bands calculation with pandas Series."""
        data = pd.Series([10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14, 
                         15, 14, 13, 14, 15, 16, 15, 14, 15, 16])
        upper, middle, lower = bollinger_bands(data, period=20, num_std=2.0)
        
        # All outputs should be same length as input
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)
        
        # Upper should be > middle > lower (where not NaN)
        valid_idx = ~pd.isna(middle)
        assert (upper[valid_idx] > middle[valid_idx]).all()
        assert (middle[valid_idx] > lower[valid_idx]).all()
    
    def test_bollinger_bands_with_array(self):
        """Test Bollinger Bands calculation with numpy array."""
        data = np.array([10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14, 
                        15, 14, 13, 14, 15, 16, 15, 14, 15, 16], dtype=float)
        upper, middle, lower = bollinger_bands(data, period=20, num_std=2.0)
        
        # All outputs should be same length as input
        assert len(upper) == len(data)
        assert len(middle) == len(data)
        assert len(lower) == len(data)
        
        # Upper should be > middle > lower (where not NaN)
        valid_idx = ~np.isnan(middle)
        assert (upper[valid_idx] > middle[valid_idx]).all()
        assert (middle[valid_idx] > lower[valid_idx]).all()
    
    def test_bollinger_bands_custom_std(self):
        """Test Bollinger Bands with custom standard deviation."""
        data = pd.Series([10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14, 
                         15, 14, 13, 14, 15, 16, 15, 14, 15, 16])
        upper_2std, middle_2std, lower_2std = bollinger_bands(data, period=20, num_std=2.0)
        upper_1std, middle_1std, lower_1std = bollinger_bands(data, period=20, num_std=1.0)
        
        # Middle bands should be the same
        pd.testing.assert_series_equal(middle_2std, middle_1std)
        
        # 2 std bands should be wider than 1 std bands
        valid_idx = ~pd.isna(middle_2std)
        assert ((upper_2std - lower_2std)[valid_idx] > (upper_1std - lower_1std)[valid_idx]).all()
    
    def test_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data points."""
        data = pd.Series([1, 2, 3, 4, 5])
        upper, middle, lower = bollinger_bands(data, period=20)
        
        # All values should be NaN when data < period
        assert upper.isna().all()
        assert middle.isna().all()
        assert lower.isna().all()
    
    def test_bollinger_bands_with_nan_values(self):
        """Test Bollinger Bands handles NaN values in input."""
        data = pd.Series([10, 11, 12, 11, 10, 11, 12, 13, 12, 11, np.nan, 11, 12, 13, 14, 
                         15, 14, 13, 14, 15, 16, 15, 14, 15, 16])
        upper, middle, lower = bollinger_bands(data, period=20)
        
        # Result should contain NaN where input has NaN
        assert pd.isna(middle.iloc[10])


class TestATR:
    """Tests for Average True Range."""
    
    def test_atr_with_series(self):
        """Test ATR calculation with pandas Series."""
        high = pd.Series([12, 13, 14, 13, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15, 16])
        low = pd.Series([10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14])
        close = pd.Series([11, 12, 13, 12, 11, 12, 13, 14, 13, 12, 11, 12, 13, 14, 15])
        
        result = atr(high, low, close, period=14)
        
        # ATR should be positive
        valid_values = result.dropna()
        assert (valid_values > 0).all()
        
        # Length should match input
        assert len(result) == len(high)
    
    def test_atr_with_array(self):
        """Test ATR calculation with numpy array."""
        high = np.array([12, 13, 14, 13, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15, 16], dtype=float)
        low = np.array([10, 11, 12, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14], dtype=float)
        close = np.array([11, 12, 13, 12, 11, 12, 13, 14, 13, 12, 11, 12, 13, 14, 15], dtype=float)
        
        result = atr(high, low, close, period=14)
        
        # ATR should be positive
        valid_values = result[~np.isnan(result)]
        assert (valid_values > 0).all()
        
        # Length should match input
        assert len(result) == len(high)
    
    def test_atr_with_gaps(self):
        """Test ATR accounts for price gaps."""
        # Create data with a gap
        high = pd.Series([12, 13, 14, 20, 21])  # Gap up
        low = pd.Series([10, 11, 12, 18, 19])
        close = pd.Series([11, 12, 13, 19, 20])
        
        result = atr(high, low, close, period=3)
        
        # ATR should increase after gap
        assert result.iloc[-1] > result.iloc[2]
    
    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data points."""
        high = pd.Series([12, 13, 14])
        low = pd.Series([10, 11, 12])
        close = pd.Series([11, 12, 13])
        
        result = atr(high, low, close, period=14)
        
        # ATR can still compute with limited data
        # Check that result is generated
        assert len(result) == len(high)
        assert not result.isna().all()
    
    def test_atr_with_nan_values(self):
        """Test ATR handles NaN values in input."""
        high = pd.Series([12, 13, np.nan, 13, 12, 13, 14, 15, 14, 13, 12, 13, 14, 15, 16])
        low = pd.Series([10, 11, np.nan, 11, 10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14])
        close = pd.Series([11, 12, np.nan, 12, 11, 12, 13, 14, 13, 12, 11, 12, 13, 14, 15])
        
        result = atr(high, low, close, period=14)
        
        # ATR with pandas ewm() interpolates through NaN values
        # Check that computation completes
        assert len(result) == len(high)
        assert not result.isna().all()
    
    def test_atr_equal_high_low(self):
        """Test ATR when high equals low (no volatility)."""
        high = pd.Series([10] * 15)
        low = pd.Series([10] * 15)
        close = pd.Series([10] * 15)
        
        result = atr(high, low, close, period=14)
        
        # ATR should be 0 or very close to 0 when no volatility
        valid_values = result.dropna()
        assert (valid_values < 0.01).all()


class TestEdgeCases:
    """Test edge cases across all indicators."""
    
    def test_empty_input(self):
        """Test indicators with empty input."""
        empty_series = pd.Series([], dtype=float)
        empty_array = np.array([], dtype=float)
        
        # SMA
        result = sma(empty_series, period=5)
        assert len(result) == 0
        
        # EMA
        result = ema(empty_series, period=5)
        assert len(result) == 0
        
        # RSI
        result = rsi(empty_series, period=14)
        assert len(result) == 0
    
    def test_single_value(self):
        """Test indicators with single value."""
        single_series = pd.Series([10.0])
        
        # SMA with period > 1 should return NaN
        result = sma(single_series, period=5)
        assert pd.isna(result.iloc[0])
        
        # SMA with period = 1 should return the value
        result = sma(single_series, period=1)
        assert result.iloc[0] == 10.0
    
    def test_all_nan_input(self):
        """Test indicators with all NaN input."""
        nan_series = pd.Series([np.nan] * 20)
        
        # SMA
        result = sma(nan_series, period=5)
        assert result.isna().all()
        
        # EMA
        result = ema(nan_series, period=5)
        assert result.isna().all()
        
        # RSI
        result = rsi(nan_series, period=14)
        assert result.isna().all()
    
    def test_negative_values(self):
        """Test indicators handle negative values."""
        data = pd.Series([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
        
        # SMA should work with negative values
        result = sma(data, period=3)
        assert result.iloc[2] == -4.0  # (-5 + -4 + -3) / 3
        
        # EMA should work with negative values
        result = ema(data, period=3)
        assert not result.isna().all()
    
    def test_very_large_values(self):
        """Test indicators handle very large values."""
        data = pd.Series([1e10, 1e10 + 1, 1e10 + 2, 1e10 + 3, 1e10 + 4])
        
        # SMA should handle large values
        result = sma(data, period=3)
        assert not result.isna().all()
        assert np.isfinite(result.dropna()).all()
    
    def test_very_small_values(self):
        """Test indicators handle very small values."""
        data = pd.Series([1e-10, 2e-10, 3e-10, 4e-10, 5e-10])
        
        # SMA should handle small values
        result = sma(data, period=3)
        assert not result.isna().all()
        assert np.isfinite(result.dropna()).all()
