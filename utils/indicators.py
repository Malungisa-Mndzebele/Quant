"""
Technical indicators for market analysis.

This module provides implementations of common technical indicators used in
trading analysis, including moving averages, momentum indicators, and volatility measures.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple


def sma(data: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
    """
    Calculate Simple Moving Average (SMA).
    
    The SMA is the unweighted mean of the previous n data points.
    
    Args:
        data: Price data (typically close prices)
        period: Number of periods for the moving average
        
    Returns:
        SMA values (same type as input)
    """
    if isinstance(data, pd.Series):
        return data.rolling(window=period).mean()
    else:
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result


def ema(data: Union[pd.Series, np.ndarray], period: int) -> Union[pd.Series, np.ndarray]:
    """
    Calculate Exponential Moving Average (EMA).
    
    The EMA gives more weight to recent prices, making it more responsive
    to new information than the SMA.
    
    Args:
        data: Price data (typically close prices)
        period: Number of periods for the moving average
        
    Returns:
        EMA values (same type as input)
    """
    if isinstance(data, pd.Series):
        return data.ewm(span=period, adjust=False).mean()
    else:
        result = np.full(len(data), np.nan)
        multiplier = 2 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result


def rsi(data: Union[pd.Series, np.ndarray], period: int = 14) -> Union[pd.Series, np.ndarray]:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI is a momentum oscillator that measures the speed and magnitude of
    price changes. Values range from 0 to 100.
    
    Args:
        data: Price data (typically close prices)
        period: Number of periods for RSI calculation (default: 14)
        
    Returns:
        RSI values (same type as input)
    """
    if isinstance(data, pd.Series):
        delta = data.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi_values = 100 - (100 / (1 + rs))
        return rsi_values
    else:
        delta = np.diff(data, prepend=data[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        result = np.full(len(data), np.nan)
        for i in range(period, len(data)):
            avg_gain = np.mean(gain[i - period + 1:i + 1])
            avg_loss = np.mean(loss[i - period + 1:i + 1])
            if avg_loss == 0:
                result[i] = 100
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
        return result


def macd(data: Union[pd.Series, np.ndarray], 
         fast_period: int = 12, 
         slow_period: int = 26, 
         signal_period: int = 9) -> Tuple[Union[pd.Series, np.ndarray], 
                                           Union[pd.Series, np.ndarray], 
                                           Union[pd.Series, np.ndarray]]:
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    MACD is a trend-following momentum indicator that shows the relationship
    between two moving averages of prices.
    
    Args:
        data: Price data (typically close prices)
        fast_period: Period for fast EMA (default: 12)
        slow_period: Period for slow EMA (default: 26)
        signal_period: Period for signal line (default: 9)
        
    Returns:
        Tuple of (MACD line, Signal line, MACD histogram)
    """
    fast_ema = ema(data, fast_period)
    slow_ema = ema(data, slow_period)
    if isinstance(data, pd.Series):
        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
    else:
        macd_line = fast_ema - slow_ema
        signal_line = ema(pd.Series(macd_line), signal_period).values
        histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(data: Union[pd.Series, np.ndarray], 
                   period: int = 20, 
                   num_std: float = 2.0) -> Tuple[Union[pd.Series, np.ndarray], 
                                                   Union[pd.Series, np.ndarray], 
                                                   Union[pd.Series, np.ndarray]]:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of a middle band (SMA) and two outer bands
    (standard deviations away from the middle band).
    
    Args:
        data: Price data (typically close prices)
        period: Number of periods for the moving average (default: 20)
        num_std: Number of standard deviations for bands (default: 2.0)
        
    Returns:
        Tuple of (upper band, middle band, lower band)
    """
    if isinstance(data, pd.Series):
        middle_band = data.rolling(window=period).mean()
        std_dev = data.rolling(window=period).std()
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
    else:
        middle_band = np.full(len(data), np.nan)
        upper_band = np.full(len(data), np.nan)
        lower_band = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            window = data[i - period + 1:i + 1]
            mean = np.mean(window)
            std = np.std(window, ddof=1)
            middle_band[i] = mean
            upper_band[i] = mean + (std * num_std)
            lower_band[i] = mean - (std * num_std)
    return upper_band, middle_band, lower_band


def atr(high: Union[pd.Series, np.ndarray], 
        low: Union[pd.Series, np.ndarray], 
        close: Union[pd.Series, np.ndarray], 
        period: int = 14) -> Union[pd.Series, np.ndarray]:
    """
    Calculate Average True Range (ATR).
    
    ATR is a volatility indicator that measures the average range between
    high and low prices over a specified period, accounting for gaps.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Number of periods for ATR calculation (default: 14)
        
    Returns:
        ATR values (same type as input)
    """
    if isinstance(high, pd.Series):
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_values = true_range.ewm(span=period, adjust=False).mean()
        return atr_values
    else:
        true_range = np.full(len(high), np.nan)
        true_range[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr1 = high[i] - low[i]
            tr2 = abs(high[i] - close[i - 1])
            tr3 = abs(low[i] - close[i - 1])
            true_range[i] = max(tr1, tr2, tr3)
        atr_values = ema(pd.Series(true_range), period).values
        return atr_values
