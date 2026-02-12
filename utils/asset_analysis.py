"""Asset-specific analysis utilities for multi-asset support."""

import logging
from typing import Dict, List, Optional
from enum import Enum

import pandas as pd
import numpy as np

from services.market_data_service import AssetClass

logger = logging.getLogger(__name__)


class AssetIndicatorType(Enum):
    """Types of indicators specific to asset classes"""
    STOCK_TECHNICAL = "stock_technical"
    CRYPTO_MOMENTUM = "crypto_momentum"
    CRYPTO_VOLATILITY = "crypto_volatility"
    FOREX_TREND = "forex_trend"


def calculate_crypto_specific_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate cryptocurrency-specific indicators.
    
    Crypto markets have unique characteristics:
    - 24/7 trading
    - Higher volatility
    - Different market microstructure
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional crypto-specific indicators
    """
    result = df.copy()
    
    # Realized volatility (24-hour rolling)
    if 'close' in result.columns:
        returns = result['close'].pct_change()
        result['realized_vol_24h'] = returns.rolling(window=24).std() * np.sqrt(24)
        
        # Crypto-specific momentum (shorter periods due to 24/7 trading)
        result['momentum_6h'] = result['close'].pct_change(periods=6)
        result['momentum_24h'] = result['close'].pct_change(periods=24)
        
        # Volume-weighted momentum
        if 'volume' in result.columns:
            result['vwap'] = (result['close'] * result['volume']).rolling(window=24).sum() / \
                            result['volume'].rolling(window=24).sum()
            result['vwap_distance'] = (result['close'] - result['vwap']) / result['vwap']
    
    # Trading intensity (unique to crypto 24/7 markets)
    if 'volume' in result.columns:
        result['volume_ma_24h'] = result['volume'].rolling(window=24).mean()
        result['volume_ratio'] = result['volume'] / result['volume_ma_24h']
    
    return result


def calculate_forex_specific_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate forex-specific indicators.
    
    Forex markets have unique characteristics:
    - Currency pair relationships
    - Interest rate differentials
    - Carry trade dynamics
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        DataFrame with additional forex-specific indicators
    """
    result = df.copy()
    
    # Forex-specific trend indicators
    if 'close' in result.columns:
        # Shorter-term trends for forex
        result['ema_8'] = result['close'].ewm(span=8, adjust=False).mean()
        result['ema_21'] = result['close'].ewm(span=21, adjust=False).mean()
        result['ema_55'] = result['close'].ewm(span=55, adjust=False).mean()
        
        # Trend strength
        result['trend_strength'] = (result['ema_8'] - result['ema_55']) / result['ema_55']
        
        # Average True Range for forex (pip-based volatility)
        if all(col in result.columns for col in ['high', 'low', 'close']):
            high_low = result['high'] - result['low']
            high_close = np.abs(result['high'] - result['close'].shift())
            low_close = np.abs(result['low'] - result['close'].shift())
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            result['atr_14'] = true_range.rolling(window=14).mean()
            result['atr_pct'] = result['atr_14'] / result['close']
    
    return result


def calculate_asset_specific_indicators(
    df: pd.DataFrame,
    asset_class: AssetClass
) -> pd.DataFrame:
    """
    Calculate indicators specific to the asset class.
    
    Args:
        df: DataFrame with OHLCV data
        asset_class: Asset class type
        
    Returns:
        DataFrame with asset-specific indicators added
    """
    if asset_class == AssetClass.CRYPTO:
        return calculate_crypto_specific_indicators(df)
    elif asset_class == AssetClass.FOREX:
        return calculate_forex_specific_indicators(df)
    else:
        # For stocks, use standard technical indicators
        return df


def calculate_correlation_matrix(
    price_data: Dict[str, pd.Series],
    window: int = 30
) -> pd.DataFrame:
    """
    Calculate rolling correlation matrix for multiple assets.
    
    Args:
        price_data: Dictionary mapping symbols to price series
        window: Rolling window for correlation calculation
        
    Returns:
        Correlation matrix DataFrame
    """
    # Combine all price series into a single DataFrame
    df = pd.DataFrame(price_data)
    
    # Calculate returns
    returns = df.pct_change().dropna()
    
    # Calculate correlation matrix
    if window:
        # Rolling correlation (use most recent window)
        corr_matrix = returns.tail(window).corr()
    else:
        # Full period correlation
        corr_matrix = returns.corr()
    
    return corr_matrix


def calculate_portfolio_allocation(
    positions: Dict[str, Dict],
    total_value: float
) -> Dict[AssetClass, float]:
    """
    Calculate portfolio allocation by asset class.
    
    Args:
        positions: Dictionary mapping symbols to position info
                  Each position should have 'value' and 'asset_class' keys
        total_value: Total portfolio value
        
    Returns:
        Dictionary mapping asset classes to allocation percentages
    """
    allocation = {
        AssetClass.STOCK: 0.0,
        AssetClass.CRYPTO: 0.0,
        AssetClass.FOREX: 0.0
    }
    
    if total_value <= 0:
        return allocation
    
    for symbol, position in positions.items():
        asset_class = position.get('asset_class', AssetClass.STOCK)
        value = position.get('value', 0.0)
        
        if isinstance(asset_class, str):
            asset_class = AssetClass(asset_class)
        
        allocation[asset_class] += value / total_value
    
    return allocation


def get_asset_class_risk_metrics(
    returns: pd.Series,
    asset_class: AssetClass
) -> Dict[str, float]:
    """
    Calculate risk metrics adjusted for asset class characteristics.
    
    Different asset classes have different risk profiles:
    - Stocks: Standard volatility, Sharpe ratio
    - Crypto: Higher volatility, 24/7 trading
    - Forex: Lower volatility, leverage considerations
    
    Args:
        returns: Series of returns
        asset_class: Asset class type
        
    Returns:
        Dictionary of risk metrics
    """
    metrics = {}
    
    # Basic metrics
    metrics['volatility'] = returns.std()
    metrics['mean_return'] = returns.mean()
    
    # Annualization factors differ by asset class
    if asset_class == AssetClass.CRYPTO:
        # Crypto trades 24/7, so more periods per year
        annual_factor = 365 * 24  # Hourly data
        metrics['annualized_vol'] = metrics['volatility'] * np.sqrt(annual_factor)
        metrics['annualized_return'] = metrics['mean_return'] * annual_factor
    elif asset_class == AssetClass.FOREX:
        # Forex trades 24/5
        annual_factor = 252 * 24  # Hourly data, 252 trading days
        metrics['annualized_vol'] = metrics['volatility'] * np.sqrt(annual_factor)
        metrics['annualized_return'] = metrics['mean_return'] * annual_factor
    else:
        # Stocks trade during market hours
        annual_factor = 252  # Daily data
        metrics['annualized_vol'] = metrics['volatility'] * np.sqrt(annual_factor)
        metrics['annualized_return'] = metrics['mean_return'] * annual_factor
    
    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
    if metrics['annualized_vol'] > 0:
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['annualized_vol']
    else:
        metrics['sharpe_ratio'] = 0.0
    
    # Downside deviation
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0:
        metrics['downside_deviation'] = downside_returns.std()
        metrics['sortino_ratio'] = metrics['mean_return'] / metrics['downside_deviation'] if metrics['downside_deviation'] > 0 else 0.0
    else:
        metrics['downside_deviation'] = 0.0
        metrics['sortino_ratio'] = 0.0
    
    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    metrics['max_drawdown'] = drawdown.min()
    
    return metrics


def suggest_asset_allocation(
    risk_tolerance: str,
    current_allocation: Dict[AssetClass, float]
) -> Dict[AssetClass, float]:
    """
    Suggest optimal asset allocation based on risk tolerance.
    
    Args:
        risk_tolerance: 'conservative', 'moderate', or 'aggressive'
        current_allocation: Current allocation by asset class
        
    Returns:
        Suggested allocation by asset class
    """
    # Target allocations by risk profile
    targets = {
        'conservative': {
            AssetClass.STOCK: 0.70,
            AssetClass.CRYPTO: 0.05,
            AssetClass.FOREX: 0.25
        },
        'moderate': {
            AssetClass.STOCK: 0.60,
            AssetClass.CRYPTO: 0.15,
            AssetClass.FOREX: 0.25
        },
        'aggressive': {
            AssetClass.STOCK: 0.40,
            AssetClass.CRYPTO: 0.35,
            AssetClass.FOREX: 0.25
        }
    }
    
    if risk_tolerance not in targets:
        risk_tolerance = 'moderate'
    
    return targets[risk_tolerance]


def calculate_diversification_ratio(
    positions: Dict[str, Dict],
    correlation_matrix: pd.DataFrame
) -> float:
    """
    Calculate portfolio diversification ratio.
    
    Higher ratio indicates better diversification.
    
    Args:
        positions: Dictionary of positions with weights
        correlation_matrix: Correlation matrix of returns
        
    Returns:
        Diversification ratio
    """
    # Extract weights
    symbols = list(positions.keys())
    weights = np.array([positions[s].get('weight', 0.0) for s in symbols])
    
    # Ensure weights sum to 1
    if weights.sum() > 0:
        weights = weights / weights.sum()
    else:
        return 0.0
    
    # Calculate weighted average volatility
    volatilities = np.array([positions[s].get('volatility', 0.0) for s in symbols])
    weighted_vol = np.dot(weights, volatilities)
    
    # Calculate portfolio volatility using correlation matrix
    # Filter correlation matrix to only include our symbols
    corr_subset = correlation_matrix.loc[symbols, symbols].values
    
    # Portfolio variance = w^T * Σ * w where Σ is covariance matrix
    # For simplicity, assume equal volatilities and use correlation as proxy
    portfolio_variance = np.dot(weights, np.dot(corr_subset, weights))
    portfolio_vol = np.sqrt(portfolio_variance)
    
    # Diversification ratio
    if portfolio_vol > 0:
        return weighted_vol / portfolio_vol
    else:
        return 0.0
