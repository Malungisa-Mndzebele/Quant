"""
Feature engineering module for ML models.

This module provides feature extraction, transformation, and selection methods
for preparing market data for machine learning models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

# Import technical indicators
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.indicators import sma, ema, rsi, macd, bollinger_bands, atr


class FeatureEngineer:
    """Feature engineering class for creating ML-ready features from market data."""
    
    def __init__(self, scaler_type: str = 'standard'):
        """Initialize the feature engineer."""
        self.scaler_type = scaler_type
        self.scaler = self._create_scaler(scaler_type)
        self.feature_names: List[str] = []
        self.selected_features: Optional[List[str]] = None
        
    def _create_scaler(self, scaler_type: str):
        """Create the appropriate scaler based on type."""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-based features including returns and volatility."""
        result = df.copy()
        result['returns_1d'] = result['close'].pct_change(1)
        result['returns_5d'] = result['close'].pct_change(5)
        result['returns_10d'] = result['close'].pct_change(10)
        result['returns_20d'] = result['close'].pct_change(20)
        result['log_returns'] = np.log(result['close'] / result['close'].shift(1))
        result['volatility_5d'] = result['returns_1d'].rolling(window=5).std()
        result['volatility_10d'] = result['returns_1d'].rolling(window=10).std()
        result['volatility_20d'] = result['returns_1d'].rolling(window=20).std()
        result['momentum_5d'] = result['close'] - result['close'].shift(5)
        result['momentum_10d'] = result['close'] - result['close'].shift(10)
        result['momentum_20d'] = result['close'] - result['close'].shift(20)
        result['high_low_range'] = result['high'] - result['low']
        result['close_open_range'] = result['close'] - result['open']
        result['volume_change'] = result['volume'].pct_change(1)
        result['volume_ma_ratio'] = result['volume'] / result['volume'].rolling(window=20).mean()
        result['price_position'] = (result['close'] - result['low']) / (result['high'] - result['low'])
        return result
    
    def create_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features."""
        result = df.copy()
        result['sma_10'] = sma(result['close'], 10)
        result['sma_20'] = sma(result['close'], 20)
        result['sma_50'] = sma(result['close'], 50)
        result['ema_10'] = ema(result['close'], 10)
        result['ema_20'] = ema(result['close'], 20)
        result['price_sma20_ratio'] = result['close'] / result['sma_20']
        result['price_sma50_ratio'] = result['close'] / result['sma_50']
        result['rsi_14'] = rsi(result['close'], 14)
        macd_line, signal_line, histogram = macd(result['close'])
        result['macd'] = macd_line
        result['macd_signal'] = signal_line
        result['macd_histogram'] = histogram
        upper, middle, lower = bollinger_bands(result['close'], period=20)
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower
        result['bb_width'] = (upper - lower) / middle
        result['bb_position'] = (result['close'] - lower) / (upper - lower)
        result['atr_14'] = atr(result['high'], result['low'], result['close'], 14)
        return result
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features from datetime index."""
        result = df.copy()
        if not isinstance(result.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex")
        result['day_of_week'] = result.index.dayofweek
        result['hour'] = result.index.hour
        result['day_of_month'] = result.index.day
        result['month'] = result.index.month
        result['quarter'] = result.index.quarter
        result['is_month_start'] = result.index.is_month_start.astype(int)
        result['is_month_end'] = result.index.is_month_end.astype(int)
        result['is_quarter_start'] = result.index.is_quarter_start.astype(int)
        result['is_quarter_end'] = result.index.is_quarter_end.astype(int)
        return result
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create all features: price-based, technical, and time-based."""
        result = df.copy()
        result = self.create_price_features(result)
        result = self.create_technical_features(result)
        result = self.create_time_features(result)
        original_cols = ['open', 'high', 'low', 'close', 'volume']
        self.feature_names = [col for col in result.columns if col not in original_cols]
        return result
    
    def normalize_features(self, df: pd.DataFrame, feature_cols: Optional[List[str]] = None, fit: bool = True) -> pd.DataFrame:
        """Normalize features using the configured scaler."""
        result = df.copy()
        if feature_cols is None:
            feature_cols = result.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['open', 'high', 'low', 'close', 'volume']
            feature_cols = [col for col in feature_cols if col not in exclude_cols]
        if fit:
            result[feature_cols] = self.scaler.fit_transform(result[feature_cols])
        else:
            result[feature_cols] = self.scaler.transform(result[feature_cols])
        return result
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20, method: str = 'f_classif') -> pd.DataFrame:
        """Select top k features using statistical tests."""
        valid_idx = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]
        if method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"Unknown method: {method}")
        selector = SelectKBest(score_func=score_func, k=min(k, X_clean.shape[1]))
        selector.fit(X_clean, y_clean)
        selected_mask = selector.get_support()
        self.selected_features = X.columns[selected_mask].tolist()
        return X[self.selected_features]
    
    def prepare_features(self, df: pd.DataFrame, target_col: Optional[str] = None, normalize: bool = True, select_k: Optional[int] = None) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Complete feature preparation pipeline."""
        result = self.create_all_features(df)
        result = result.dropna()
        target = None
        if target_col and target_col in result.columns:
            target = result[target_col]
            result = result.drop(columns=[target_col])
        if normalize:
            result = self.normalize_features(result, fit=True)
        if select_k and target is not None:
            result = self.select_features(result, target, k=select_k)
        return result, target
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted scaler and selected features."""
        result = self.create_all_features(df)
        result = self.normalize_features(result, fit=False)
        if self.selected_features:
            result = result[self.selected_features]
        return result


def create_target_variable(df: pd.DataFrame, method: str = 'classification', horizon: int = 1, threshold: float = 0.0) -> pd.Series:
    """Create target variable for supervised learning."""
    if method == 'classification':
        future_returns = df['close'].pct_change(horizon).shift(-horizon)
        if threshold == 0.0:
            target = (future_returns > 0).astype(int)
        else:
            target = pd.Series(1, index=df.index)
            target[future_returns > threshold] = 2
            target[future_returns < -threshold] = 0
    elif method == 'regression':
        target = df['close'].shift(-horizon)
    else:
        raise ValueError(f"Unknown method: {method}")
    return target
