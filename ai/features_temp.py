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
