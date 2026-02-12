"""
Model training pipeline for AI trading agent.

This module provides functionality for training machine learning models
including data preparation, training loops, evaluation, and model persistence.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Import models
from ai.models.lstm_model import LSTMPricePredictor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Handles training of ML models for price prediction.
    """
    
    def __init__(self, model_dir: str = "data/models"):
        """
        Initialize model trainer.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scaler = MinMaxScaler()
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_col: str = 'close',
        feature_cols: Optional[List[str]] = None,
        test_size: float = 0.2,
        val_size: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with OHLCV data and features
            target_col: Column name for target variable
            feature_cols: List of feature column names (if None, uses all numeric columns)
            test_size: Proportion of data for testing
            val_size: Proportion of training data for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Select features
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col in feature_cols and target_col != feature_cols[0]:
                # Move target to first position
                feature_cols.remove(target_col)
                feature_cols.insert(0, target_col)
        
        # Extract data
        data = df[feature_cols].values
        target = df[target_col].values
        
        # Handle missing values
        if np.any(np.isnan(data)):
            logger.warning("Found NaN values in data, forward filling...")
            df_clean = df[feature_cols].fillna(method='ffill').fillna(method='bfill')
            data = df_clean.values
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        
        # Split data (time series, so no shuffling)
        train_size = int(len(data_scaled) * (1 - test_size))
        
        X_train_full = data_scaled[:train_size]
        y_train_full = target[:train_size]
        
        X_test = data_scaled[train_size:]
        y_test = target[train_size:]
        
        # Further split training into train and validation
        val_split = int(len(X_train_full) * (1 - val_size))
        
        X_train = X_train_full[:val_split]
        y_train = y_train_full[:val_split]
        
        X_val = X_train_full[val_split:]
        y_val = y_train_full[val_split:]
        
        logger.info(f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_lstm(
        self,
        df: pd.DataFrame,
        symbol: str,
        sequence_length: int = 60,
        epochs: int = 100,
        batch_size: int = 32,
        feature_cols: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train LSTM model for price prediction.
        
        Args:
            df: DataFrame with historical price data
            symbol: Stock symbol
            sequence_length: Number of time steps to look back
            epochs: Number of training epochs
            batch_size: Batch size for training
            feature_cols: List of feature columns to use
            **kwargs: Additional arguments for LSTM model
            
        Returns:
            Dictionary with training results and metrics
        """
        logger.info(f"Training LSTM model for {symbol}...")
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(
            df, feature_cols=feature_cols
        )
        
        # Create model
        n_features = X_train.shape[1]
        model = LSTMPricePredictor(
            sequence_length=sequence_length,
            n_features=n_features,
            **kwargs
        )
        
        # Train model
        logger.info("Starting training...")
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        
        # Evaluate on test set
        logger.info("Evaluating model...")
        test_metrics = model.evaluate(X_test, y_test)
        
        # Save model
        model_path = os.path.join(self.model_dir, f"lstm_{symbol}")
        model.save(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Prepare results
        results = {
            'symbol': symbol,
            'model_type': 'LSTM',
            'sequence_length': sequence_length,
            'n_features': n_features,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'training_history': history,
            'test_metrics': test_metrics,
            'model_path': model_path,
            'trained_at': datetime.now().isoformat()
        }
        
        # Log results
        logger.info(f"Training complete!")
        logger.info(f"Test RMSE: {test_metrics['rmse']:.4f}")
        logger.info(f"Test MAE: {test_metrics['mae']:.4f}")
        logger.info(f"Test MAPE: {test_metrics['mape']:.2f}%")
        logger.info(f"Direction Accuracy: {test_metrics['direction_accuracy']:.2%}")
        
        return results
    
    def train_multiple_symbols(
        self,
        data_dict: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train models for multiple symbols.
        
        Args:
            data_dict: Dictionary mapping symbols to DataFrames
            **kwargs: Arguments passed to train_lstm
            
        Returns:
            Dictionary mapping symbols to training results
        """
        results = {}
        
        for symbol, df in data_dict.items():
            try:
                result = self.train_lstm(df, symbol, **kwargs)
                results[symbol] = result
            except Exception as e:
                logger.error(f"Failed to train model for {symbol}: {e}")
                results[symbol] = {'error': str(e)}
        
        return results
    
    def load_model(self, symbol: str, model_type: str = 'lstm') -> Any:
        """
        Load a trained model from disk.
        
        Args:
            symbol: Stock symbol
            model_type: Type of model ('lstm', 'rf', etc.)
            
        Returns:
            Loaded model instance
        """
        model_path = os.path.join(self.model_dir, f"{model_type}_{symbol}")
        
        if model_type == 'lstm':
            model = LSTMPricePredictor()
            model.load(model_path)
            return model
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def get_training_progress(self, history: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Convert training history to DataFrame for analysis.
        
        Args:
            history: Training history dictionary
            
        Returns:
            DataFrame with training metrics per epoch
        """
        return pd.DataFrame(history)


def train_model_from_csv(
    csv_path: str,
    symbol: str,
    model_dir: str = "data/models",
    **kwargs
) -> Dict[str, Any]:
    """
    Train model from CSV file.
    
    Args:
        csv_path: Path to CSV file with historical data
        symbol: Stock symbol
        model_dir: Directory to save model
        **kwargs: Additional training arguments
        
    Returns:
        Training results dictionary
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Ensure datetime index
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.set_index('timestamp')
    elif 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')
    
    # Train model
    trainer = ModelTrainer(model_dir=model_dir)
    results = trainer.train_lstm(df, symbol, **kwargs)
    
    return results


def train_model_from_api(
    symbol: str,
    start_date: str,
    end_date: str,
    api_key: str,
    api_secret: str,
    model_dir: str = "data/models",
    **kwargs
) -> Dict[str, Any]:
    """
    Train model using data from Alpaca API.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        api_key: Alpaca API key
        api_secret: Alpaca API secret
        model_dir: Directory to save model
        **kwargs: Additional training arguments
        
    Returns:
        Training results dictionary
    """
    from services.market_data_service import MarketDataService
    
    # Fetch data
    logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}...")
    market_data = MarketDataService(api_key, api_secret)
    df = market_data.get_bars(
        symbol,
        timeframe='1Day',
        start=pd.to_datetime(start_date),
        end=pd.to_datetime(end_date)
    )
    
    if df.empty:
        raise ValueError(f"No data retrieved for {symbol}")
    
    logger.info(f"Retrieved {len(df)} bars")
    
    # Train model
    trainer = ModelTrainer(model_dir=model_dir)
    results = trainer.train_lstm(df, symbol, **kwargs)
    
    return results


if __name__ == "__main__":
    """
    Example usage for training models.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Train LSTM model for stock prediction')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--csv', type=str, help='Path to CSV file')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--api-key', type=str, help='Alpaca API key')
    parser.add_argument('--api-secret', type=str, help='Alpaca API secret')
    parser.add_argument('--model-dir', type=str, default='data/models', help='Model directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length')
    
    args = parser.parse_args()
    
    if args.csv:
        # Train from CSV
        results = train_model_from_csv(
            args.csv,
            args.symbol,
            model_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length
        )
    elif args.api_key and args.api_secret and args.start_date and args.end_date:
        # Train from API
        results = train_model_from_api(
            args.symbol,
            args.start_date,
            args.end_date,
            args.api_key,
            args.api_secret,
            model_dir=args.model_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length
        )
    else:
        parser.error("Either --csv or (--api-key, --api-secret, --start-date, --end-date) required")
    
    print("\nTraining Results:")
    print(f"Symbol: {results['symbol']}")
    print(f"Test RMSE: {results['test_metrics']['rmse']:.4f}")
    print(f"Test MAE: {results['test_metrics']['mae']:.4f}")
    print(f"Test MAPE: {results['test_metrics']['mape']:.2f}%")
    print(f"Direction Accuracy: {results['test_metrics']['direction_accuracy']:.2%}")
    print(f"Model saved to: {results['model_path']}")
