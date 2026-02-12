"""
LSTM price prediction model for stock price forecasting.

This module implements a Long Short-Term Memory (LSTM) neural network
for predicting future stock prices based on historical price data and
technical indicators.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import os
import json
from datetime import datetime

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
except ImportError:
    raise ImportError(
        "TensorFlow is required for LSTM model. "
        "Install with: pip install tensorflow"
    )


class LSTMPricePredictor:
    """
    LSTM-based stock price prediction model.
    
    This model uses historical price sequences to predict future prices.
    It supports training, prediction, and model persistence.
    """
    
    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 5,
        lstm_units: list = [128, 64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM model.
        
        Args:
            sequence_length: Number of time steps to look back
            n_features: Number of input features per time step
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Learning rate for optimizer
        """
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.scaler = None
        self.history = None
        
    def build_model(self) -> None:
        """
        Build the LSTM model architecture.
        
        Creates a sequential model with multiple LSTM layers,
        dropout for regularization, and a dense output layer.
        """
        self.model = Sequential()
        
        # First LSTM layer with return sequences
        self.model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True if len(self.lstm_units) > 1 else False,
            input_shape=(self.sequence_length, self.n_features)
        ))
        self.model.add(Dropout(self.dropout_rate))
        self.model.add(BatchNormalization())
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:], 1):
            return_seq = i < len(self.lstm_units) - 1
            self.model.add(LSTM(units=units, return_sequences=return_seq))
            self.model.add(Dropout(self.dropout_rate))
            self.model.add(BatchNormalization())
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
    def prepare_sequences(
        self,
        data: np.ndarray,
        target: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Prepare sequences for LSTM input.
        
        Args:
            data: Input data array of shape (n_samples, n_features)
            target: Target values (optional, for training)
            
        Returns:
            Tuple of (X, y) where X is sequences and y is targets
        """
        X = []
        y = [] if target is not None else None
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i - self.sequence_length:i])
            if target is not None:
                y.append(target[i])
        
        X = np.array(X)
        y = np.array(y) if y is not None else None
        
        return X, y
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the LSTM model.
        
        Args:
            X_train: Training sequences
            y_train: Training targets
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Training history dictionary
        """
        if self.model is None:
            self.build_model()
        
        # Prepare sequences
        X_train_seq, y_train_seq = self.prepare_sequences(X_train, y_train)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_seq, y_val_seq = self.prepare_sequences(X_val, y_val)
            validation_data = (X_val_seq, y_val_seq)
        
        # Callbacks
        callback_list = [
            EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=verbose
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=verbose
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        return self.history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Input data array
            
        Returns:
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_seq, _ = self.prepare_sequences(X)
        predictions = self.model.predict(X_seq, verbose=0)
        
        return predictions.flatten()
    
    def predict_next(self, recent_data: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        Predict next N steps into the future.
        
        Args:
            recent_data: Recent data of shape (sequence_length, n_features)
            steps: Number of steps to predict ahead
            
        Returns:
            Array of predicted values
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        if len(recent_data) < self.sequence_length:
            raise ValueError(
                f"Need at least {self.sequence_length} data points, "
                f"got {len(recent_data)}"
            )
        
        predictions = []
        current_sequence = recent_data[-self.sequence_length:].copy()
        
        for _ in range(steps):
            # Reshape for prediction
            X = current_sequence.reshape(1, self.sequence_length, self.n_features)
            
            # Predict next value
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence (shift and add prediction)
            # For simplicity, we only update the first feature (price)
            # Other features would need to be calculated or estimated
            new_row = current_sequence[-1].copy()
            new_row[0] = pred
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        return np.array(predictions)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test sequences
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        X_test_seq, y_test_seq = self.prepare_sequences(X_test, y_test)
        
        # Get predictions
        predictions = self.model.predict(X_test_seq, verbose=0).flatten()
        
        # Calculate metrics
        mse = np.mean((predictions - y_test_seq) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - y_test_seq))
        mape = np.mean(np.abs((y_test_seq - predictions) / y_test_seq)) * 100
        
        # Direction accuracy (did we predict up/down correctly?)
        if len(y_test_seq) > 1:
            actual_direction = np.diff(y_test_seq) > 0
            pred_direction = np.diff(predictions) > 0
            direction_accuracy = np.mean(actual_direction == pred_direction)
        else:
            direction_accuracy = 0.0
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }
    
    def save(self, filepath: str) -> None:
        """
        Save model to disk.
        
        Args:
            filepath: Path to save model (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Save model
        self.model.save(f"{filepath}.h5")
        
        # Save configuration
        config = {
            'sequence_length': self.sequence_length,
            'n_features': self.n_features,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """
        Load model from disk.
        
        Args:
            filepath: Path to load model from (without extension)
        """
        # Load configuration
        with open(f"{filepath}_config.json", 'r') as f:
            config = json.load(f)
        
        self.sequence_length = config['sequence_length']
        self.n_features = config['n_features']
        self.lstm_units = config['lstm_units']
        self.dropout_rate = config['dropout_rate']
        self.learning_rate = config['learning_rate']
        
        # Load model
        self.model = load_model(f"{filepath}.h5")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not built yet"
        
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


def create_default_lstm() -> LSTMPricePredictor:
    """
    Create LSTM model with default configuration.
    
    Returns:
        Configured LSTMPricePredictor instance
    """
    return LSTMPricePredictor(
        sequence_length=60,
        n_features=5,
        lstm_units=[128, 64, 32],
        dropout_rate=0.2,
        learning_rate=0.001
    )
