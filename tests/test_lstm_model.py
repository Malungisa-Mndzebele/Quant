"""
Unit tests for LSTM price prediction model.

Tests model architecture, prediction output shape and range, and model save/load functionality.
Requirements: 2.1
"""

import pytest
import numpy as np
import pandas as pd
import os
import tempfile
import shutil
from ai.models.lstm_model import LSTMPricePredictor, create_default_lstm

try:
    import tensorflow as tf
except ImportError:
    pytest.skip("TensorFlow not installed", allow_module_level=True)


class TestLSTMModelArchitecture:
    """Tests for LSTM model architecture."""
    
    def test_model_initialization(self):
        """Test LSTM model initializes with correct parameters."""
        model = LSTMPricePredictor(
            sequence_length=60,
            n_features=5,
            lstm_units=[128, 64, 32],
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        assert model.sequence_length == 60
        assert model.n_features == 5
        assert model.lstm_units == [128, 64, 32]
        assert model.dropout_rate == 0.2
        assert model.learning_rate == 0.001
        assert model.model is None  # Not built yet
    
    def test_model_build(self):
        """Test model builds with correct architecture."""
        model = LSTMPricePredictor(
            sequence_length=60,
            n_features=5,
            lstm_units=[128, 64, 32],
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        model.build_model()
        
        # Model should be built
        assert model.model is not None
        
        # Check input shape
        assert model.model.input_shape == (None, 60, 5)
        
        # Check output shape
        assert model.model.output_shape == (None, 1)
        
        # Check number of layers (LSTM + Dropout + BatchNorm for each LSTM unit + Dense)
        # 3 LSTM layers * 3 (LSTM + Dropout + BatchNorm) + 1 Dense = 10 layers
        assert len(model.model.layers) == 10
    
    def test_model_build_single_lstm_layer(self):
        """Test model builds with single LSTM layer."""
        model = LSTMPricePredictor(
            sequence_length=30,
            n_features=3,
            lstm_units=[64],
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        model.build_model()
        
        # Model should be built
        assert model.model is not None
        
        # Check input shape
        assert model.model.input_shape == (None, 30, 3)
        
        # Check output shape
        assert model.model.output_shape == (None, 1)
    
    def test_model_summary(self):
        """Test model summary generation."""
        model = LSTMPricePredictor(
            sequence_length=60,
            n_features=5,
            lstm_units=[128, 64],
            dropout_rate=0.2,
            learning_rate=0.001
        )
        
        # Before building
        summary = model.get_model_summary()
        assert "Model not built yet" in summary
        
        # After building
        model.build_model()
        summary = model.get_model_summary()
        assert "lstm" in summary.lower()
        assert "dense" in summary.lower()
    
    def test_create_default_lstm(self):
        """Test default LSTM creation."""
        model = create_default_lstm()
        
        assert model.sequence_length == 60
        assert model.n_features == 5
        assert model.lstm_units == [128, 64, 32]
        assert model.dropout_rate == 0.2
        assert model.learning_rate == 0.001


class TestLSTMPredictionOutput:
    """Tests for LSTM prediction output shape and range."""
    
    def test_prepare_sequences(self):
        """Test sequence preparation for LSTM input."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        # Create sample data
        data = np.random.randn(50, 3)
        target = np.random.randn(50)
        
        X, y = model.prepare_sequences(data, target)
        
        # Check shapes
        assert X.shape == (40, 10, 3)  # 50 - 10 = 40 sequences
        assert y.shape == (40,)
        
        # Check that sequences are correct
        np.testing.assert_array_equal(X[0], data[0:10])
        np.testing.assert_array_equal(X[1], data[1:11])
        assert y[0] == target[10]
        assert y[1] == target[11]
    
    def test_prepare_sequences_without_target(self):
        """Test sequence preparation without target values."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        # Create sample data
        data = np.random.randn(50, 3)
        
        X, y = model.prepare_sequences(data, target=None)
        
        # Check shapes
        assert X.shape == (40, 10, 3)
        assert y is None
    
    def test_prepare_sequences_insufficient_data(self):
        """Test sequence preparation with insufficient data."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        # Create data shorter than sequence length
        data = np.random.randn(5, 3)
        target = np.random.randn(5)
        
        X, y = model.prepare_sequences(data, target)
        
        # Should return empty arrays
        assert X.shape == (0, 10, 3)
        assert y.shape == (0,)
    
    def test_prediction_output_shape(self):
        """Test prediction output has correct shape."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        model.build_model()
        
        # Create sample data
        data = np.random.randn(50, 3)
        
        predictions = model.predict(data)
        
        # Should have predictions for all sequences
        assert predictions.shape == (40,)  # 50 - 10 = 40
        assert predictions.ndim == 1
    
    def test_prediction_output_range(self):
        """Test prediction output is in reasonable range."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        model.build_model()
        
        # Create sample data with known range
        data = np.random.uniform(0, 100, size=(50, 3))
        
        predictions = model.predict(data)
        
        # Predictions should be finite
        assert np.all(np.isfinite(predictions))
        
        # For untrained model, predictions might be anywhere,
        # but should not be extreme values
        assert np.all(np.abs(predictions) < 1e6)
    
    def test_predict_without_training_raises_error(self):
        """Test prediction without training raises error."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        data = np.random.randn(50, 3)
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict(data)
    
    def test_predict_next_single_step(self):
        """Test predicting next single step."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        model.build_model()
        
        # Create recent data
        recent_data = np.random.randn(10, 3)
        
        predictions = model.predict_next(recent_data, steps=1)
        
        # Should return single prediction
        assert predictions.shape == (1,)
        assert np.isfinite(predictions[0])
    
    def test_predict_next_multiple_steps(self):
        """Test predicting multiple steps ahead."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        model.build_model()
        
        # Create recent data
        recent_data = np.random.randn(10, 3)
        
        predictions = model.predict_next(recent_data, steps=5)
        
        # Should return 5 predictions
        assert predictions.shape == (5,)
        assert np.all(np.isfinite(predictions))
    
    def test_predict_next_insufficient_data_raises_error(self):
        """Test predict_next with insufficient data raises error."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        model.build_model()
        
        # Create data shorter than sequence length
        recent_data = np.random.randn(5, 3)
        
        with pytest.raises(ValueError, match="Need at least"):
            model.predict_next(recent_data, steps=1)
    
    def test_predict_next_without_training_raises_error(self):
        """Test predict_next without training raises error."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        recent_data = np.random.randn(10, 3)
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.predict_next(recent_data, steps=1)


class TestLSTMTraining:
    """Tests for LSTM model training."""
    
    def test_train_basic(self):
        """Test basic model training."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        # Create sample training data
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        
        # Train for just 2 epochs to test functionality
        history = model.train(X_train, y_train, epochs=2, verbose=0)
        
        # Model should be built and trained
        assert model.model is not None
        assert model.history is not None
        
        # History should contain loss
        assert 'loss' in history
        assert len(history['loss']) <= 2  # May stop early
    
    def test_train_with_validation(self):
        """Test training with validation data."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        # Create sample data
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        X_val = np.random.randn(30, 3)
        y_val = np.random.randn(30)
        
        # Train for just 2 epochs
        history = model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=2,
            verbose=0
        )
        
        # History should contain validation loss
        assert 'val_loss' in history
        assert len(history['val_loss']) <= 2
    
    def test_train_builds_model_if_not_built(self):
        """Test training builds model if not already built."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        # Model not built yet
        assert model.model is None
        
        # Create sample data
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        
        # Train
        model.train(X_train, y_train, epochs=1, verbose=0)
        
        # Model should now be built
        assert model.model is not None


class TestLSTMEvaluation:
    """Tests for LSTM model evaluation."""
    
    def test_evaluate_basic(self):
        """Test basic model evaluation."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        model.build_model()
        
        # Create sample test data
        X_test = np.random.randn(50, 3)
        y_test = np.random.randn(50)
        
        metrics = model.evaluate(X_test, y_test)
        
        # Should return all metrics
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
        assert 'direction_accuracy' in metrics
        
        # All metrics should be finite
        assert np.isfinite(metrics['mse'])
        assert np.isfinite(metrics['rmse'])
        assert np.isfinite(metrics['mae'])
        assert np.isfinite(metrics['mape'])
        assert np.isfinite(metrics['direction_accuracy'])
        
        # RMSE should be sqrt of MSE
        assert np.isclose(metrics['rmse'], np.sqrt(metrics['mse']))
    
    def test_evaluate_without_training_raises_error(self):
        """Test evaluation without training raises error."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        X_test = np.random.randn(50, 3)
        y_test = np.random.randn(50)
        
        with pytest.raises(ValueError, match="Model not trained"):
            model.evaluate(X_test, y_test)
    
    def test_evaluate_direction_accuracy(self):
        """Test direction accuracy calculation."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        model.build_model()
        
        # Create test data with clear trend
        X_test = np.random.randn(50, 3)
        y_test = np.arange(50, dtype=float)  # Clear uptrend
        
        metrics = model.evaluate(X_test, y_test)
        
        # Direction accuracy should be between 0 and 1
        assert 0 <= metrics['direction_accuracy'] <= 1


class TestLSTMSaveLoad:
    """Tests for LSTM model save/load functionality."""
    
    def test_save_model(self):
        """Test saving model to disk."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        model.build_model()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model")
            
            # Save model
            model.save(filepath)
            
            # Check files exist
            assert os.path.exists(f"{filepath}.h5")
            assert os.path.exists(f"{filepath}_config.json")
    
    def test_save_without_model_raises_error(self):
        """Test saving without building model raises error."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model")
            
            with pytest.raises(ValueError, match="No model to save"):
                model.save(filepath)
    
    def test_load_model(self):
        """Test loading model from disk."""
        # Create and save model
        model1 = LSTMPricePredictor(
            sequence_length=15,
            n_features=4,
            lstm_units=[64, 32],
            dropout_rate=0.3,
            learning_rate=0.002
        )
        model1.build_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model")
            model1.save(filepath)
            
            # Load model
            model2 = LSTMPricePredictor(sequence_length=10, n_features=3)
            model2.load(filepath)
            
            # Check configuration was loaded
            assert model2.sequence_length == 15
            assert model2.n_features == 4
            assert model2.lstm_units == [64, 32]
            assert model2.dropout_rate == 0.3
            assert model2.learning_rate == 0.002
            
            # Check model was loaded
            assert model2.model is not None
    
    def test_save_load_preserves_predictions(self):
        """Test that save/load preserves model predictions."""
        # Create and train model
        model1 = LSTMPricePredictor(sequence_length=10, n_features=3)
        
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)
        model1.train(X_train, y_train, epochs=2, verbose=0)
        
        # Make predictions
        X_test = np.random.randn(50, 3)
        predictions1 = model1.predict(X_test)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "test_model")
            
            # Save model
            model1.save(filepath)
            
            # Load model
            model2 = LSTMPricePredictor(sequence_length=10, n_features=3)
            model2.load(filepath)
            
            # Make predictions with loaded model
            predictions2 = model2.predict(X_test)
            
            # Predictions should be identical
            np.testing.assert_allclose(predictions1, predictions2, rtol=1e-5)
    
    def test_save_creates_directory(self):
        """Test save creates directory if it doesn't exist."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        model.build_model()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested path that doesn't exist
            filepath = os.path.join(tmpdir, "subdir", "test_model")
            
            # Save should create directory
            model.save(filepath)
            
            # Check files exist
            assert os.path.exists(f"{filepath}.h5")
            assert os.path.exists(f"{filepath}_config.json")


class TestLSTMEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_zero_sequence_length(self):
        """Test model with zero sequence length."""
        # This should work but may not be practical
        model = LSTMPricePredictor(sequence_length=0, n_features=3)
        
        # Building might fail or create unusual architecture
        # Just test that it doesn't crash catastrophically
        try:
            model.build_model()
        except Exception:
            pass  # Expected to potentially fail
    
    def test_zero_features(self):
        """Test model with zero features."""
        model = LSTMPricePredictor(sequence_length=10, n_features=0)
        
        # Building should fail or create unusual architecture
        try:
            model.build_model()
        except Exception:
            pass  # Expected to potentially fail
    
    def test_empty_lstm_units(self):
        """Test model with empty LSTM units list."""
        model = LSTMPricePredictor(
            sequence_length=10,
            n_features=3,
            lstm_units=[]
        )
        
        # Building should fail
        with pytest.raises(Exception):
            model.build_model()
    
    def test_negative_dropout_rate(self):
        """Test model with negative dropout rate."""
        # Keras should handle this, but test behavior
        model = LSTMPricePredictor(
            sequence_length=10,
            n_features=3,
            dropout_rate=-0.1
        )
        
        # Building might fail or clamp to 0
        try:
            model.build_model()
        except Exception:
            pass  # Expected to potentially fail
    
    def test_dropout_rate_greater_than_one(self):
        """Test model with dropout rate > 1."""
        # Keras should handle this, but test behavior
        model = LSTMPricePredictor(
            sequence_length=10,
            n_features=3,
            dropout_rate=1.5
        )
        
        # Building might fail or clamp to 1
        try:
            model.build_model()
        except Exception:
            pass  # Expected to potentially fail
    
    def test_very_large_sequence_length(self):
        """Test model with very large sequence length."""
        model = LSTMPricePredictor(sequence_length=10000, n_features=3)
        
        # Should build but may be slow
        model.build_model()
        assert model.model is not None
    
    def test_predict_with_wrong_feature_count(self):
        """Test prediction with wrong number of features."""
        model = LSTMPricePredictor(sequence_length=10, n_features=3)
        model.build_model()
        
        # Create data with wrong number of features
        data = np.random.randn(50, 5)  # 5 features instead of 3
        
        # Should raise error
        with pytest.raises(Exception):
            model.predict(data)
