"""
Random Forest classifier for trading signal generation.

This module implements a Random Forest classifier for predicting
buy/sell/hold trading signals based on technical indicators and
market features.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple, Union
import os
import json
import pickle
from datetime import datetime

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    raise ImportError(
        "scikit-learn is required for Random Forest model. "
        "Install with: pip install scikit-learn"
    )


class RandomForestTrader:
    """
    Random Forest classifier for trading signal prediction.
    
    This model classifies market conditions into buy, sell, or hold signals
    based on technical indicators and price features. It supports training
    with cross-validation, feature importance analysis, and probability-based
    predictions.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = 20,
        min_samples_split: int = 10,
        min_samples_leaf: int = 5,
        max_features: str = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Random Forest classifier.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at leaf node
            max_features: Number of features to consider for best split
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 for all cores)
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.feature_importances_ = None
        self.cv_scores = None
        self.classes_ = ['hold', 'buy', 'sell']  # Default classes
        
    def build_model(self) -> None:
        """
        Build the Random Forest classifier.
        
        Creates a RandomForestClassifier with configured parameters.
        """
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            class_weight='balanced',  # Handle imbalanced classes
            bootstrap=True,
            oob_score=True  # Out-of-bag score for validation
        )
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        feature_names: Optional[List[str]] = None,
        cv_folds: int = 5,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Train the Random Forest model with cross-validation.
        
        Args:
            X_train: Training features of shape (n_samples, n_features)
            y_train: Training labels (buy/sell/hold)
            feature_names: Names of features for interpretability
            cv_folds: Number of cross-validation folds
            verbose: Verbosity level (0, 1, or 2)
            
        Returns:
            Training results dictionary with metrics and CV scores
        """
        if self.model is None:
            self.build_model()
        
        # Store feature names
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        # Encode labels if they are strings
        if y_train.dtype == object or isinstance(y_train[0], str):
            y_train_encoded = self.label_encoder.fit_transform(y_train)
            self.classes_ = self.label_encoder.classes_.tolist()
        else:
            y_train_encoded = y_train
            self.label_encoder.fit(y_train)
            self.classes_ = self.label_encoder.classes_.tolist()
        
        # Cross-validation
        if verbose > 0:
            print(f"Performing {cv_folds}-fold cross-validation...")
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        self.cv_scores = cross_val_score(
            self.model,
            X_train,
            y_train_encoded,
            cv=cv,
            scoring='accuracy',
            n_jobs=self.n_jobs
        )
        
        if verbose > 0:
            print(f"Cross-validation scores: {self.cv_scores}")
            print(f"Mean CV accuracy: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std() * 2:.4f})")
        
        # Train final model on all data
        if verbose > 0:
            print("Training final model on all training data...")
        
        self.model.fit(X_train, y_train_encoded)
        
        # Extract feature importances
        self.feature_importances_ = self.model.feature_importances_
        
        # Calculate training metrics
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train_encoded, y_train_pred)
        
        # OOB score (out-of-bag)
        oob_score = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else None
        
        results = {
            'train_accuracy': float(train_accuracy),
            'cv_mean': float(self.cv_scores.mean()),
            'cv_std': float(self.cv_scores.std()),
            'cv_scores': self.cv_scores.tolist(),
            'oob_score': float(oob_score) if oob_score is not None else None,
            'n_features': X_train.shape[1],
            'n_samples': X_train.shape[0],
            'classes': self.classes_
        }
        
        if verbose > 0:
            print(f"\nTraining Results:")
            print(f"  Training Accuracy: {train_accuracy:.4f}")
            if oob_score is not None:
                print(f"  OOB Score: {oob_score:.4f}")
            print(f"  Classes: {self.classes_}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict trading signals for input features.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted class labels (buy/sell/hold)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions_encoded = self.model.predict(X)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for input features.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Probability scores of shape (n_samples, n_classes)
            Order corresponds to self.classes_
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        probabilities = self.model.predict_proba(X)
        return probabilities
    
    def predict_with_confidence(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict with confidence scores for each prediction.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            List of dictionaries containing prediction and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        results = []
        for i, pred in enumerate(predictions):
            probs_dict = {
                cls: float(prob) 
                for cls, prob in zip(self.classes_, probabilities[i])
            }
            
            results.append({
                'prediction': pred,
                'confidence': float(probabilities[i].max()),
                'probabilities': probs_dict
            })
        
        return results
    
    def get_feature_importance(
        self,
        top_n: Optional[int] = None,
        as_dataframe: bool = True
    ) -> Union[pd.DataFrame, Dict[str, float]]:
        """
        Get feature importance scores.
        
        Args:
            top_n: Return only top N features (None for all)
            as_dataframe: Return as DataFrame if True, dict if False
            
        Returns:
            Feature importances as DataFrame or dictionary
        """
        if self.model is None or self.feature_importances_ is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create importance dictionary
        importance_dict = {
            name: float(importance)
            for name, importance in zip(self.feature_names, self.feature_importances_)
        }
        
        # Sort by importance
        sorted_importance = sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Limit to top N if specified
        if top_n is not None:
            sorted_importance = sorted_importance[:top_n]
        
        if as_dataframe:
            df = pd.DataFrame(sorted_importance, columns=['feature', 'importance'])
            return df
        else:
            return dict(sorted_importance)
    
    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        verbose: int = 1
    ) -> Dict[str, Any]:
        """
        Evaluate model performance on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            verbose: Verbosity level
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Encode labels if needed
        if y_test.dtype == object or isinstance(y_test[0], str):
            y_test_encoded = self.label_encoder.transform(y_test)
        else:
            y_test_encoded = y_test
        
        # Get predictions
        y_pred_encoded = self.model.predict(X_test)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
        conf_matrix = confusion_matrix(y_test_encoded, y_pred_encoded)
        class_report = classification_report(
            y_test_encoded,
            y_pred_encoded,
            target_names=self.classes_,
            output_dict=True
        )
        
        results = {
            'accuracy': float(accuracy),
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report,
            'n_samples': len(y_test)
        }
        
        if verbose > 0:
            print(f"\nTest Results:")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"\nConfusion Matrix:")
            print(conf_matrix)
            print(f"\nClassification Report:")
            print(classification_report(
                y_test_encoded,
                y_pred_encoded,
                target_names=self.classes_
            ))
        
        return results
    
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
        
        # Save model using pickle
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save label encoder
        with open(f"{filepath}_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save configuration and metadata
        config = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'feature_names': self.feature_names,
            'classes': self.classes_,
            'feature_importances': self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
            'cv_scores': self.cv_scores.tolist() if self.cv_scores is not None else None,
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
        
        self.n_estimators = config['n_estimators']
        self.max_depth = config['max_depth']
        self.min_samples_split = config['min_samples_split']
        self.min_samples_leaf = config['min_samples_leaf']
        self.max_features = config['max_features']
        self.random_state = config['random_state']
        self.feature_names = config['feature_names']
        self.classes_ = config['classes']
        self.feature_importances_ = np.array(config['feature_importances']) if config['feature_importances'] else None
        self.cv_scores = np.array(config['cv_scores']) if config['cv_scores'] else None
        
        # Load model
        with open(f"{filepath}.pkl", 'rb') as f:
            self.model = pickle.load(f)
        
        # Load label encoder
        with open(f"{filepath}_encoder.pkl", 'rb') as f:
            self.label_encoder = pickle.load(f)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information and statistics.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"status": "Model not trained"}
        
        info = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'classes': self.classes_,
            'oob_score': float(self.model.oob_score_) if hasattr(self.model, 'oob_score_') else None,
            'cv_mean': float(self.cv_scores.mean()) if self.cv_scores is not None else None,
            'cv_std': float(self.cv_scores.std()) if self.cv_scores is not None else None
        }
        
        return info


def create_default_rf() -> RandomForestTrader:
    """
    Create Random Forest model with default configuration.
    
    Returns:
        Configured RandomForestTrader instance
    """
    return RandomForestTrader(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
