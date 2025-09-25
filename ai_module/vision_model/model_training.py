"""
Step 5: Model Training
Train XGBoost model on extracted features for drowsiness detection.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path


class DrowsinessModelTrainer:
    """Train and evaluate drowsiness detection models."""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model to train ('xgboost', 'logistic', 'random_forest')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_history = {}
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for training with train/test split and scaling.
        
        Args:
            X: Feature matrix
            y: Target labels
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if X.size == 0:
            raise ValueError("No training data provided")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
        print(f"Class distribution - Test: {np.bincount(y_test.astype(int))}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> xgb.XGBClassifier:
        """
        Train XGBoost model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            
        Returns:
            Trained XGBoost model
        """
        print("Training XGBoost model...")
        
        # Default parameters
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        # Hyperparameter grid for tuning
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.05, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Create base model
        base_model = xgb.XGBClassifier(**default_params)
        
        # Perform grid search if validation data is available
        if X_val is not None and y_val is not None:
            print("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, scoring='roc_auc', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {grid_search.best_score_:.3f}")
            
        else:
            # Use default parameters
            best_model = base_model
            best_model.fit(X_train, y_train)
        
        return best_model
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train logistic regression model."""
        from sklearn.linear_model import LogisticRegression
        
        print("Training Logistic Regression model...")
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train random forest model."""
        from sklearn.ensemble import RandomForestClassifier
        
        print("Training Random Forest model...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   feature_names: List[str] = None) -> Dict:
        """
        Train the specified model type.
        
        Args:
            X: Feature matrix
            y: Target labels
            feature_names: List of feature names
            
        Returns:
            Dictionary with training results
        """
        self.feature_names = feature_names
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        # Train model based on type
        if self.model_type == 'xgboost':
            self.model = self.train_xgboost(X_train, y_train)
        elif self.model_type == 'logistic':
            self.model = self.train_logistic_regression(X_train, y_train)
        elif self.model_type == 'random_forest':
            self.model = self.train_random_forest(X_train, y_train)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Evaluate model
        results = self.evaluate_model(X_test, y_test)
        
        # Store training history
        self.training_history = {
            'model_type': self.model_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'feature_count': X_train.shape[1],
            'results': results
        }
        
        return results
    
    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the trained model.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = np.mean(y_pred == y_test)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist()
        }
        
        # Print results
        print(f"\nModel Evaluation Results:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"AUC Score: {auc_score:.3f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.model.n_features_in_)]
        else:
            feature_names = self.feature_names
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            return {}
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def plot_feature_importance(self, top_n: int = 20, save_path: str = None):
        """Plot feature importance."""
        importance_dict = self.get_feature_importance()
        
        if not importance_dict:
            print("No feature importance available for this model type")
            return
        
        # Get top N features
        top_features = dict(list(importance_dict.items())[:top_n])
        
        # Create plot
        plt.figure(figsize=(10, 8))
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None):
        """Plot ROC curve."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"ROC curve saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str, scaler_path: str = None):
        """Save trained model and scaler."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Save model
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")
        
        # Save scaler
        if scaler_path:
            joblib.dump(self.scaler, scaler_path)
            print(f"Scaler saved to {scaler_path}")
        
        # Save training history
        history_path = model_path.replace('.pkl', '_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"Training history saved to {history_path}")
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """Load trained model and scaler."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
        
        if scaler_path:
            self.scaler = joblib.load(scaler_path)
            print(f"Scaler loaded from {scaler_path}")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        return predictions, probabilities


def main():
    """Demo the model training."""
    # Create sample training data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)
    
    # Create synthetic labels based on some features
    # Simulate drowsiness based on low EAR, high MAR, etc.
    drowsiness_score = (
        -X[:, 0] * 0.5 +  # Low EAR (negative correlation)
        X[:, 1] * 0.3 +   # High MAR (positive correlation)
        -X[:, 2] * 0.2 +  # Low blink frequency
        np.random.randn(n_samples) * 0.1  # Noise
    )
    
    y = (drowsiness_score > np.median(drowsiness_score)).astype(int)
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(n_features)]
    
    # Train model
    trainer = DrowsinessModelTrainer(model_type='xgboost')
    results = trainer.train_model(X, y, feature_names)
    
    # Show feature importance
    importance = trainer.get_feature_importance()
    print(f"\nTop 5 most important features:")
    for i, (feature, imp) in enumerate(list(importance.items())[:5]):
        print(f"  {i+1}. {feature}: {imp:.3f}")
    
    print("Model training module ready!")


if __name__ == "__main__":
    main()
