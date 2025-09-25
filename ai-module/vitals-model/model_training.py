"""
Vitals Model Training Module
Train XGBoost model on vitals data to predict heart rate variability (Hv).
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class VitalsModelTrainer:
    """Train and evaluate vitals models for Hv prediction."""
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize model trainer.
        
        Args:
            model_type: Type of model to train ('xgboost', 'linear', 'random_forest')
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
            y: Target values (Hv scores)
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        if X.size == 0:
            raise ValueError("No training data provided")
        
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape[0]} samples, {X_train_scaled.shape[1]} features")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        print(f"Target range - Train: [{y_train.min():.3f}, {y_train.max():.3f}]")
        print(f"Target range - Test: [{y_test.min():.3f}, {y_test.max():.3f}]")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray = None, y_val: np.ndarray = None) -> xgb.XGBRegressor:
        """
        Train XGBoost regression model with hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Trained XGBoost model
        """
        print("Training XGBoost regression model...")
        
        # Default parameters for regression
        default_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse'
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
        base_model = xgb.XGBRegressor(**default_params)
        
        # Perform grid search if validation data is available
        if X_val is not None and y_val is not None:
            print("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                base_model, param_grid, cv=3, scoring='neg_mean_squared_error', 
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best CV score: {-grid_search.best_score_:.3f}")
            
        else:
            # Use default parameters
            best_model = base_model
            best_model.fit(X_train, y_train)
        
        return best_model
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train linear regression model."""
        from sklearn.linear_model import LinearRegression
        
        print("Training Linear Regression model...")
        model = LinearRegression()
        model.fit(X_train, y_train)
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train random forest regression model."""
        from sklearn.ensemble import RandomForestRegressor
        
        print("Training Random Forest regression model...")
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def train_model(self, X: np.ndarray, y: np.ndarray, 
                   feature_names: List[str] = None) -> Dict:
        """
        Train the specified model type.
        
        Args:
            X: Feature matrix
            y: Target values
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
        elif self.model_type == 'linear':
            self.model = self.train_linear_regression(X_train, y_train)
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
            y_test: Test targets
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate regression metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
        
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape,
            'predictions': y_pred.tolist(),
            'actual': y_test.tolist()
        }
        
        # Print results
        print(f"\nModel Evaluation Results:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        
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
            importances = np.abs(self.model.coef_)
        else:
            return {}
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def plot_feature_importance(self, top_n: int = 15, save_path: str = None):
        """Plot feature importance."""
        importance_dict = self.get_feature_importance()
        
        if not importance_dict:
            print("No feature importance available for this model type")
            return
        
        # Get top N features
        top_features = dict(list(importance_dict.items())[:top_n])
        
        # Create plot
        plt.figure(figsize=(12, 8))
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {top_n} Feature Importances for Hv Prediction')
        plt.gca().invert_yaxis()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_predictions(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None):
        """Plot actual vs predicted values."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X_test)
        
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Hv')
        plt.ylabel('Predicted Hv')
        plt.title('Actual vs Predicted Hv Values')
        plt.grid(True, alpha=0.3)
        
        # Add R² score to plot
        r2 = r2_score(y_test, y_pred)
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Predictions plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, X_test: np.ndarray, y_test: np.ndarray, save_path: str = None):
        """Plot residuals for model diagnostics."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        y_pred = self.model.predict(X_test)
        residuals = y_test - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Hv')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Residuals plot saved to {save_path}")
        
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
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted Hv values
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        return predictions


def main():
    """Demo the vitals model training."""
    # Create sample training data similar to your dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic vitals features (similar to your dataset.py)
    age = np.random.randint(18, 90, size=n_samples)
    hr_median = np.random.normal(loc=75, scale=12, size=n_samples)
    spo2_median = np.clip(np.random.normal(loc=97, scale=2, size=n_samples), 80, 100)
    diabetes = np.random.binomial(1, 0.15, size=n_samples)
    hypertension = np.random.binomial(1, 0.25, size=n_samples)
    heart_disease = np.random.binomial(1, 0.10, size=n_samples)
    respiratory_condition = np.random.binomial(1, 0.08, size=n_samples)
    smoker = np.random.binomial(1, 0.20, size=n_samples)
    
    # Stack features
    X = np.column_stack([age, hr_median, spo2_median, diabetes, hypertension, 
                        heart_disease, respiratory_condition, smoker])
    
    # Create synthetic Hv target (heart rate variability score)
    Hv = (
        0.3 * (np.abs(hr_median - 75) / 75) +  # deviation from normal HR
        0.4 * ((100 - spo2_median) / 20) +     # penalty for low SpO2
        0.1 * diabetes +
        0.1 * hypertension +
        0.15 * heart_disease +
        0.1 * respiratory_condition +
        0.05 * smoker +
        np.random.normal(0, 0.05, n_samples)  # Add some noise
    )
    
    # Clip to [0,1]
    Hv = np.clip(Hv, 0, 1)
    
    # Feature names
    feature_names = ['age', 'hr_median', 'spo2_median', 'diabetes', 'hypertension', 
                    'heart_disease', 'respiratory_condition', 'smoker']
    
    # Train model
    trainer = VitalsModelTrainer(model_type='xgboost')
    results = trainer.train_model(X, Hv, feature_names)
    
    # Show feature importance
    importance = trainer.get_feature_importance()
    print(f"\nTop 5 most important features for Hv prediction:")
    for i, (feature, imp) in enumerate(list(importance.items())[:5]):
        print(f"  {i+1}. {feature}: {imp:.3f}")
    
    print("Vitals model training module ready!")


if __name__ == "__main__":
    main()
