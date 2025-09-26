"""
Vitals Model Inference Module
Real-time inference for heart rate variability (Hv) prediction using trained XGBoost model.
"""

import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Union, Optional
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class VitalsInferenceEngine:
    """Real-time inference engine for Hv prediction from vitals data."""
    
    def __init__(self, model_path: str = None, scaler_path: str = None):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model file
            scaler_path: Path to feature scaler file
        """
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_info = {}
        
        if model_path:
            self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path: str, scaler_path: str = None):
        """
        Load trained model and scaler.
        
        Args:
            model_path: Path to trained model file
            scaler_path: Path to feature scaler file
        """
        try:
            # Load model
            self.model = joblib.load(model_path)
            print(f"✓ Model loaded from {model_path}")
            
            # Load scaler
            if scaler_path and Path(scaler_path).exists():
                self.scaler = joblib.load(scaler_path)
                print(f"✓ Scaler loaded from {scaler_path}")
            else:
                print("⚠ No scaler provided - using raw features")
            
            # Load model history if available
            history_path = model_path.replace('.pkl', '_history.json')
            if Path(history_path).exists():
                with open(history_path, 'r') as f:
                    self.model_info = json.load(f)
                print(f"✓ Model info loaded from {history_path}")
            
            # Extract feature names if available
            if hasattr(self.model, 'feature_names_in_'):
                self.feature_names = list(self.model.feature_names_in_)
            elif 'feature_names' in self.model_info:
                self.feature_names = self.model_info['feature_names']
            
            print(f"Model ready for inference!")
            if self.feature_names:
                print(f"Expected features: {self.feature_names}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_input(self, vitals_data: Union[Dict, pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Preprocess input vitals data for inference.
        
        Args:
            vitals_data: Input data in various formats
            
        Returns:
            Preprocessed feature array
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Convert input to numpy array
        if isinstance(vitals_data, dict):
            # Single sample as dictionary
            if self.feature_names:
                # Use feature names to order data
                X = np.array([[vitals_data.get(name, 0) for name in self.feature_names]])
            else:
                # Use values in order
                X = np.array([list(vitals_data.values())])
                
        elif isinstance(vitals_data, pd.DataFrame):
            # DataFrame input
            if self.feature_names:
                # Ensure correct feature order
                missing_features = set(self.feature_names) - set(vitals_data.columns)
                if missing_features:
                    raise ValueError(f"Missing features: {missing_features}")
                X = vitals_data[self.feature_names].values
            else:
                X = vitals_data.values
                
        elif isinstance(vitals_data, (list, np.ndarray)):
            # Array-like input
            X = np.array(vitals_data)
            if X.ndim == 1:
                X = X.reshape(1, -1)
        else:
            raise ValueError(f"Unsupported input type: {type(vitals_data)}")
        
        # Validate feature count
        expected_features = self.model.n_features_in_
        if X.shape[1] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[1]}")
        
        # Apply scaling if available
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def predict_hv(self, vitals_data: Union[Dict, pd.DataFrame, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Predict heart rate variability (Hv) from vitals data.
        
        Args:
            vitals_data: Input vitals data
            
        Returns:
            Predicted Hv value(s)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        print(f"*************Input vitals_data: {vitals_data}")
        X = self.preprocess_input(vitals_data)
        print(f"*************Preprocessed X: {X}")
        
        # Make prediction
        prediction = self.model.predict(X)
        print(f"*************Raw Prediction: {prediction}")
        print(f"*************Prediction shape: {prediction.shape}")
        
        # Return single value for single sample, array for multiple samples
        if prediction.shape[0] == 1:
            result = float(prediction[0])
            print(f"*************Final HV Score: {result}")
            return result
        else:
            print(f"*************Multiple predictions: {prediction}")
            return prediction
    
    def predict_with_confidence(self, vitals_data: Union[Dict, pd.DataFrame, np.ndarray]) -> Dict:
        """
        Predict Hv with additional confidence metrics (if supported by model).
        
        Args:
            vitals_data: Input vitals data
            
        Returns:
            Dictionary with prediction and confidence metrics
        """
        # Get basic prediction
        hv_prediction = self.predict_hv(vitals_data)
        
        # Prepare result
        result = {
            'hv_prediction': hv_prediction,
            'confidence_available': False
        }
        
        # Add confidence intervals if model supports it (e.g., for tree-based models)
        try:
            X = self.preprocess_input(vitals_data)
            
            # For XGBoost, we can get prediction intervals using quantile regression
            # This is a simplified approach - in practice, you might want to train
            # separate models for confidence intervals
            if hasattr(self.model, 'predict'):
                # Use model's built-in prediction
                prediction = self.model.predict(X)
                
                # Estimate uncertainty based on training performance
                if 'results' in self.model_info and 'rmse' in self.model_info['results']:
                    rmse = self.model_info['results']['rmse']
                    
                    # Simple confidence interval based on RMSE
                    confidence_interval = 1.96 * rmse  # 95% confidence
                    
                    result.update({
                        'confidence_available': True,
                        'confidence_interval': confidence_interval,
                        'lower_bound': prediction - confidence_interval,
                        'upper_bound': prediction + confidence_interval,
                        'model_rmse': rmse
                    })
                    
        except Exception as e:
            print(f"Warning: Could not compute confidence metrics: {e}")
        
        return result
    
    def batch_predict(self, vitals_data_list: List[Union[Dict, np.ndarray]]) -> List[float]:
        """
        Predict Hv for multiple samples efficiently.
        
        Args:
            vitals_data_list: List of vitals data samples
            
        Returns:
            List of Hv predictions
        """
        if not vitals_data_list:
            return []
        
        # Convert all samples to consistent format
        X_batch = []
        for vitals_data in vitals_data_list:
            X_sample = self.preprocess_input(vitals_data)
            X_batch.append(X_sample[0])  # Take first row since preprocess_input adds batch dimension
        
        X_batch = np.array(X_batch)
        
        # Make batch prediction
        predictions = self.model.predict(X_batch)
        
        return predictions.tolist()
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the loaded model."""
        if self.model is None:
            return None
        
        if not hasattr(self.model, 'feature_importances_'):
            return None
        
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.model.feature_importances_))]
        else:
            feature_names = self.feature_names
        
        # Create importance dictionary
        importance_dict = dict(zip(feature_names, self.model.feature_importances_))
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_dict.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance
    
    def interpret_hv_score(self, hv_score: float) -> Dict[str, str]:
        """
        Interpret Hv score and provide health insights.
        
        Args:
            hv_score: Predicted Hv score (0-1 range)
            
        Returns:
            Dictionary with interpretation and recommendations
        """
        # Clamp score to valid range
        hv_score = max(0.0, min(1.0, hv_score))
        
        # Interpret score (these thresholds would be calibrated based on medical knowledge)
        if hv_score <= 0.2:
            risk_level = "Low"
            interpretation = "Low health risk. Vitals appear normal."
            recommendations = [
                "Continue maintaining healthy lifestyle",
                "Regular health check-ups recommended",
                "Monitor any changes in symptoms"
            ]
        elif hv_score <= 0.5:
            risk_level = "Moderate"
            interpretation = "Moderate health risk detected. Some vitals may be concerning."
            recommendations = [
                "Consider consulting healthcare provider",
                "Monitor vitals more frequently",
                "Evaluate lifestyle factors (diet, exercise, stress)"
            ]
        elif hv_score <= 0.7:
            risk_level = "High"
            interpretation = "High health risk. Multiple concerning vitals detected."
            recommendations = [
                "Consult healthcare provider soon",
                "Consider immediate lifestyle changes",
                "Monitor symptoms closely"
            ]
        else:
            risk_level = "Critical"
            interpretation = "Critical health risk. Immediate attention may be needed."
            recommendations = [
                "Seek immediate medical attention",
                "Do not delay healthcare consultation",
                "Monitor vital signs continuously"
            ]
        
        return {
            'hv_score': hv_score,
            'risk_level': risk_level,
            'interpretation': interpretation,
            'recommendations': recommendations
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        if self.model is None:
            return {"status": "No model loaded"}
        
        info = {
            "status": "Model loaded",
            "model_type": type(self.model).__name__,
            "feature_count": self.model.n_features_in_,
            "feature_names": self.feature_names,
            "scaler_loaded": self.scaler is not None
        }
        
        # Add training info if available
        if self.model_info:
            info.update({
                "training_info": self.model_info
            })
        
        return info


def main():
    """Demo the vitals inference engine."""
    print("Vitals Inference Engine Demo")
    print("=" * 40)
    
    # Create a dummy inference engine for demo
    engine = VitalsInferenceEngine()
    
    # For demo, we'll create a simple mock model
    from sklearn.ensemble import RandomForestRegressor
    import numpy as np
    
    # Create and train a simple model for demo
    np.random.seed(42)
    X_train = np.random.randn(100, 8)  # 8 features
    y_train = np.random.rand(100)      # Random Hv scores
    
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    
    # Manually set up the engine
    engine.model = model
    engine.feature_names = ['age', 'hr_median', 'spo2_median', 'diabetes', 
                           'hypertension', 'heart_disease', 'respiratory_condition', 'smoker']
    
    # Demo single prediction
    sample_vitals = {
        'age': 45,
        'hr_median': 80,
        'spo2_median': 95,
        'diabetes': 0,
        'hypertension': 1,
        'heart_disease': 0,
        'respiratory_condition': 0,
        'smoker': 0
    }
    
    print("Sample vitals data:")
    for key, value in sample_vitals.items():
        print(f"  {key}: {value}")
    
    # Predict Hv
    hv_score = engine.predict_hv(sample_vitals)
    print(f"\nPredicted Hv score: {hv_score:.3f}")
    
    # Get interpretation
    interpretation = engine.interpret_hv_score(hv_score)
    print(f"\nRisk Level: {interpretation['risk_level']}")
    print(f"Interpretation: {interpretation['interpretation']}")
    print("Recommendations:")
    for rec in interpretation['recommendations']:
        print(f"  • {rec}")
    
    print("\nVitals inference engine ready!")


if __name__ == "__main__":
    main()
