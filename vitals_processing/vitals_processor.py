"""
Vitals Processor for HV Prediction
Processes HR, SpO2, age, and health conditions to predict HV (Health Vitals) score.
"""

import sys
import os
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Add ai-module to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ai-module', 'vitals-model'))

try:
    from ai_module.vitals_model.inference import VitalsInferenceEngine
except ImportError:
    # Fallback for when running from different directory
    sys.path.append(os.path.join(os.path.dirname(__file__), 'ai-module', 'vitals-model'))
    from ai_module.vitals_model.inference import VitalsInferenceEngine

logger = logging.getLogger(__name__)


class VitalsProcessor:
    """Processes vitals data and predicts HV score using trained XGBoost model."""
    
    def __init__(self, 
                 model_path: str = "trained_models/vital-training/vitals_hv_model_xgboost.pkl",
                 scaler_path: str = "trained_models/vital-training/vitals_feature_scaler_xgboost.pkl"):
        """
        Initialize vitals processor.
        
        Args:
            model_path: Path to trained XGBoost model
            scaler_path: Path to feature scaler
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.inference_engine = None
        self.is_loaded = False
        
        # Default health conditions (can be overridden)
        self.default_health_conditions = {
            'diabetes': 0,
            'hypertension': 0,
            'heart_disease': 0,
            'respiratory_condition': 0,
            'smoker': 0
        }
        
        # Environmental conditions thresholds
        self.environmental_thresholds = {
            'temperature_celsius': {
                'min_safe': 18.0,    # 18°C minimum safe temperature
                'max_safe': 25.0,    # 25°C maximum safe temperature
                'critical_high': 30.0,  # 30°C critical high
                'critical_low': 10.0    # 10°C critical low
            },
            'co2_ppm': {
                'normal': 400,       # Normal outdoor CO2 level
                'acceptable': 1000,  # Acceptable indoor level
                'concerning': 2000,  # Concerning level
                'dangerous': 5000    # Dangerous level
            }
        }
        
        # Load the model
        self._load_model()
    
    def _load_model(self):
        """Load the trained vitals model and scaler."""
        try:
            # Check if model files exist
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            if not Path(self.scaler_path).exists():
                raise FileNotFoundError(f"Scaler file not found: {self.scaler_path}")
            
            # Initialize inference engine
            self.inference_engine = VitalsInferenceEngine()
            self.inference_engine.load_model(self.model_path, self.scaler_path)
            
            self.is_loaded = True
            logger.info("✅ Vitals model loaded successfully")
            
            # Print model info
            model_info = self.inference_engine.get_model_info()
            logger.info(f"Model type: {model_info.get('model_type', 'Unknown')}")
            logger.info(f"Feature count: {model_info.get('feature_count', 'Unknown')}")
            logger.info(f"Expected features: {model_info.get('feature_names', 'Unknown')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load vitals model: {e}")
            self.is_loaded = False
            raise
    
    def set_health_conditions(self, health_conditions: Dict[str, int]):
        """
        Set health conditions for the user.
        
        Args:
            health_conditions: Dictionary with health condition flags
                Keys: 'diabetes', 'hypertension', 'heart_disease', 'respiratory_condition', 'smoker'
                Values: 0 (no condition) or 1 (has condition)
        """
        for key, value in health_conditions.items():
            if key in self.default_health_conditions:
                self.default_health_conditions[key] = int(value)
        
        logger.info(f"Health conditions updated: {self.default_health_conditions}")
    
    def set_environmental_thresholds(self, temperature_thresholds: Optional[Dict[str, float]] = None,
                                   co2_thresholds: Optional[Dict[str, int]] = None):
        """
        Set environmental condition thresholds.
        
        Args:
            temperature_thresholds: Temperature thresholds in Celsius
            co2_thresholds: CO2 thresholds in PPM
        """
        if temperature_thresholds:
            self.environmental_thresholds['temperature_celsius'].update(temperature_thresholds)
        
        if co2_thresholds:
            self.environmental_thresholds['co2_ppm'].update(co2_thresholds)
        
        logger.info(f"Environmental thresholds updated: {self.environmental_thresholds}")
    
    def check_environmental_conditions(self, temperature: float, co2_level: float) -> Dict[str, Any]:
        """
        Check environmental conditions and return status.
        
        Args:
            temperature: Ambient temperature in Celsius
            co2_level: CO2 level in PPM
            
        Returns:
            Dictionary with environmental status and alerts
        """
        temp_thresholds = self.environmental_thresholds['temperature_celsius']
        co2_thresholds = self.environmental_thresholds['co2_ppm']
        
        # Check temperature
        temp_status = "normal"
        temp_alert = None
        
        if temperature < temp_thresholds['critical_low']:
            temp_status = "critical_low"
            temp_alert = f"Critical low temperature: {temperature}°C"
        elif temperature < temp_thresholds['min_safe']:
            temp_status = "low"
            temp_alert = f"Low temperature: {temperature}°C"
        elif temperature > temp_thresholds['critical_high']:
            temp_status = "critical_high"
            temp_alert = f"Critical high temperature: {temperature}°C"
        elif temperature > temp_thresholds['max_safe']:
            temp_status = "high"
            temp_alert = f"High temperature: {temperature}°C"
        
        # Check CO2 level
        co2_status = "normal"
        co2_alert = None
        
        if co2_level > co2_thresholds['dangerous']:
            co2_status = "dangerous"
            co2_alert = f"Dangerous CO2 level: {co2_level} PPM"
        elif co2_level > co2_thresholds['concerning']:
            co2_status = "concerning"
            co2_alert = f"Concerning CO2 level: {co2_level} PPM"
        elif co2_level > co2_thresholds['acceptable']:
            co2_status = "elevated"
            co2_alert = f"Elevated CO2 level: {co2_level} PPM"
        
        # Overall environmental risk
        environmental_risk = "low"
        if temp_status in ["critical_low", "critical_high"] or co2_status == "dangerous":
            environmental_risk = "critical"
        elif temp_status in ["low", "high"] or co2_status in ["concerning", "elevated"]:
            environmental_risk = "moderate"
        
        return {
            'temperature': {
                'value': temperature,
                'status': temp_status,
                'alert': temp_alert
            },
            'co2_level': {
                'value': co2_level,
                'status': co2_status,
                'alert': co2_alert
            },
            'overall_risk': environmental_risk,
            'alerts': [alert for alert in [temp_alert, co2_alert] if alert is not None]
        }
    
    def process_vitals(self, hr: float, spo2: float, age: int, 
                      health_conditions: Optional[Dict[str, int]] = None,
                      temperature: Optional[float] = None,
                      co2_level: Optional[float] = None) -> Dict[str, Any]:
        """
        Process vitals data and predict HV score.
        
        Args:
            hr: Heart rate in BPM
            spo2: Blood oxygen saturation percentage
            age: Age in years
            health_conditions: Optional health conditions dict (uses defaults if not provided)
            temperature: Optional ambient temperature in Celsius
            co2_level: Optional CO2 level in PPM
            
        Returns:
            Dictionary with HV prediction, interpretation, and environmental conditions
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Cannot process vitals.")
        
        # Use provided health conditions or defaults
        if health_conditions is None:
            health_conditions = self.default_health_conditions.copy()
        else:
            # Merge with defaults
            merged_conditions = self.default_health_conditions.copy()
            merged_conditions.update(health_conditions)
            health_conditions = merged_conditions
        
        # Prepare input data in the exact format expected by the model
        vitals_data = {
            'age': int(age),
            'hr_median': float(hr),
            'spo2_median': float(spo2),
            'diabetes': int(health_conditions.get('diabetes', 0)),
            'hypertension': int(health_conditions.get('hypertension', 0)),
            'heart_disease': int(health_conditions.get('heart_disease', 0)),
            'respiratory_condition': int(health_conditions.get('respiratory_condition', 0)),
            'smoker': int(health_conditions.get('smoker', 0))
        }
        
        try:
            print(f"##########################Vitals Data Input: {vitals_data}")
            # Predict HV score
            hv_score = self.inference_engine.predict_hv(vitals_data)
            print(f"##########################HV Score from Model: {hv_score}")
            
            # Get interpretation
            interpretation = self.inference_engine.interpret_hv_score(hv_score)
            
            # Check environmental conditions if provided
            environmental_data = None
            if temperature is not None and co2_level is not None:
                environmental_data = self.check_environmental_conditions(temperature, co2_level)
            
            # Prepare result
            result = {
                'input_data': vitals_data,
                'hv_score': float(hv_score),
                'risk_level': interpretation['risk_level'],
                'interpretation': interpretation['interpretation'],
                'recommendations': interpretation['recommendations'],
                'timestamp': time.time(),
                'environmental_conditions': environmental_data
            }
            
            logger.info(f"HV prediction: {hv_score:.3f} ({interpretation['risk_level']} risk)")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing vitals: {e}")
            raise
    
    def process_websocket_vitals(self, vitals_data: Dict[str, Any], 
                                age: int = 35,
                                health_conditions: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
        """
        Process vitals data from WebSocket and predict HV score.
        
        Args:
            vitals_data: WebSocket vitals data with 'hr', 'spo2', 'temperature', 'co2_level' keys
            age: Age in years (default: 35)
            health_conditions: Optional health conditions dict
            
        Returns:
            Dictionary with HV prediction, interpretation, and environmental conditions
        """
        hr = vitals_data.get('hr', 0)
        spo2 = vitals_data.get('spo2', 0)
        temperature = vitals_data.get('temperature', None)
        co2_level = vitals_data.get('co2_level', None)
        
        print(f"##########################WebSocket Vitals: HR={hr}, SpO2={spo2}, Age={age}")
        print(f"##########################Health Conditions: {health_conditions}")
        
        if hr <= 0 or spo2 <= 0:
            print(f"##########################ERROR: Invalid vitals data: HR={hr}, SpO2={spo2}")
            raise ValueError(f"Invalid vitals data: HR={hr}, SpO2={spo2}")
        
        return self.process_vitals(hr, spo2, age, health_conditions, temperature, co2_level)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"status": "Model not loaded"}
        
        return self.inference_engine.get_model_info()
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from the model."""
        if not self.is_loaded:
            return None
        
        return self.inference_engine.get_feature_importance()


# Example usage and testing
def main():
    """Test the vitals processor."""
    import time
    
    print("Vitals Processor Test")
    print("=" * 40)
    
    try:
        # Initialize processor
        processor = VitalsProcessor()
        
        # Set some health conditions for testing
        health_conditions = {
            'diabetes': 0,
            'hypertension': 1,  # Has hypertension
            'heart_disease': 0,
            'respiratory_condition': 0,
            'smoker': 0
        }
        processor.set_health_conditions(health_conditions)
        
        # Test with sample vitals data
        test_cases = [
            {'hr': 75, 'spo2': 98, 'age': 35},  # Normal
            {'hr': 95, 'spo2': 92, 'age': 45},  # Elevated HR, low SpO2
            {'hr': 60, 'spo2': 99, 'age': 25},  # Low HR, good SpO2
            {'hr': 110, 'spo2': 88, 'age': 55}, # High HR, very low SpO2
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"  HR: {test_case['hr']} BPM")
            print(f"  SpO2: {test_case['spo2']}%")
            print(f"  Age: {test_case['age']} years")
            
            result = processor.process_vitals(
                hr=test_case['hr'],
                spo2=test_case['spo2'],
                age=test_case['age']
            )
            
            print(f"  HV Score: {result['hv_score']:.3f}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Interpretation: {result['interpretation']}")
        
        # Test WebSocket data format
        print(f"\nWebSocket Data Test:")
        websocket_data = {'hr': 80, 'spo2': 96, 'quality': 'good'}
        result = processor.process_websocket_vitals(websocket_data, age=40)
        print(f"  HV Score: {result['hv_score']:.3f}")
        print(f"  Risk Level: {result['risk_level']}")
        
        # Show feature importance
        importance = processor.get_feature_importance()
        if importance:
            print(f"\nFeature Importance:")
            for feature, imp in list(importance.items())[:5]:
                print(f"  {feature}: {imp:.3f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    main()
