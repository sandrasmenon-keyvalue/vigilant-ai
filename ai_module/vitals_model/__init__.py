"""
Vitals Model Module
Heart rate variability (Hv) prediction using XGBoost on vitals data.
"""

from .model_training import VitalsModelTrainer
from .inference import VitalsInferenceEngine
from .utils import (
    load_vitals_data, 
    create_synthetic_vitals_data, 
    evaluate_model_performance,
    create_model_report
)

__version__ = "1.0.0"
__all__ = [
    "VitalsModelTrainer",
    "VitalsInferenceEngine", 
    "load_vitals_data",
    "create_synthetic_vitals_data",
    "evaluate_model_performance",
    "create_model_report"
]
