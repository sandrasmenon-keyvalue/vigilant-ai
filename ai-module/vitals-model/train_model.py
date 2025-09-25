"""
Training Script for Vitals-based Hv Prediction Model
Use this script to train an XGBoost model on your vitals dataset for heart rate variability prediction.
"""

import os
import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add the ai-module to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import df as sample_df  # Import the dataset from dataset.py
from model_training import VitalsModelTrainer
from inference import VitalsInferenceEngine


def load_dataset(dataset_path: str = None) -> pd.DataFrame:
    """
    Load vitals dataset for training.
    
    Args:
        dataset_path: Path to CSV file with vitals data (optional)
        
    Returns:
        DataFrame with vitals features and Hv target
    """
    if dataset_path and Path(dataset_path).exists():
        print(f"Loading dataset from {dataset_path}")
        df = pd.read_csv(dataset_path)
        print(f"Loaded {len(df)} samples from external dataset")
    else:
        print("Using synthetic dataset from dataset.py")
        df = sample_df.copy()
        print(f"Loaded {len(df)} samples from synthetic dataset")
    
    # Validate required columns
    required_columns = ['Hv']  # Target column
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Show dataset info
    print(f"\nDataset Info:")
    print(f"  Samples: {len(df)}")
    print(f"  Features: {len(df.columns) - 1}")  # Excluding target
    print(f"  Target range: [{df['Hv'].min():.3f}, {df['Hv'].max():.3f}]")
    print(f"  Target mean: {df['Hv'].mean():.3f}")
    
    # Show feature columns
    feature_columns = [col for col in df.columns if col != 'Hv']
    print(f"  Feature columns: {feature_columns}")
    
    return df


def prepare_training_data(df: pd.DataFrame) -> tuple:
    """
    Prepare features and target for training.
    
    Args:
        df: DataFrame with vitals data
        
    Returns:
        Tuple of (X, y, feature_names)
    """
    # Separate features and target
    feature_columns = [col for col in df.columns if col != 'Hv']
    X = df[feature_columns].values
    y = df['Hv'].values
    
    print(f"\nTraining Data Preparation:")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Target vector shape: {y.shape}")
    
    # Check for missing values
    if np.isnan(X).any():
        print("  ⚠ Warning: Missing values detected in features")
        # Simple imputation with median
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        print("  ✓ Missing values imputed with median")
    
    if np.isnan(y).any():
        print("  ⚠ Warning: Missing values detected in target")
        # Remove samples with missing target
        valid_indices = ~np.isnan(y)
        X = X[valid_indices]
        y = y[valid_indices]
        print(f"  ✓ Removed {np.sum(~valid_indices)} samples with missing target")
    
    return X, y, feature_columns


def train_model(X: np.ndarray, y: np.ndarray, feature_names: list, 
                output_dir: str, model_type: str = 'xgboost') -> tuple:
    """
    Train a vitals-based Hv prediction model.
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        output_dir: Directory to save trained model
        model_type: Type of model to train
        
    Returns:
        Tuple of (model_path, scaler_path, results)
    """
    print(f"\nTraining {model_type} model for Hv prediction...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Train model
    trainer = VitalsModelTrainer(model_type=model_type)
    results = trainer.train_model(X, y, feature_names)
    
    # Save model and scaler
    model_path = output_path / f"vitals_hv_model_{model_type}.pkl"
    scaler_path = output_path / f"vitals_feature_scaler_{model_type}.pkl"
    
    trainer.save_model(str(model_path), str(scaler_path))
    
    # Show and save feature importance
    importance = trainer.get_feature_importance()
    if importance:
        print(f"\nTop {min(10, len(importance))} most important features:")
        for i, (feature, imp) in enumerate(list(importance.items())[:10]):
            print(f"  {i+1:2d}. {feature:20s}: {imp:.4f}")
        
        # Save feature importance
        importance_path = output_path / f"feature_importance_{model_type}.json"
        # Convert numpy float32 to Python float for JSON serialization
        importance_serializable = {k: float(v) for k, v in importance.items()}
        with open(importance_path, 'w') as f:
            json.dump(importance_serializable, f, indent=2)
        print(f"  ✓ Feature importance saved to {importance_path}")
    
    # Create visualizations
    try:
        print("\nCreating visualizations...")
        
        # Feature importance plot
        plots_dir = output_path / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        importance_plot_path = plots_dir / f"feature_importance_{model_type}.png"
        trainer.plot_feature_importance(save_path=str(importance_plot_path))
        
        # Get test data for additional plots
        X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
        
        # Predictions plot
        predictions_plot_path = plots_dir / f"predictions_{model_type}.png"
        trainer.plot_predictions(X_test, y_test, save_path=str(predictions_plot_path))
        
        # Residuals plot
        residuals_plot_path = plots_dir / f"residuals_{model_type}.png"
        trainer.plot_residuals(X_test, y_test, save_path=str(residuals_plot_path))
        
        print(f"  ✓ Plots saved to {plots_dir}")
        
    except Exception as e:
        print(f"  ⚠ Warning: Could not create plots: {e}")
    
    return str(model_path), str(scaler_path), results


def test_model(model_path: str, scaler_path: str, test_samples: int = 5):
    """
    Test the trained model with sample predictions.
    
    Args:
        model_path: Path to trained model
        scaler_path: Path to feature scaler
        test_samples: Number of test samples to generate
    """
    print(f"\nTesting trained model...")
    
    # Load inference engine
    engine = VitalsInferenceEngine(model_path, scaler_path)
    
    # Generate test samples
    np.random.seed(42)
    test_data = []
    
    for i in range(test_samples):
        sample = {
            'age': np.random.randint(25, 80),
            'hr_median': np.random.normal(75, 12),
            'spo2_median': np.clip(np.random.normal(97, 2), 85, 100),
            'diabetes': np.random.binomial(1, 0.15),
            'hypertension': np.random.binomial(1, 0.25),
            'heart_disease': np.random.binomial(1, 0.10),
            'respiratory_condition': np.random.binomial(1, 0.08),
            'smoker': np.random.binomial(1, 0.20)
        }
        test_data.append(sample)
    
    # Make predictions
    print(f"\nSample Predictions:")
    print("-" * 80)
    
    for i, sample in enumerate(test_data, 1):
        # Get prediction with confidence
        result = engine.predict_with_confidence(sample)
        hv_score = result['hv_prediction']
        
        # Get interpretation
        interpretation = engine.interpret_hv_score(hv_score)
        
        print(f"Sample {i}:")
        print(f"  Input: Age={sample['age']}, HR={sample['hr_median']:.1f}, "
              f"SpO2={sample['spo2_median']:.1f}")
        print(f"  Conditions: DM={sample['diabetes']}, HTN={sample['hypertension']}, "
              f"HD={sample['heart_disease']}, Smoker={sample['smoker']}")
        print(f"  Predicted Hv: {hv_score:.3f} ({interpretation['risk_level']} risk)")
        
        if result['confidence_available']:
            print(f"  Confidence interval: ±{result['confidence_interval']:.3f}")
        
        print()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train vitals-based Hv prediction model')
    parser.add_argument('--dataset', 
                       help='Path to CSV file with vitals data (optional, uses synthetic data if not provided)')
    parser.add_argument('--output', default='trained_models', 
                       help='Output directory for trained models')
    parser.add_argument('--model', choices=['xgboost', 'linear', 'random_forest'], 
                       default='xgboost', help='Model type to train')
    parser.add_argument('--test', action='store_true',
                       help='Test the trained model with sample predictions')
    
    args = parser.parse_args()
    
    try:
        print("🧠 Vitals-based Hv Prediction Model Training")
        print("=" * 50)
        
        # Load dataset
        df = load_dataset(args.dataset)
        
        # Prepare training data
        X, y, feature_names = prepare_training_data(df)
        
        # Train model
        model_path, scaler_path, results = train_model(
            X, y, feature_names, args.output, args.model
        )
        
        print(f"\n🎉 Training completed successfully!")
        print(f"📁 Model saved to: {model_path}")
        print(f"📁 Scaler saved to: {scaler_path}")
        
        # Print key results
        print(f"\n📊 Model Performance:")
        print(f"  R² Score: {results['r2_score']:.4f}")
        print(f"  RMSE: {results['rmse']:.4f}")
        print(f"  MAE: {results['mae']:.4f}")
        print(f"  MAPE: {results['mape']:.2f}%")
        
        # Test model if requested
        if args.test:
            test_model(model_path, scaler_path)
        
        print(f"\n💡 Usage Examples:")
        print(f"  # Test the model:")
        print(f"  python train_model.py --dataset your_data.csv --test")
        print(f"  ")
        print(f"  # Use for inference:")
        print(f"  from inference import VitalsInferenceEngine")
        print(f"  engine = VitalsInferenceEngine('{model_path}', '{scaler_path}')")
        print(f"  hv_score = engine.predict_hv(your_vitals_data)")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
