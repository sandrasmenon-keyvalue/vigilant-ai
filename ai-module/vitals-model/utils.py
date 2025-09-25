"""
Utilities for Vitals Model
Helper functions for data processing, visualization, and model evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')


def load_vitals_data(data_path: str, target_column: str = 'Hv') -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Load and prepare vitals data for training.
    
    Args:
        data_path: Path to CSV file with vitals data
        target_column: Name of target column
        
    Returns:
        Tuple of (dataframe, X, y, feature_names)
    """
    df = pd.read_csv(data_path)
    
    # Validate target column
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")
    
    # Separate features and target
    feature_columns = [col for col in df.columns if col != target_column]
    X = df[feature_columns].values
    y = df[target_column].values
    
    return df, X, y, feature_columns


def create_synthetic_vitals_data(n_samples: int = 1000, random_seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic vitals data for testing and demonstration.
    
    Args:
        n_samples: Number of samples to generate
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic vitals data
    """
    np.random.seed(random_seed)
    
    # Generate synthetic features
    age = np.random.randint(18, 90, size=n_samples)
    hr_median = np.random.normal(loc=75, scale=12, size=n_samples)
    spo2_median = np.clip(np.random.normal(loc=97, scale=2, size=n_samples), 80, 100)
    
    # Binary health conditions
    diabetes = np.random.binomial(1, 0.15, size=n_samples)
    hypertension = np.random.binomial(1, 0.25, size=n_samples)
    heart_disease = np.random.binomial(1, 0.10, size=n_samples)
    respiratory_condition = np.random.binomial(1, 0.08, size=n_samples)
    smoker = np.random.binomial(1, 0.20, size=n_samples)
    
    # Additional vitals features
    systolic_bp = np.random.normal(loc=120, scale=15, size=n_samples)
    diastolic_bp = np.random.normal(loc=80, scale=10, size=n_samples)
    temperature = np.random.normal(loc=98.6, scale=1, size=n_samples)
    respiratory_rate = np.random.normal(loc=16, scale=3, size=n_samples)
    
    # Target: Hv (heart rate variability risk score)
    hv_score = (
        0.2 * (np.abs(hr_median - 75) / 75) +  # deviation from normal HR
        0.3 * ((100 - spo2_median) / 20) +     # penalty for low SpO2
        0.1 * diabetes +
        0.1 * hypertension +
        0.15 * heart_disease +
        0.1 * respiratory_condition +
        0.05 * smoker +
        0.1 * (np.abs(systolic_bp - 120) / 120) +  # BP deviation
        np.random.normal(0, 0.05, n_samples)  # Add noise
    )
    
    # Clip to [0,1]
    hv_score = np.clip(hv_score, 0, 1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'age': age,
        'hr_median': hr_median,
        'spo2_median': spo2_median,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'temperature': temperature,
        'respiratory_rate': respiratory_rate,
        'diabetes': diabetes,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'respiratory_condition': respiratory_condition,
        'smoker': smoker,
        'Hv': hv_score
    })
    
    return df


def plot_data_distribution(df: pd.DataFrame, target_column: str = 'Hv', save_path: str = None):
    """
    Plot distribution of features and target variable.
    
    Args:
        df: DataFrame with vitals data
        target_column: Name of target column
        save_path: Path to save plot (optional)
    """
    # Get numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove target from features for separate plotting
    feature_columns = [col for col in numeric_columns if col != target_column]
    
    # Calculate number of subplots
    n_features = len(feature_columns) + 1  # +1 for target
    n_cols = 4
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Plot feature distributions
    for i, column in enumerate(feature_columns):
        axes[i].hist(df[column], bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{column} Distribution')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    # Plot target distribution
    if target_column in df.columns:
        axes[len(feature_columns)].hist(df[target_column], bins=30, alpha=0.7, 
                                       edgecolor='black', color='red')
        axes[len(feature_columns)].set_title(f'{target_column} Distribution (Target)')
        axes[len(feature_columns)].set_xlabel(target_column)
        axes[len(feature_columns)].set_ylabel('Frequency')
        axes[len(feature_columns)].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Data distribution plot saved to {save_path}")
    
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None):
    """
    Plot correlation matrix of features.
    
    Args:
        df: DataFrame with vitals data
        save_path: Path to save plot (optional)
    """
    # Get numeric columns only
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create plot
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Correlation matrix saved to {save_path}")
    
    plt.show()


def plot_feature_target_relationships(df: pd.DataFrame, target_column: str = 'Hv', 
                                     max_features: int = 8, save_path: str = None):
    """
    Plot relationships between features and target variable.
    
    Args:
        df: DataFrame with vitals data
        target_column: Name of target column
        max_features: Maximum number of features to plot
        save_path: Path to save plot (optional)
    """
    # Get numeric feature columns
    feature_columns = [col for col in df.select_dtypes(include=[np.number]).columns 
                      if col != target_column][:max_features]
    
    n_features = len(feature_columns)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    for i, feature in enumerate(feature_columns):
        # Scatter plot with trend line
        axes[i].scatter(df[feature], df[target_column], alpha=0.6)
        
        # Add trend line
        z = np.polyfit(df[feature], df[target_column], 1)
        p = np.poly1d(z)
        axes[i].plot(df[feature], p(df[feature]), "r--", alpha=0.8)
        
        # Calculate correlation
        corr = df[feature].corr(df[target_column])
        
        axes[i].set_xlabel(feature)
        axes[i].set_ylabel(target_column)
        axes[i].set_title(f'{feature} vs {target_column}\nCorrelation: {corr:.3f}')
        axes[i].grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Feature-target relationships plot saved to {save_path}")
    
    plt.show()


def plot_learning_curves(model, X: np.ndarray, y: np.ndarray, cv: int = 5, 
                        save_path: str = None):
    """
    Plot learning curves to analyze model performance vs training size.
    
    Args:
        model: Trained model
        X: Feature matrix
        y: Target values
        cv: Number of cross-validation folds
        save_path: Path to save plot (optional)
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='neg_mean_squared_error'
    )
    
    # Convert to positive RMSE
    train_rmse = np.sqrt(-train_scores)
    val_rmse = np.sqrt(-val_scores)
    
    train_rmse_mean = np.mean(train_rmse, axis=1)
    train_rmse_std = np.std(train_rmse, axis=1)
    val_rmse_mean = np.mean(val_rmse, axis=1)
    val_rmse_std = np.std(val_rmse, axis=1)
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(train_sizes, train_rmse_mean, 'o-', color='blue', label='Training RMSE')
    plt.fill_between(train_sizes, train_rmse_mean - train_rmse_std, 
                     train_rmse_mean + train_rmse_std, alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_rmse_mean, 'o-', color='red', label='Validation RMSE')
    plt.fill_between(train_sizes, val_rmse_mean - val_rmse_std, 
                     val_rmse_mean + val_rmse_std, alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Learning curves plot saved to {save_path}")
    
    plt.show()


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model") -> Dict:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        model_name: Name of the model for reporting
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Calculate metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # Additional metrics
    residuals = y_true - y_pred
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # Percentage of predictions within certain error bounds
    within_5_percent = np.mean(np.abs(residuals / y_true) < 0.05) * 100
    within_10_percent = np.mean(np.abs(residuals / y_true) < 0.10) * 100
    within_20_percent = np.mean(np.abs(residuals / y_true) < 0.20) * 100
    
    metrics = {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2_score': r2,
        'mape': mape,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'within_5_percent': within_5_percent,
        'within_10_percent': within_10_percent,
        'within_20_percent': within_20_percent
    }
    
    # Print summary
    print(f"\n{model_name} Performance Summary:")
    print("-" * 40)
    print(f"R² Score:           {r2:.4f}")
    print(f"RMSE:               {rmse:.4f}")
    print(f"MAE:                {mae:.4f}")
    print(f"MAPE:               {mape:.2f}%")
    print(f"Mean Residual:      {mean_residual:.4f}")
    print(f"Std Residual:       {std_residual:.4f}")
    print(f"Within 5% error:    {within_5_percent:.1f}%")
    print(f"Within 10% error:   {within_10_percent:.1f}%")
    print(f"Within 20% error:   {within_20_percent:.1f}%")
    
    return metrics


def create_model_report(model_metrics: Dict, feature_importance: Dict = None, 
                       save_path: str = None) -> str:
    """
    Create a comprehensive model report.
    
    Args:
        model_metrics: Dictionary with model evaluation metrics
        feature_importance: Dictionary with feature importance scores
        save_path: Path to save report (optional)
        
    Returns:
        Report as string
    """
    report_lines = []
    report_lines.append("VITALS-BASED HV PREDICTION MODEL REPORT")
    report_lines.append("=" * 50)
    report_lines.append("")
    
    # Model performance
    report_lines.append("MODEL PERFORMANCE:")
    report_lines.append("-" * 20)
    report_lines.append(f"Model Type:         {model_metrics.get('model_name', 'Unknown')}")
    report_lines.append(f"R² Score:           {model_metrics.get('r2_score', 0):.4f}")
    report_lines.append(f"RMSE:               {model_metrics.get('rmse', 0):.4f}")
    report_lines.append(f"MAE:                {model_metrics.get('mae', 0):.4f}")
    report_lines.append(f"MAPE:               {model_metrics.get('mape', 0):.2f}%")
    report_lines.append("")
    
    # Accuracy breakdown
    report_lines.append("PREDICTION ACCURACY:")
    report_lines.append("-" * 20)
    report_lines.append(f"Within 5% error:    {model_metrics.get('within_5_percent', 0):.1f}%")
    report_lines.append(f"Within 10% error:   {model_metrics.get('within_10_percent', 0):.1f}%")
    report_lines.append(f"Within 20% error:   {model_metrics.get('within_20_percent', 0):.1f}%")
    report_lines.append("")
    
    # Feature importance
    if feature_importance:
        report_lines.append("FEATURE IMPORTANCE:")
        report_lines.append("-" * 20)
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:10]):
            report_lines.append(f"{i+1:2d}. {feature:20s}: {importance:.4f}")
        report_lines.append("")
    
    # Usage instructions
    report_lines.append("USAGE INSTRUCTIONS:")
    report_lines.append("-" * 20)
    report_lines.append("1. Load the inference engine:")
    report_lines.append("   from inference import VitalsInferenceEngine")
    report_lines.append("   engine = VitalsInferenceEngine(model_path, scaler_path)")
    report_lines.append("")
    report_lines.append("2. Make predictions:")
    report_lines.append("   vitals_data = {'age': 45, 'hr_median': 80, ...}")
    report_lines.append("   hv_score = engine.predict_hv(vitals_data)")
    report_lines.append("")
    report_lines.append("3. Get interpretation:")
    report_lines.append("   result = engine.interpret_hv_score(hv_score)")
    report_lines.append("")
    
    report = "\n".join(report_lines)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Model report saved to {save_path}")
    
    return report


def main():
    """Demo the utilities."""
    print("Vitals Model Utilities Demo")
    print("=" * 30)
    
    # Create synthetic data
    print("Creating synthetic vitals data...")
    df = create_synthetic_vitals_data(n_samples=500)
    print(f"Created dataset with {len(df)} samples and {len(df.columns)-1} features")
    
    # Show data info
    print(f"\nDataset Info:")
    print(df.info())
    
    print(f"\nFirst few rows:")
    print(df.head())
    
    print(f"\nTarget statistics:")
    print(df['Hv'].describe())
    
    # Demo evaluation
    print(f"\nDemo model evaluation:")
    # Create fake predictions for demo
    y_true = df['Hv'].values
    y_pred = y_true + np.random.normal(0, 0.05, len(y_true))  # Add noise
    
    metrics = evaluate_model_performance(y_true, y_pred, "Demo Model")
    
    print("\nVitals model utilities ready!")


if __name__ == "__main__":
    main()
