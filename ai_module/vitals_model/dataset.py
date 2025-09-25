import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of synthetic samples
n_samples = 10000

# Generate synthetic features
age = np.random.randint(18, 90, size=n_samples)  # Age between 18 and 90
hr_median = np.random.normal(loc=75, scale=12, size=n_samples)  # Avg HR ~75 bpm
spo2_median = np.clip(np.random.normal(loc=97, scale=2, size=n_samples), 80, 100)  # SpO2 ~97%

# Binary health conditions (0/1)
diabetes = np.random.binomial(1, 0.15, size=n_samples)  # ~15% prevalence
hypertension = np.random.binomial(1, 0.25, size=n_samples)  # ~25% prevalence
heart_disease = np.random.binomial(1, 0.10, size=n_samples)  # ~10% prevalence
respiratory_condition = np.random.binomial(1, 0.08, size=n_samples)  # ~8% prevalence
smoker = np.random.binomial(1, 0.20, size=n_samples)  # ~20% prevalence

# Target: health risk score between 0 and 1 (simulated)
# Higher HR (tachy/brady), low SpO2, and conditions increase risk
risk_score = (
    0.3 * (np.abs(hr_median - 75) / 75) +  # deviation from normal HR
    0.4 * ((100 - spo2_median) / 20) +     # penalty for low SpO2
    0.1 * diabetes +
    0.1 * hypertension +
    0.15 * heart_disease +
    0.1 * respiratory_condition +
    0.05 * smoker
)

# Clip to [0,1]
risk_score = np.clip(risk_score, 0, 1)

# Create DataFrame
df = pd.DataFrame({
    "age": age,
    "hr_median": hr_median,
    "spo2_median": spo2_median,
    "diabetes": diabetes,
    "hypertension": hypertension,
    "heart_disease": heart_disease,
    "respiratory_condition": respiratory_condition,
    "smoker": smoker,
    "Hv": risk_score
})