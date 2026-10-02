"""
BatterySentinel – Source Package
=================================
BATTERYSENTINEL: Probabilistic Early-Failure Prediction for Lithium-Ion Batteries

Modules
-------
data_loader        : Raw .mat file loading and parsing
preprocessing      : Cycle-level data cleaning and normalisation
feature_engineering: Derived feature computation (SOH, EOL flag, energy, etc.)
statistics_engine  : Descriptive statistics (mean, variance, quantiles, cov, etc.)
probability_engine : Continuous RV analysis, KDE, PDF, CDF, anomaly detection
bayesian_risk      : Bayesian degradation-risk engine (P(D|Evidence))
polynomial_model   : Polynomial curve fitting and SOH forecasting
clustering         : K-Means unsupervised battery-state clustering
evaluation         : Model metrics (Brier, LogLoss, ROC-AUC, calibration, etc.)
visualization      : Reusable Plotly / Matplotlib / Seaborn plotting utilities
"""

__version__ = "1.0.0"
__author__ = "BatterySentinel Project"
