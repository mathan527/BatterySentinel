# BatterySentinel

## Probabilistic Early-Failure Prediction for Lithium-Ion Batteries

> **Research Question:** Can statistical dependencies and probabilistic evidence
> from battery telemetry provide an early indication of lithium-ion battery
> degradation before conventional end-of-life conditions are reached?

---

## Overview

BatterySentinel is an **educational / research ML project** that applies
Unit 1 probability and statistics concepts to lithium-ion battery
degradation analysis.

**Primary focus:** Statistics + Probability + Bayesian Risk Estimation +
Battery Degradation — NOT a generic Random Forest / LSTM black-box predictor.

---

## Project Structure

```
BatterySentinel/
├── app.py                    # Streamlit dashboard (Phase 9)
├── README.md
├── requirements.txt
├── .gitignore
│
├── config/
│   └── config.yaml           # Tunable parameters (EOL threshold, etc.)
│
├── data/
│   ├── raw/                  # Place NASA .mat files here
│   ├── interim/              # Intermediate processed data
│   └── processed/            # Final datasets and reports
│
├── notebooks/                # Jupyter notebooks (one per concept)
│
├── src/                      # Core Python modules
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── statistics_engine.py
│   ├── probability_engine.py
│   ├── bayesian_risk.py
│   ├── polynomial_model.py
│   ├── clustering.py
│   ├── evaluation.py
│   └── visualization.py
│
├── scripts/                  # Runnable pipeline scripts
│   ├── inspect_dataset.py    # PHASE 1
│   ├── preprocess_dataset.py # PHASE 2-3
│   └── train_models.py       # PHASE 5-7
│
├── experiments/              # Reproducible experiment scripts
│
├── models/                   # Saved model artefacts
├── reports/                  # Figures and CSVs
└── tests/                    # pytest unit tests
```

---

## Getting Started

### 1. Download the Dataset

Download the NASA Lithium-Ion Battery Aging Dataset from Kaggle:

- **URL:** https://www.kaggle.com/datasets/patrickfleith/nasa-battery-dataset
- **Search:** `NASA Li-ion Battery Aging Dataset` on Kaggle
- **Expected files:** `B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` (and others)

Place the downloaded `.mat` files inside:

```
data/raw/
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Phase 1 — Dataset Inspection

```bash
python scripts/inspect_dataset.py
```

This produces:
- `data/processed/dataset_report.json`
- `data/processed/dataset_report.txt`

---

## Unit 1 Concepts Demonstrated

| Concept | Implementation |
|---|---|
| Machine Learning – What & Why | Framing: supervised + unsupervised |
| Supervised Learning | Gaussian Naive Bayes (healthy vs degraded) |
| Unsupervised Learning | K-Means battery clustering |
| Polynomial Curve Fitting | Linear / Quadratic / Cubic SOH models |
| Probability Theory | Full Bayesian degradation risk engine |
| Discrete Random Variables | EOL flag distribution |
| Fundamental Probability Rules | Prior, likelihood, evidence, posterior |
| Bayes Rule | P(Degradation\|Evidence) engine |
| Independence | Statistical tests on features |
| Conditional Independence | Naive Bayes assumption analysis |
| Continuous Random Variables | Voltage, Current, Temp, Capacity |
| Quantiles | IQR-based anomaly detection |
| Mean / Variance | Per-cycle summary statistics |
| Probability Density | KDE plots: healthy vs degraded |
| Expectation | E[capacity], E[SOH], etc. |
| Covariance | Temperature ↔ Capacity ↔ SOH |

---

## Important Limitations

1. NASA laboratory data may not represent all real-world battery chemistries.
2. The EOL threshold (default 70% SOH) is a **configurable research parameter** — not a safety certification standard.
3. Bayesian conditional-independence assumptions simplify the true joint distribution.
4. Polynomial extrapolation is an **estimate** — not a guaranteed forecast.
5. Risk probability scores are model outputs — not guaranteed failure probabilities.
6. **This project is for educational / research purposes only. It is NOT a safety certification system.**

---

## Hypothesis

- **H₀:** Battery degradation cannot be reliably distinguished from normal behavior using the selected statistical features.
- **H₁:** Statistical relationships among battery telemetry variables provide measurable evidence of early battery degradation.

---

## Development Phases

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Dataset inspection | ✅ Complete |
| Phase 2 | Data pipeline & preprocessing | ⬜ Pending |
| Phase 3 | Cycle-level feature extraction | ⬜ Pending |
| Phase 4 | Statistical analysis | ⬜ Pending |
| Phase 5 | Bayesian risk engine | ⬜ Pending |
| Phase 6 | Polynomial degradation model | ⬜ Pending |
| Phase 7 | Clustering | ⬜ Pending |
| Phase 8 | Model evaluation | ⬜ Pending |
| Phase 9 | Streamlit dashboard | ⬜ Pending |
| Phase 10 | Tests | ⬜ Pending |
| Phase 11 | Final research report | ⬜ Pending |
