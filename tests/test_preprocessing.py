"""
BatterySentinel — tests/test_preprocessing.py
===============================================
Pytest tests for the preprocessing and feature engineering pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import compute_soh, assign_degradation_label, report_missing_values
from src.feature_engineering import engineer_features, get_feature_columns


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_cycle_df():
    """Minimal synthetic cycle DataFrame for unit tests."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "battery_id": ["B9999"] * n,
        "cycle_number": range(1, n + 1),
        "uid": range(1, n + 1),
        "ambient_temperature": [24.0] * n,
        "voltage_mean": np.random.uniform(3.3, 3.9, n),
        "voltage_std": np.random.uniform(0.1, 0.4, n),
        "voltage_min": np.random.uniform(2.5, 3.0, n),
        "voltage_max": np.random.uniform(4.0, 4.2, n),
        "current_mean": np.random.uniform(-2.0, -0.5, n),
        "current_std": np.random.uniform(0.1, 0.5, n),
        "current_min": np.random.uniform(-2.5, -2.0, n),
        "current_max": np.zeros(n),
        "temperature_mean": np.random.uniform(20, 40, n),
        "temperature_std": np.random.uniform(1, 5, n),
        "temperature_min": np.random.uniform(15, 25, n),
        "temperature_max": np.random.uniform(35, 45, n),
        "discharge_duration": np.random.uniform(3000, 6000, n),
        # Decreasing capacity with noise
        "capacity": np.linspace(1.85, 1.30, n) + np.random.normal(0, 0.02, n),
        "soh": np.linspace(100.0, 70.0, n),
        "Re_Ohm": np.linspace(0.055, 0.095, n),
        "Rct_Ohm": np.linspace(0.15, 0.22, n),
        "eol_flag": [0] * 40 + [1] * 10,
        "degradation_label": (["Healthy"] * 15 + ["Early Degradation"] * 25 + ["Degraded"] * 10),
        "data_quality_flag": [""] * n,
    })


@pytest.fixture
def sample_cfg():
    return {
        "soh": {"eol_threshold": 0.70},
        "labels": {"early_degradation_soh": 0.85},
        "anomaly": {"iqr_multiplier": 1.5},
    }


# ─────────────────────────────────────────────────────────────────────────────
# compute_soh tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeSOH:
    def test_max_reference_anchors_at_100(self):
        caps = pd.Series([1.8, 1.75, 1.7, 1.65, 1.6, 1.55, 1.5])
        soh = compute_soh(caps, reference="max")
        assert float(soh.max()) == pytest.approx(100.0, abs=0.01)

    def test_soh_is_decreasing_for_monotonic_capacity(self):
        caps = pd.Series([1.8, 1.7, 1.6, 1.5, 1.4])
        soh = compute_soh(caps, reference="max")
        for i in range(1, len(soh)):
            assert soh.iloc[i] <= soh.iloc[i - 1] + 0.001

    def test_max_reference_avoids_formation_distortion(self):
        # Battery starts at 0.1 (formation) then climbs to 1.8
        caps = pd.Series([0.1, 0.5, 1.0, 1.5, 1.8, 1.75, 1.70])
        soh = compute_soh(caps, reference="max")
        # With max reference, peak SOH = 100%, not >1000%
        assert float(soh.max()) == pytest.approx(100.0, abs=0.01)
        assert float(soh.iloc[0]) < 20.0  # first cycle is low

    def test_first_reference(self):
        caps = pd.Series([2.0, 1.8, 1.6, 1.4])
        soh = compute_soh(caps, reference="first")
        assert float(soh.iloc[0]) == pytest.approx(100.0, abs=0.01)
        assert float(soh.iloc[-1]) == pytest.approx(70.0, abs=0.01)

    def test_nan_handling(self):
        caps = pd.Series([1.8, np.nan, 1.6, 1.5])
        soh = compute_soh(caps, reference="max")
        assert soh.notna().sum() == 3  # NaN preserved

    def test_all_nan_returns_nan_series(self):
        caps = pd.Series([np.nan, np.nan, np.nan])
        soh = compute_soh(caps, reference="max")
        assert soh.isna().all()

    def test_invalid_reference_raises(self):
        caps = pd.Series([1.8, 1.7])
        with pytest.raises(ValueError):
            compute_soh(caps, reference="invalid_ref")


# ─────────────────────────────────────────────────────────────────────────────
# assign_degradation_label tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDegradationLabel:
    def test_healthy_above_85(self):
        assert assign_degradation_label(90.0, 0.70, 0.85) == "Healthy"

    def test_early_degradation_between_70_85(self):
        assert assign_degradation_label(77.0, 0.70, 0.85) == "Early Degradation"

    def test_degraded_below_70(self):
        assert assign_degradation_label(65.0, 0.70, 0.85) == "Degraded"
        assert assign_degradation_label(0.0, 0.70, 0.85) == "Degraded"

    def test_exactly_at_eol_boundary(self):
        assert assign_degradation_label(70.0, 0.70, 0.85) == "Degraded"

    def test_nan_returns_unknown(self):
        assert assign_degradation_label(np.nan) == "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFeatureEngineering:
    def test_adds_range_features(self, sample_cycle_df, sample_cfg):
        result = engineer_features(sample_cycle_df, sample_cfg)
        for col in ["voltage_range", "current_range", "temperature_range"]:
            assert col in result.columns

    def test_range_equals_max_minus_min(self, sample_cycle_df, sample_cfg):
        result = engineer_features(sample_cycle_df, sample_cfg)
        expected = result["voltage_max"] - result["voltage_min"]
        pd.testing.assert_series_equal(result["voltage_range"], expected,
                                       check_names=False, rtol=1e-5)

    def test_adds_coefficient_of_variation(self, sample_cycle_df, sample_cfg):
        result = engineer_features(sample_cycle_df, sample_cfg)
        assert "voltage_cv" in result.columns

    def test_adds_capacity_fade(self, sample_cycle_df, sample_cfg):
        result = engineer_features(sample_cycle_df, sample_cfg)
        assert "capacity_fade" in result.columns
        # Fade should be >= 0 (capacity declining from max)
        assert (result["capacity_fade"].dropna() >= -0.1).all()

    def test_adds_cycle_norm(self, sample_cycle_df, sample_cfg):
        result = engineer_features(sample_cycle_df, sample_cfg)
        assert "cycle_norm" in result.columns
        max_norm = result["cycle_norm"].max()
        assert max_norm == pytest.approx(1.0, abs=0.001)

    def test_log_resistance_all_positive(self, sample_cycle_df, sample_cfg):
        result = engineer_features(sample_cycle_df, sample_cfg)
        assert "log_Re" in result.columns
        # log_Re should be finite (Re_Ohm > 0 in sample)
        assert result["log_Re"].isna().sum() == 0

    def test_degradation_bin_values(self, sample_cycle_df, sample_cfg):
        result = engineer_features(sample_cycle_df, sample_cfg)
        assert "degradation_bin" in result.columns
        valid_bins = {-1, 0, 1, 2}
        assert set(result["degradation_bin"].unique()).issubset(valid_bins)

    def test_no_rows_dropped(self, sample_cycle_df, sample_cfg):
        result = engineer_features(sample_cycle_df, sample_cfg)
        assert len(result) == len(sample_cycle_df)

    def test_original_df_not_mutated(self, sample_cycle_df, sample_cfg):
        original_cols = set(sample_cycle_df.columns)
        _ = engineer_features(sample_cycle_df, sample_cfg)
        assert set(sample_cycle_df.columns) == original_cols

    def test_get_feature_columns_structure(self):
        feat_groups = get_feature_columns()
        assert "raw" in feat_groups
        assert "derived" in feat_groups
        assert "bayesian" in feat_groups
        assert "target" in feat_groups
        assert "soh" in feat_groups["target"]


# ─────────────────────────────────────────────────────────────────────────────
# Missing value report tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingValueReport:
    def test_no_missing_returns_empty(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report = report_missing_values(df)
        assert len(report) == 0

    def test_detects_missing(self):
        df = pd.DataFrame({"a": [1, np.nan, 3], "b": [4, 5, 6]})
        report = report_missing_values(df)
        assert "a" in report["feature"].values
        assert "b" not in report["feature"].values

    def test_percentage_correct(self):
        df = pd.DataFrame({"x": [1, 2, np.nan, np.nan]})
        report = report_missing_values(df)
        row = report[report["feature"] == "x"].iloc[0]
        assert row["pct_missing"] == pytest.approx(50.0, abs=0.01)
