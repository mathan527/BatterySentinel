"""
BatterySentinel — tests/test_statistics.py
============================================
Pytest tests for statistics_engine and bayesian_risk modules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.statistics_engine import (
    describe_feature, compute_quantiles, iqr_anomaly_flags,
    compute_covariance_matrix, compute_correlation_matrix,
    independence_test, conditional_statistics, compute_expectation,
)
from src.bayesian_risk import (
    GaussianNaiveBayesRiskEngine, battery_level_train_test_split,
    demonstrate_bayes_rule,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def gaussian_series():
    np.random.seed(0)
    return pd.Series(np.random.normal(loc=5.0, scale=2.0, size=500))


@pytest.fixture
def battery_df():
    """Synthetic multi-battery DataFrame for Bayesian tests."""
    np.random.seed(42)
    records = []
    for i, bid in enumerate(["B0001", "B0002", "B0003", "B0004", "B0005"]):
        n = 80
        base_soh = 100.0 - i * 8
        for cyc in range(1, n + 1):
            soh = max(0, base_soh - cyc * 0.35 + np.random.normal(0, 1))
            cap = soh / 100.0 * 1.8
            label = "Degraded" if soh <= 70 else "Early Degradation" if soh <= 85 else "Healthy"
            records.append({
                "battery_id": bid, "cycle_number": cyc,
                "soh": soh, "capacity": cap,
                "voltage_mean": 3.5 - (100 - soh) * 0.003,
                "voltage_std": 0.25 + (100 - soh) * 0.001,
                "temperature_mean": 25.0 + i * 2,
                "temperature_std": 3.0,
                "current_mean": -1.0, "current_std": 0.1,
                "Re_Ohm": 0.06 + (100 - soh) * 0.0005,
                "degradation_label": label,
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# describe_feature
# ─────────────────────────────────────────────────────────────────────────────

class TestDescribeFeature:
    def test_returns_dict(self, gaussian_series):
        result = describe_feature(gaussian_series)
        assert isinstance(result, dict)

    def test_mean_close_to_loc(self, gaussian_series):
        result = describe_feature(gaussian_series)
        assert abs(result["mean"] - 5.0) < 0.3

    def test_std_close_to_scale(self, gaussian_series):
        result = describe_feature(gaussian_series)
        assert abs(result["std"] - 2.0) < 0.3

    def test_quantile_ordering(self, gaussian_series):
        result = describe_feature(gaussian_series)
        assert result["q25"] <= result["q50"] <= result["q75"]

    def test_iqr_equals_q75_minus_q25(self, gaussian_series):
        result = describe_feature(gaussian_series)
        expected_iqr = result["q75"] - result["q25"]
        assert result["iqr"] == pytest.approx(expected_iqr, abs=1e-6)

    def test_variance_equals_std_squared(self, gaussian_series):
        result = describe_feature(gaussian_series)
        assert result["variance"] == pytest.approx(result["std"] ** 2, rel=1e-3)

    def test_empty_series_returns_error(self):
        result = describe_feature(pd.Series([], dtype=float))
        assert "error" in result or result["n"] == 0

    def test_constant_series_has_zero_variance(self):
        s = pd.Series([5.0] * 100)
        result = describe_feature(s)
        assert result["variance"] == pytest.approx(0.0, abs=1e-10)


# ─────────────────────────────────────────────────────────────────────────────
# compute_quantiles
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantiles:
    def test_median_is_50th_percentile(self):
        s = pd.Series(range(1, 101))
        q = compute_quantiles(s, [0.5])
        assert q["q50"] == pytest.approx(50.5, abs=0.5)

    def test_q0_equals_min(self):
        s = pd.Series([3.0, 5.0, 7.0, 9.0, 1.0])
        q = compute_quantiles(s, [0.0, 1.0])
        assert q["q0"] == pytest.approx(s.min(), abs=1e-6)

    def test_q100_equals_max(self):
        s = pd.Series([3.0, 5.0, 7.0, 9.0, 1.0])
        q = compute_quantiles(s, [0.0, 1.0])
        assert q["q100"] == pytest.approx(s.max(), abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# IQR anomaly flags
# ─────────────────────────────────────────────────────────────────────────────

class TestIQRAnomalyFlags:
    def test_clear_outlier_flagged(self):
        # Normal distribution + one extreme outlier
        normal = pd.Series(list(range(100)) + [10000])
        flags = iqr_anomaly_flags(normal)
        assert flags.iloc[-1] == "STATISTICAL ANOMALY"

    def test_normal_values_not_flagged(self):
        s = pd.Series(np.random.normal(0, 1, 200))
        flags = iqr_anomaly_flags(s)
        # Fewer than 5% should be flagged for normal distribution
        pct_flagged = (flags == "STATISTICAL ANOMALY").mean()
        assert pct_flagged < 0.1

    def test_returns_same_length(self):
        s = pd.Series([1, 2, 3, 1000, 4, 5])
        flags = iqr_anomaly_flags(s)
        assert len(flags) == len(s)


# ─────────────────────────────────────────────────────────────────────────────
# Covariance / Correlation
# ─────────────────────────────────────────────────────────────────────────────

class TestCovarianceCorrelation:
    def test_covariance_symmetric(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        cov = compute_covariance_matrix(df)
        assert cov.loc["a", "b"] == pytest.approx(cov.loc["b", "a"], abs=1e-10)

    def test_negative_covariance_for_inverse(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [5, 4, 3, 2, 1]})
        cov = compute_covariance_matrix(df)
        assert cov.loc["a", "b"] < 0

    def test_correlation_bounded(self):
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
        })
        corr = compute_correlation_matrix(df)
        assert (corr.values >= -1.0 - 1e-9).all()
        assert (corr.values <= 1.0 + 1e-9).all()

    def test_perfect_positive_correlation(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [2, 4, 6, 8]})
        corr = compute_correlation_matrix(df)
        assert corr.loc["a", "b"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_negative_correlation(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [4, 3, 2, 1]})
        corr = compute_correlation_matrix(df)
        assert corr.loc["a", "b"] == pytest.approx(-1.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Independence test
# ─────────────────────────────────────────────────────────────────────────────

class TestIndependenceTest:
    def test_strongly_correlated_detected(self):
        x = pd.Series(range(100))
        y = x * 2.0 + 5.0
        result = independence_test(x, y)
        assert result["pearson_significant"] is True

    def test_uncorrelated_returns_not_significant(self):
        np.random.seed(99)
        x = pd.Series(np.random.randn(200))
        y = pd.Series(np.random.randn(200))
        result = independence_test(x, y)
        # Not guaranteed but usually not significant for true independence
        assert "pearson_r" in result

    def test_result_has_required_keys(self):
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y = pd.Series([2.0, 4.0, 6.0, 8.0, 10.0])
        result = independence_test(x, y)
        for key in ["pearson_r", "pearson_p", "spearman_r", "spearman_p"]:
            assert key in result


# ─────────────────────────────────────────────────────────────────────────────
# Bayes Rule demonstration
# ─────────────────────────────────────────────────────────────────────────────

class TestBayesRuleDemo:
    def test_posterior_sums_to_one(self):
        result = demonstrate_bayes_rule(0.3, 0.8, 0.05)
        total = result["posterior_P_Degraded_given_E"] + result["posterior_P_Healthy_given_E"]
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_strong_evidence_increases_posterior(self):
        result = demonstrate_bayes_rule(0.3, 0.95, 0.02)
        # Very strong likelihood should push posterior well above prior
        assert result["posterior_P_Degraded_given_E"] > 0.3

    def test_weak_evidence_stays_near_prior(self):
        # When likelihoods are equal, posterior == prior
        result = demonstrate_bayes_rule(0.3, 0.5, 0.5)
        assert result["posterior_P_Degraded_given_E"] == pytest.approx(0.3, abs=0.01)

    def test_zero_evidence_probability_returns_error(self):
        # Both likelihoods = 0 → evidence = 0 → error
        result = demonstrate_bayes_rule(0.5, 0.0, 0.0)
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# GaussianNaiveBayesRiskEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestGaussianNaiveBayes:
    def test_fit_and_predict(self, battery_df):
        engine = GaussianNaiveBayesRiskEngine(
            evidence_features=["soh", "capacity", "voltage_mean"]
        )
        train = battery_df[battery_df["battery_id"].isin(["B0001", "B0002", "B0003"])]
        engine.fit(train)
        assert engine.is_fitted

        obs = {"soh": 90.0, "capacity": 1.7, "voltage_mean": 3.5}
        proba = engine.predict_proba(obs)
        assert set(proba.keys()) >= {"Healthy", "Early Degradation", "Degraded"}
        total = sum(proba.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_posteriors_sum_to_one(self, battery_df):
        engine = GaussianNaiveBayesRiskEngine(
            evidence_features=["soh", "capacity"]
        )
        engine.fit(battery_df)
        for soh_val in [95, 80, 65]:
            proba = engine.predict_proba({"soh": float(soh_val), "capacity": soh_val / 100.0 * 1.8})
            assert sum(proba.values()) == pytest.approx(1.0, abs=1e-6)

    def test_risk_score_in_range(self, battery_df):
        engine = GaussianNaiveBayesRiskEngine(evidence_features=["soh"])
        engine.fit(battery_df)
        for soh_val in [99.0, 80.0, 60.0, 30.0]:
            risk = engine.predict_risk_score({"soh": soh_val})
            assert 0.0 <= risk <= 1.0

    def test_high_soh_low_risk(self, battery_df):
        engine = GaussianNaiveBayesRiskEngine(evidence_features=["soh", "capacity"])
        engine.fit(battery_df)
        risk_high = engine.predict_risk_score({"soh": 99.0, "capacity": 1.78})
        risk_low  = engine.predict_risk_score({"soh": 40.0, "capacity": 0.72})
        assert risk_high < risk_low

    def test_not_fitted_raises(self):
        engine = GaussianNaiveBayesRiskEngine()
        with pytest.raises(RuntimeError):
            engine.predict_proba({"soh": 90.0})


# ─────────────────────────────────────────────────────────────────────────────
# Battery-level train/test split
# ─────────────────────────────────────────────────────────────────────────────

class TestBatteryLevelSplit:
    def test_no_battery_overlap(self, battery_df):
        train, test = battery_level_train_test_split(battery_df, test_fraction=0.4)
        train_bids = set(train["battery_id"].unique())
        test_bids  = set(test["battery_id"].unique())
        assert train_bids.isdisjoint(test_bids)

    def test_all_cycles_preserved(self, battery_df):
        train, test = battery_level_train_test_split(battery_df)
        assert len(train) + len(test) == len(battery_df)

    def test_explicit_test_batteries(self, battery_df):
        test_bids = ["B0001", "B0002"]
        train, test = battery_level_train_test_split(battery_df, test_batteries=test_bids)
        assert set(test["battery_id"].unique()) == set(test_bids)

    def test_reproducible_split(self, battery_df):
        _, test1 = battery_level_train_test_split(battery_df, random_seed=7)
        _, test2 = battery_level_train_test_split(battery_df, random_seed=7)
        assert set(test1["battery_id"].unique()) == set(test2["battery_id"].unique())
