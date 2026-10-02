"""
BatterySentinel — src/bayesian_risk.py
========================================
Bayesian early-failure prediction engine.

Unit 1 Concepts Covered
------------------------
  • Bayes' Rule:             P(D|E) = P(E|D) * P(D) / P(E)
  • Prior Probability:       P(D)   = base rate of degradation
  • Likelihood:              P(E|D) = P(evidence | class)  — Gaussian model
  • Evidence (normaliser):   P(E)   = sum over classes of P(E|class)*P(class)
  • Posterior Probability:   P(D|E) = updated belief after observing evidence
  • Conditional Independence: P(E1,E2,...|D) = prod P(Ei|D)  (Naive Bayes)
  • Independence assumption documented explicitly with caveats

Core Question Answered
-----------------------
    Given a battery with observed features E = {voltage, temp, Re, ...},
    what is the probability it is in each degradation state?

    P(Degraded  | E) = P(E | Degraded) * P(Degraded) / P(E)
    P(Healthy   | E) = P(E | Healthy)  * P(Healthy)  / P(E)

Architecture
------------
  GaussianNaiveBayesRiskEngine:
    - Fit Gaussian likelihood P(Ei | class) for each evidence feature
    - Compute prior P(class) from training data
    - Predict posterior P(class | evidence) via Bayes' rule
    - Output risk score: P(Degraded | evidence) in [0, 1]
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CLASSES = ["Healthy", "Early Degradation", "Degraded"]
CLASS_RISK = {"Healthy": 0.0, "Early Degradation": 0.5, "Degraded": 1.0}
# Smooth prior to avoid zero-probability classes
LAPLACE_ALPHA = 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Gaussian Naive Bayes Risk Engine
# ─────────────────────────────────────────────────────────────────────────────

class GaussianNaiveBayesRiskEngine:
    """
    Bayesian degradation risk estimator using Gaussian Naive Bayes.

    Model
    -----
    Given degradation class C and evidence features E = [E1, E2, ..., Ek]:

    1. PRIOR:
       P(C) = (n_class + alpha) / (N + K * alpha)   (Laplace-smoothed)
       where N = total training samples, K = number of classes.

    2. LIKELIHOOD (Gaussian):
       P(Ei | C) = N(Ei; mu_{Ci}, sigma^2_{Ci})
                = (1 / (sigma*sqrt(2*pi))) * exp(-(Ei - mu)^2 / (2*sigma^2))
       mu_{Ci}    = mean of feature Ei in class C
       sigma_{Ci} = std  of feature Ei in class C

    3. CONDITIONAL INDEPENDENCE ASSUMPTION (Naive Bayes):
       P(E | C) = product_{i=1}^{k} P(Ei | C)
       This assumes each feature is conditionally independent given the class.
       NOTE: In reality, features like voltage and current are correlated.
       The Naive Bayes model trades accuracy for tractability and
       interpretability.

    4. JOINT (unnormalised posterior):
       P(C, E) = P(C) * product_i P(Ei | C)

    5. POSTERIOR (Bayes' Rule):
       P(C | E) = P(C, E) / P(E)
       P(E) = sum_C' P(C') * product_i P(Ei | C')   (normalisation constant)

    6. RISK SCORE:
       risk(E) = P(Degraded | E) + 0.5 * P(Early Degradation | E)
       Maps posterior to a scalar risk in [0, 1].

    Attributes
    ----------
    class_params : dict — fitted Gaussian parameters per class and feature
    class_priors : dict — P(class)
    evidence_features : list — feature names used
    is_fitted : bool
    """

    def __init__(self, evidence_features: list[str] | None = None):
        self.evidence_features: list[str] = evidence_features or [
            "capacity", "soh", "voltage_mean", "voltage_std",
            "temperature_mean", "temperature_std",
            "Re_Ohm", "Rct_Ohm", "current_mean",
        ]
        self.class_params: dict[str, dict[str, dict[str, float]]] = {}
        self.class_priors: dict[str, float] = {}
        self.is_fitted: bool = False
        self.training_summary: dict[str, Any] = {}

    # ── Fitting ──────────────────────────────────────────────────────────────

    def fit(
        self,
        train_df: pd.DataFrame,
        label_col: str = "degradation_label",
    ) -> "GaussianNaiveBayesRiskEngine":
        """
        Fit Gaussian parameters (mu, sigma) per class per feature.

        Steps:
          1. Compute class priors from training frequency.
          2. For each class, compute mean and std of each evidence feature.
          3. Store these as the Gaussian likelihood parameters.

        Parameters
        ----------
        train_df  : pd.DataFrame — training cycles (battery-level split)
        label_col : str — column containing class labels

        Returns self (for method chaining).
        """
        log.info("  Fitting Bayesian Risk Engine ...")
        n_total = len(train_df)
        k = len(CLASSES)

        # ── Class priors ────────────────────────────────────────────────────
        class_counts = {c: 0 for c in CLASSES}
        for cls in train_df[label_col].dropna():
            if cls in class_counts:
                class_counts[cls] += 1

        # Laplace smoothing
        self.class_priors = {
            c: (class_counts[c] + LAPLACE_ALPHA) / (n_total + k * LAPLACE_ALPHA)
            for c in CLASSES
        }
        log.info(f"  Class priors (Laplace-smoothed alpha={LAPLACE_ALPHA}):")
        for cls, p in self.class_priors.items():
            log.info(f"    P({cls}) = {p:.4f}")

        # ── Gaussian likelihood parameters ──────────────────────────────────
        self.class_params = {}
        for cls in CLASSES:
            subset = train_df[train_df[label_col] == cls]
            self.class_params[cls] = {}
            for feat in self.evidence_features:
                if feat not in train_df.columns:
                    continue
                data = subset[feat].dropna()
                if len(data) < 2:
                    mu_val, sigma_val = 0.0, 1.0
                    log.warning(f"  Feature {feat} in class {cls}: < 2 samples — using defaults.")
                else:
                    mu_val = float(data.mean())
                    sigma_val = max(float(data.std(ddof=1)), 1e-6)  # avoid sigma=0
                self.class_params[cls][feat] = {"mu": mu_val, "sigma": sigma_val}
            log.info(f"  Fitted Gaussian params for {cls} ({len(subset)} samples)")

        self.is_fitted = True
        self.training_summary = {
            "n_training": int(n_total),
            "class_counts": class_counts,
            "class_priors": self.class_priors,
            "evidence_features": self.evidence_features,
            "class_params": {
                cls: {
                    feat: {k: round(v, 8) for k, v in params.items()}
                    for feat, params in feat_params.items()
                }
                for cls, feat_params in self.class_params.items()
            },
        }
        return self

    # ── Gaussian log-likelihood ───────────────────────────────────────────────

    def _log_likelihood(
        self,
        value: float,
        mu: float,
        sigma: float,
    ) -> float:
        """
        Compute log P(value; mu, sigma^2) under a Gaussian distribution.

        Using log-space avoids numerical underflow when multiplying many
        small probabilities together.

        log P(x; mu, sigma) = -0.5 * log(2*pi*sigma^2) - (x-mu)^2 / (2*sigma^2)
        """
        return float(stats.norm.logpdf(value, loc=mu, scale=sigma))

    # ── Predict ──────────────────────────────────────────────────────────────

    def predict_proba(
        self,
        observation: dict[str, float],
    ) -> dict[str, float]:
        """
        Compute P(class | evidence) for one observation.

        Algorithm (Bayes' Rule):
          1. log_prior = log P(class)
          2. For each available evidence feature Ei:
             log_likelihood += log P(Ei | class)    (Naive Bayes: sum of logs)
          3. log_joint[class] = log_prior + sum of log_likelihoods
          4. Normalise:
             P(class | E) = exp(log_joint[class]) / sum_class' exp(log_joint[class'])
             (Use log-sum-exp trick for numerical stability)

        Parameters
        ----------
        observation : dict mapping feature_name -> observed_value

        Returns
        -------
        dict : {class_name: posterior_probability}
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")

        log_joints: dict[str, float] = {}

        for cls in CLASSES:
            # Start with log prior
            log_joint = np.log(self.class_priors[cls])

            # Add log likelihoods for each available evidence feature
            n_features_used = 0
            for feat, value in observation.items():
                if feat not in self.class_params.get(cls, {}):
                    continue
                if np.isnan(value):
                    continue  # skip missing evidence
                params = self.class_params[cls][feat]
                log_joint += self._log_likelihood(value, params["mu"], params["sigma"])
                n_features_used += 1

            log_joints[cls] = log_joint

        # ── Log-sum-exp normalisation ─────────────────────────────────────────
        # log_evidence = log sum_c exp(log_joint[c])
        log_values = np.array([log_joints[c] for c in CLASSES])
        log_sum = np.logaddexp.reduce(log_values)

        posteriors: dict[str, float] = {}
        for cls, lv in zip(CLASSES, log_values):
            posteriors[cls] = float(np.exp(lv - log_sum))

        return posteriors

    def predict_risk_score(self, observation: dict[str, float]) -> float:
        """
        Compute a scalar risk score in [0, 1].

        Risk = P(Degraded | E) + 0.5 * P(Early Degradation | E)

        Interpretation:
            0.0 — very low risk (strongly Healthy)
            0.5 — uncertain / borderline
            1.0 — very high risk (strongly Degraded)

        IMPORTANT: This is a statistical model output, NOT a guaranteed
        failure probability. It should be used as a decision-support
        indicator, not as a safety certification.
        """
        posteriors = self.predict_proba(observation)
        risk = posteriors["Degraded"] + 0.5 * posteriors.get("Early Degradation", 0.0)
        return min(1.0, float(risk))

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict posterior probabilities and risk scores for all rows.

        Parameters
        ----------
        df : pd.DataFrame — cycle-level dataset

        Returns
        -------
        pd.DataFrame with columns:
            P_Healthy, P_EarlyDegradation, P_Degraded, risk_score, predicted_class
        """
        records: list[dict] = []
        evidence_feats = [f for f in self.evidence_features if f in df.columns]

        for _, row in df.iterrows():
            obs = {f: float(row[f]) for f in evidence_feats if not np.isnan(row[f])
                   if f in row.index}
            posteriors = self.predict_proba(obs)
            risk = self.predict_risk_score(obs)
            predicted = max(posteriors, key=posteriors.get)  # type: ignore[arg-type]
            records.append({
                "P_Healthy": posteriors["Healthy"],
                "P_Early_Degradation": posteriors["Early Degradation"],
                "P_Degraded": posteriors["Degraded"],
                "risk_score": risk,
                "predicted_class": predicted,
            })

        return pd.DataFrame(records, index=df.index)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def save(self, path: Path) -> None:
        """Save fitted model parameters to JSON."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted.")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.training_summary, fh, indent=2, default=str)
        log.info(f"  Bayesian model saved: {path}")

    def load(self, path: Path) -> None:
        """Load fitted model parameters from JSON."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        self.class_priors = data["class_priors"]
        self.class_params = data["class_params"]
        self.evidence_features = data["evidence_features"]
        self.training_summary = data
        self.is_fitted = True
        log.info(f"  Bayesian model loaded: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Battery-level train/test split (CRITICAL — no data leakage)
# ─────────────────────────────────────────────────────────────────────────────

def battery_level_train_test_split(
    df: pd.DataFrame,
    test_batteries: list[str] | None = None,
    test_fraction: float = 0.3,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset at the BATTERY level — not cycle level.

    CRITICAL DESIGN DECISION:
        Cycles from the same battery are NOT independent observations.
        A battery's cycle N is temporally correlated with cycle N-1.
        Randomly splitting cycles from the SAME battery between train and
        test would be data leakage — the model would "see" future cycles of
        the test battery during training.

        This function ensures that ALL cycles of a battery appear in
        EXACTLY ONE of train or test.

    Parameters
    ----------
    df             : cycle-level DataFrame
    test_batteries : explicit list of battery IDs for test set (None = random)
    test_fraction  : fraction of batteries to use for testing (if auto-select)
    random_seed    : reproducibility seed

    Returns
    -------
    (train_df, test_df)
    """
    batteries = sorted(df["battery_id"].unique().tolist())

    if test_batteries is not None:
        test_bids = test_batteries
    else:
        rng = np.random.default_rng(random_seed)
        n_test = max(1, int(len(batteries) * test_fraction))
        test_bids = rng.choice(batteries, size=n_test, replace=False).tolist()

    train_bids = [b for b in batteries if b not in test_bids]

    train_df = df[df["battery_id"].isin(train_bids)].copy()
    test_df = df[df["battery_id"].isin(test_bids)].copy()

    log.info(f"  Train: {len(train_bids)} batteries, {len(train_df)} cycles")
    log.info(f"  Test : {len(test_bids)} batteries, {len(test_df)} cycles")
    log.info(f"  Test batteries: {sorted(test_bids)}")

    return train_df, test_df


# ─────────────────────────────────────────────────────────────────────────────
# Bayes' Rule demonstration (standalone utility)
# ─────────────────────────────────────────────────────────────────────────────

def demonstrate_bayes_rule(
    prior_degraded: float,
    likelihood_low_capacity_given_degraded: float,
    likelihood_low_capacity_given_healthy: float,
    evidence_description: str = "low capacity",
) -> dict[str, float]:
    """
    Explicit step-by-step Bayes' Rule demonstration.

    This function shows the manual calculation of Bayes' Rule for
    a binary classification scenario.

    Formula:
        P(Degraded | Evidence) = P(Evidence | Degraded) * P(Degraded)
                                 -----------------------------------------
                                 P(Evidence)

        where:
        P(Evidence) = P(E | Degraded) * P(Degraded)
                    + P(E | Healthy) * P(Healthy)

    Example:
        Prior:     P(Degraded) = 0.30  (30% of cycles are degraded)
        Likelihood P(low_cap | Degraded) = 0.80  (80% of degraded have low capacity)
        Likelihood P(low_cap | Healthy)  = 0.05  (5% of healthy have low capacity)

        => Posterior P(Degraded | low_cap) = 0.80*0.30 / (0.80*0.30 + 0.05*0.70)
                                           = 0.240 / 0.275
                                           = 0.873  (87.3% probability of degradation)

    Parameters
    ----------
    prior_degraded                          : P(Degraded)
    likelihood_low_capacity_given_degraded  : P(Evidence | Degraded)
    likelihood_low_capacity_given_healthy   : P(Evidence | Healthy)
    evidence_description                    : string label for the evidence

    Returns
    -------
    dict with all intermediate values and the final posterior.
    """
    prior_healthy = 1.0 - prior_degraded

    # Numerators
    num_degraded = likelihood_low_capacity_given_degraded * prior_degraded
    num_healthy = likelihood_low_capacity_given_healthy * prior_healthy

    # Normalising constant (total probability of evidence)
    evidence = num_degraded + num_healthy

    if evidence == 0:
        return {"error": "Total probability of evidence is 0 — invalid inputs."}

    posterior_degraded = num_degraded / evidence
    posterior_healthy = num_healthy / evidence

    return {
        "evidence_description": evidence_description,
        "prior_P_Degraded": round(prior_degraded, 4),
        "prior_P_Healthy": round(prior_healthy, 4),
        "likelihood_E_given_Degraded": round(likelihood_low_capacity_given_degraded, 4),
        "likelihood_E_given_Healthy": round(likelihood_low_capacity_given_healthy, 4),
        "joint_Degraded": round(num_degraded, 6),
        "joint_Healthy": round(num_healthy, 6),
        "P_Evidence": round(evidence, 6),
        "posterior_P_Degraded_given_E": round(posterior_degraded, 4),
        "posterior_P_Healthy_given_E": round(posterior_healthy, 4),
        "bayes_update_ratio": round(posterior_degraded / prior_degraded, 4),
        "interpretation": (
            f"Observing '{evidence_description}' updates belief from "
            f"P(Degraded)={prior_degraded:.2%} to "
            f"P(Degraded|E)={posterior_degraded:.2%}. "
            f"Evidence multiplied prior by {posterior_degraded/prior_degraded:.2f}x."
        ),
    }
