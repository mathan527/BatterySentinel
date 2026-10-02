"""
BatterySentinel — app.py
=========================
10-page Streamlit dashboard for the BatterySentinel project.

Pages
-----
 1  Home / Overview
 2  Dataset Explorer
 3  Statistical Analysis
 4  Probability Density
 5  Covariance & Correlation
 6  Bayesian Risk Engine
 7  Polynomial Degradation Model
 8  K-Means Clustering
 9  Independence Testing
10  Model Evaluation & Limitations

Run:
    streamlit run app.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import gaussian_kde

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_config
from src.statistics_engine import (
    describe_feature, describe_all_features,
    compute_covariance_matrix, compute_correlation_matrix,
    independence_test, conditional_statistics, iqr_anomaly_flags,
)
from src.bayesian_risk import (
    GaussianNaiveBayesRiskEngine, battery_level_train_test_split,
    demonstrate_bayes_rule, CLASSES,
)
from src.polynomial_model import (
    fit_polynomial, compare_polynomial_degrees, predict_eol_cycle,
)
from src.clustering import (
    prepare_cluster_features, elbow_analysis, fit_kmeans,
    interpret_clusters, assign_cluster_names,
)

# ── App configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BatterySentinel",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

LABEL_COLORS = {
    "Healthy": "#2ecc71",
    "Early Degradation": "#f39c12",
    "Degraded": "#e74c3c",
    "Unknown": "#95a5a6",
}


# ── Cached data loaders ───────────────────────────────────────────────────────

@st.cache_data
def load_cycle_data() -> pd.DataFrame:
    cfg = load_config()
    path = PROJECT_ROOT / cfg["data"]["processed_dir"] / "cycle_data.csv"
    if not path.exists():
        st.error(f"cycle_data.csv not found. Run: `python scripts/preprocess_dataset.py`")
        st.stop()
    return pd.read_csv(path)


@st.cache_data
def load_cfg() -> dict:
    return load_config()


@st.cache_data
def train_bayesian_model(_df: pd.DataFrame, _cfg: dict) -> GaussianNaiveBayesRiskEngine:
    evidence_feats = [f for f in _cfg.get("bayesian", {}).get("evidence_features", [
        "capacity", "soh", "voltage_mean", "voltage_std",
        "temperature_mean", "temperature_std", "Re_Ohm",
    ]) if f in _df.columns]
    test_bids = _cfg.get("training", {}).get("test_batteries", [])
    train_df, _ = battery_level_train_test_split(_df, test_batteries=test_bids if test_bids else None)
    engine = GaussianNaiveBayesRiskEngine(evidence_features=evidence_feats)
    engine.fit(train_df.dropna(subset=["degradation_label"]))
    return engine


# ── Sidebar navigation ────────────────────────────────────────────────────────

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Battery_icon.svg/200px-Battery_icon.svg.png", width=60)
st.sidebar.title("🔋 BatterySentinel")
st.sidebar.caption("Probabilistic Early-Failure Prediction\nfor Lithium-Ion Batteries")
st.sidebar.divider()

PAGES = [
    "🏠 Home & Overview",
    "📊 Dataset Explorer",
    "📈 Statistical Analysis",
    "〰️ Probability Density",
    "🔗 Covariance & Correlation",
    "🧠 Bayesian Risk Engine",
    "📉 Polynomial Degradation Model",
    "🔵 K-Means Clustering",
    "⚖️ Independence Testing",
    "✅ Model Evaluation & Limitations",
]

page = st.sidebar.radio("Navigate", PAGES)

cfg  = load_cfg()
df   = load_cycle_data()
eol_pct = cfg["soh"]["eol_threshold"] * 100


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════

if page == PAGES[0]:
    st.title("🔋 BatterySentinel")
    st.subheader("Probabilistic Early-Failure Prediction for Lithium-Ion Batteries")

    st.info("""
    **Academic ML project demonstrating Unit 1 concepts:**
    Machine Learning · Supervised & Unsupervised Learning · Polynomial Curve Fitting ·
    Probability Theory · Bayesian Inference · Random Variables · Mean · Variance ·
    Covariance · Independence · Probability Density · Quantiles · Expectation
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Batteries", df["battery_id"].nunique())
    col2.metric("Discharge Cycles", len(df))
    col3.metric("Features", df.shape[1])
    n_eol = int(df["eol_flag"].sum())
    col4.metric("EOL Cycles", n_eol, f"{n_eol/len(df)*100:.1f}%")

    st.divider()
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("SOH Trajectories — All 34 Batteries")
        fig = go.Figure()
        for bid in sorted(df["battery_id"].unique()):
            bdf = df[df["battery_id"] == bid].sort_values("cycle_number")
            soh = bdf["soh"].dropna()
            cyc = bdf.loc[soh.index, "cycle_number"]
            fig.add_trace(go.Scatter(
                x=cyc, y=soh, mode="lines", name=bid,
                line=dict(width=1.2), opacity=0.7,
            ))
        fig.add_hline(y=eol_pct, line_dash="dash", line_color="red",
                      annotation_text=f"EOL {eol_pct:.0f}% (configurable)")
        fig.update_layout(
            xaxis_title="Discharge Cycle Number",
            yaxis_title="State of Health SOH (%)",
            height=420, showlegend=False,
            title="SOH(t) = C(t) / C_max × 100%",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Degradation Label Distribution")
        label_counts = df["degradation_label"].value_counts()
        fig_pie = px.pie(
            values=label_counts.values,
            names=label_counts.index,
            color=label_counts.index,
            color_discrete_map=LABEL_COLORS,
            hole=0.45,
        )
        fig_pie.update_layout(height=280)
        st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("Unit 1 Concept Map")
        concept_map = {
            "Mean / Variance": "E[SOH], Var(SOH) across cycles",
            "Probability Density": "PDF of capacity per class",
            "Quantiles / IQR": "Q25, Q50, Q75 of SOH",
            "Covariance": "Cov(cycle, SOH)",
            "Bayes' Rule": "P(Degraded | Evidence)",
            "Polynomial Fit": "SOH(n) ~ poly degree 1-4",
            "K-Means": "Unsupervised cycle clustering",
        }
        for concept, where in concept_map.items():
            st.markdown(f"**{concept}** → _{where}_")

    st.divider()
    st.warning("""
    ⚠️ **Limitations & Research Disclaimer**
    - EOL threshold (70% SOH) is a **configurable research parameter** — not a universal safety standard.
    - Risk scores are statistical model outputs — NOT guaranteed failure probabilities.
    - This project is for **educational / research purposes only**.
    - Do not use for safety-critical decisions without physical verification.
    """)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[1]:
    st.title("📊 Dataset Explorer")
    st.caption("NASA Lithium-Ion Battery Aging Dataset — patrickfleith/nasa-battery-dataset (Kaggle)")

    col1, col2 = st.columns([1, 3])
    with col1:
        selected_battery = st.selectbox("Select Battery", sorted(df["battery_id"].unique()))
    with col2:
        feature = st.selectbox("Plot Feature", ["soh", "capacity", "voltage_mean",
                                                "temperature_mean", "Re_Ohm", "current_mean"])

    bdf = df[df["battery_id"] == selected_battery].sort_values("cycle_number")

    # SOH + selected feature twin-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    soh_s = bdf["soh"].dropna()
    cyc_s = bdf.loc[soh_s.index, "cycle_number"]

    fig.add_trace(go.Scatter(x=cyc_s, y=soh_s, name="SOH (%)",
                             line=dict(color="#3498db", width=2)), secondary_y=False)
    feat_s = bdf[feature].dropna()
    cyc_f = bdf.loc[feat_s.index, "cycle_number"]
    fig.add_trace(go.Scatter(x=cyc_f, y=feat_s, name=feature,
                             line=dict(color="#e67e22", width=2, dash="dot")), secondary_y=True)
    fig.add_hline(y=eol_pct, line_dash="dash", line_color="red")
    fig.update_layout(title=f"Battery {selected_battery} — SOH & {feature} vs Cycle",
                      height=380)
    fig.update_yaxes(title_text="SOH (%)", secondary_y=False)
    fig.update_yaxes(title_text=feature, secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Battery Summary Statistics")
    bstats = describe_feature(bdf["soh"], "SOH")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Cycles", int(bstats.get("n", 0)))
    c2.metric("Mean SOH", f"{bstats.get('mean', 0):.1f}%")
    c3.metric("Std SOH", f"{bstats.get('std', 0):.2f}%")
    c4.metric("Min SOH", f"{bstats.get('min', 0):.1f}%")
    c5.metric("At EOL?", "Yes ✅" if bdf["eol_flag"].any() else "No ❌")

    st.subheader("Raw Cycle Data")
    st.dataframe(bdf[[c for c in ["cycle_number","soh","capacity","voltage_mean",
                                   "temperature_mean","Re_Ohm","degradation_label",
                                   "data_quality_flag"] if c in bdf.columns]],
                 use_container_width=True, height=280)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[2]:
    st.title("📈 Statistical Analysis")
    st.markdown("""
    ### Unit 1: Mean · Variance · Standard Deviation · Quantiles · Skewness · Kurtosis

    For a continuous random variable X with n observations:

    $$E[X] = \\bar{x} = \\frac{1}{n} \\sum_{i=1}^{n} x_i$$

    $$\\text{Var}(X) = \\frac{1}{n-1} \\sum_{i=1}^{n} (x_i - \\bar{x})^2$$

    $$Q_p = F^{-1}(p) \\quad \\text{(quantile at level } p)$$
    """)

    st.divider()
    key_features = [f for f in ["soh", "capacity", "voltage_mean", "voltage_std",
                                 "temperature_mean", "Re_Ohm", "discharge_duration",
                                 "cycle_number"] if f in df.columns]
    stats_df = describe_all_features(df, key_features)
    display_cols = ["n", "mean", "std", "variance", "min", "q25", "q50", "q75",
                    "max", "iqr", "skewness", "kurtosis_excess", "n_outliers"]
    display_cols = [c for c in display_cols if c in stats_df.columns]
    st.subheader("Descriptive Statistics Table")
    st.dataframe(stats_df[display_cols].round(4), use_container_width=True)

    st.divider()
    st.subheader("SOH Summary — Fleet-wide")
    soh = df["soh"].dropna()
    cc1, cc2, cc3, cc4, cc5, cc6 = st.columns(6)
    cc1.metric("E[SOH]", f"{soh.mean():.2f}%")
    cc2.metric("Var(SOH)", f"{soh.var():.1f}")
    cc3.metric("std(SOH)", f"{soh.std():.2f}%")
    cc4.metric("Q25", f"{soh.quantile(0.25):.2f}%")
    cc5.metric("Median", f"{soh.quantile(0.50):.2f}%")
    cc6.metric("IQR", f"{(soh.quantile(0.75)-soh.quantile(0.25)):.2f}%")

    st.divider()
    st.subheader("IQR Anomaly Detection")
    feat_sel = st.selectbox("Feature for anomaly analysis",
                             ["soh", "voltage_mean", "temperature_mean", "Re_Ohm"])
    iqr_mult = st.slider("IQR multiplier (Tukey k)", 1.0, 3.0, 1.5, 0.1)
    flags = iqr_anomaly_flags(df[feat_sel], iqr_multiplier=iqr_mult)
    n_anom = (flags == "STATISTICAL ANOMALY").sum()
    st.metric("Statistical Anomalies", f"{n_anom} ({n_anom/len(df)*100:.1f}%)")

    q1 = df[feat_sel].quantile(0.25)
    q3 = df[feat_sel].quantile(0.75)
    iqr_val = q3 - q1
    st.info(f"Q1={q1:.4f} | Q3={q3:.4f} | IQR={iqr_val:.4f} | "
            f"Lower fence={q1-iqr_mult*iqr_val:.4f} | Upper fence={q3+iqr_mult*iqr_val:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PROBABILITY DENSITY
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[3]:
    st.title("〰️ Probability Density")
    st.markdown("""
    ### Unit 1: Continuous Random Variables · PDF · CDF · Expectation

    For a continuous random variable X, the PDF f(x) satisfies:
    $$P(a \\leq X \\leq b) = \\int_a^b f(x)\\, dx$$

    **KDE Estimate:**
    $$\\hat{f}(x) = \\frac{1}{nh} \\sum_{i=1}^{n} K\\!\\left(\\frac{x - x_i}{h}\\right)$$
    where K is the Gaussian kernel.
    """)

    feat = st.selectbox("Select feature", ["soh", "capacity", "voltage_mean",
                                            "temperature_mean", "Re_Ohm", "voltage_std"])
    show_classes = st.checkbox("Show per-class PDF (Conditional distributions)", value=True)

    fig = go.Figure()
    if show_classes:
        for cls in ["Healthy", "Early Degradation", "Degraded"]:
            data = df[df["degradation_label"] == cls][feat].dropna()
            if len(data) < 5:
                continue
            x_grid = np.linspace(data.min() - 0.1*data.std(), data.max() + 0.1*data.std(), 300)
            kde = gaussian_kde(data, bw_method="scott")
            fig.add_trace(go.Scatter(
                x=x_grid, y=kde(x_grid),
                name=f"P({feat} | {cls})",
                fill="tozeroy", fillcolor=LABEL_COLORS.get(cls, "#aaa"),
                line=dict(color=LABEL_COLORS.get(cls, "#aaa"), width=2),
                opacity=0.4,
            ))
    else:
        data = df[feat].dropna()
        x_grid = np.linspace(data.min(), data.max(), 300)
        kde = gaussian_kde(data, bw_method="scott")
        fig.add_trace(go.Scatter(x=x_grid, y=kde(x_grid),
                                  name=f"P({feat})", fill="tozeroy"))
        fig.add_vline(x=data.mean(), line_dash="dash", line_color="red",
                      annotation_text=f"E[{feat}]={data.mean():.3f}")

    fig.update_layout(
        title=f"Probability Density Function — {feat}",
        xaxis_title=feat, yaxis_title="Density f(x)",
        height=420,
    )
    st.plotly_chart(fig, use_container_width=True)

    if show_classes:
        st.subheader("Conditional Statistics: E[X | class]")
        rows = []
        for cls in ["Healthy", "Early Degradation", "Degraded"]:
            data = df[df["degradation_label"] == cls][feat].dropna()
            if len(data) > 0:
                rows.append({"Class": cls, "n": len(data),
                              "E[X]": round(float(data.mean()), 4),
                              "Var(X)": round(float(data.var()), 6),
                              "std(X)": round(float(data.std()), 4),
                              "Median": round(float(data.median()), 4)})
        if rows:
            st.table(pd.DataFrame(rows).set_index("Class"))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — COVARIANCE & CORRELATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[4]:
    st.title("🔗 Covariance & Correlation")
    st.markdown("""
    ### Unit 1: Covariance · Correlation · Joint Variation

    $$\\text{Cov}(X,Y) = E[(X - E[X])(Y - E[Y])]$$

    $$\\text{Corr}(X,Y) = \\frac{\\text{Cov}(X,Y)}{\\text{std}(X) \\cdot \\text{std}(Y)} \\in [-1, +1]$$

    **Key distinction:** Cov captures magnitude + direction; Corr is dimensionless.
    """)

    cov_features = [f for f in ["cycle_number","capacity","soh","voltage_mean",
                                  "voltage_std","temperature_mean","Re_Ohm","Rct_Ohm",
                                  "discharge_duration"] if f in df.columns]
    method = st.radio("Correlation method", ["pearson", "spearman"], horizontal=True)

    corr = compute_correlation_matrix(df, cov_features, method=method)
    fig_corr = px.imshow(corr, text_auto=".3f", color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, aspect="auto",
                          title=f"{method.capitalize()} Correlation Matrix")
    fig_corr.update_layout(height=480)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Key Correlation Findings")
    pairs = [("cycle_number", "soh"), ("cycle_number", "capacity"),
             ("Re_Ohm", "soh"), ("temperature_mean", "capacity"),
             ("voltage_std", "soh")]
    rows = []
    for a, b in pairs:
        if a in corr.columns and b in corr.columns:
            r = corr.loc[a, b]
            interp = "negative (degradation signal)" if r < -0.1 else \
                     "positive (confound — likely temperature grouping)" if r > 0.1 else "weak"
            rows.append({"Pair": f"Corr({a}, {b})", "r": round(r, 4), "Interpretation": interp})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.caption("Note: Correlation ≠ Causation. Observed correlations may reflect "
               "confounding variables such as ambient temperature groups.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — BAYESIAN RISK ENGINE
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[5]:
    st.title("🧠 Bayesian Risk Engine")
    st.markdown("""
    ### Unit 1: Bayes' Rule · Prior · Likelihood · Posterior · Conditional Independence

    $$P(\\text{Degraded} \\mid E) = \\frac{P(E \\mid \\text{Degraded}) \\cdot P(\\text{Degraded})}{P(E)}$$

    **Naive Bayes independence assumption:**
    $$P(E_1, E_2, \\ldots, E_k \\mid C) = \\prod_{i=1}^{k} P(E_i \\mid C)$$
    """)

    st.subheader("Step-by-Step Bayes' Rule Demo")
    col1, col2, col3 = st.columns(3)
    prior_deg = col1.slider("Prior P(Degraded)", 0.01, 0.99, float(
        (df["degradation_label"] == "Degraded").sum() / df["degradation_label"].notna().sum()
    ), 0.01)
    lik_deg = col2.slider("P(Evidence | Degraded)", 0.01, 0.99, 0.70, 0.01)
    lik_hlt = col3.slider("P(Evidence | Healthy)", 0.01, 0.99, 0.10, 0.01)

    demo = demonstrate_bayes_rule(prior_deg, lik_deg, lik_hlt)
    if "error" not in demo:
        dc1, dc2, dc3 = st.columns(3)
        dc1.metric("Prior P(Degraded)", f"{demo['prior_P_Degraded']:.3f}")
        dc2.metric("Posterior P(Degraded|E)", f"{demo['posterior_P_Degraded_given_E']:.3f}",
                   delta=f"+{demo['posterior_P_Degraded_given_E'] - demo['prior_P_Degraded']:.3f}")
        dc3.metric("Bayes Update Ratio", f"{demo['bayes_update_ratio']:.2f}×")
        st.info(demo["interpretation"])

    st.divider()
    st.subheader("Live Risk Predictor")
    with st.spinner("Training Bayesian model..."):
        engine = train_bayesian_model(df, cfg)

    evidence_cols = st.columns(4)
    obs = {}
    evidence_feats = [f for f in engine.evidence_features if f in df.columns]
    for i, feat in enumerate(evidence_feats[:8]):
        col = evidence_cols[i % 4]
        mn = float(df[feat].dropna().min())
        mx = float(df[feat].dropna().max())
        mean_v = float(df[feat].dropna().mean())
        obs[feat] = col.slider(feat, float(mn), float(mx), float(mean_v),
                               step=float((mx-mn)/200))

    if st.button("🔮 Compute Risk", type="primary"):
        posteriors = engine.predict_proba(obs)
        risk = engine.predict_risk_score(obs)
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("Risk Score", f"{risk:.3f}")
        rc2.metric("P(Healthy)", f"{posteriors['Healthy']:.3f}")
        rc3.metric("P(Early Degradation)", f"{posteriors['Early Degradation']:.3f}")
        rc4.metric("P(Degraded)", f"{posteriors['Degraded']:.3f}")

        risk_pct = risk * 100
        if risk_pct < 30:
            st.success(f"🟢 Low risk ({risk_pct:.1f}%) — Battery appears healthy.")
        elif risk_pct < 60:
            st.warning(f"🟡 Moderate risk ({risk_pct:.1f}%) — Monitor closely.")
        else:
            st.error(f"🔴 High risk ({risk_pct:.1f}%) — Degradation likely.")

        st.caption("⚠️ Risk score is a statistical model output — NOT a guaranteed failure prediction.")

    st.divider()
    st.subheader("Fitted Gaussian Likelihood Parameters")
    if engine.is_fitted:
        rows = []
        for feat in engine.evidence_features[:8]:
            row = {"Feature": feat}
            for cls in CLASSES:
                params = engine.class_params.get(cls, {}).get(feat, {})
                row[f"{cls} μ"] = round(params.get("mu", float("nan")), 4)
                row[f"{cls} σ"] = round(params.get("sigma", float("nan")), 4)
            rows.append(row)
        st.dataframe(pd.DataFrame(rows).set_index("Feature"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — POLYNOMIAL DEGRADATION MODEL
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[6]:
    st.title("📉 Polynomial Degradation Model")
    st.markdown("""
    ### Unit 1: Polynomial Curve Fitting · Least Squares · Model Selection

    $$\\text{SOH}(n) = w_0 + w_1 n + w_2 n^2 + \\cdots + w_d n^d$$

    Solved by least squares:
    $$\\mathbf{w}^* = (\\mathbf{X}^T \\mathbf{X})^{-1} \\mathbf{X}^T \\mathbf{y}$$

    Quality: $R^2 = 1 - SS_{\\text{res}} / SS_{\\text{tot}}$,  $\\text{RMSE} = \\sqrt{\\frac{1}{n}\\sum(y_i - \\hat{y}_i)^2}$
    """)

    bid = st.selectbox("Select Battery", sorted(df["battery_id"].unique()))
    bdf = df[df["battery_id"] == bid].sort_values("cycle_number")
    soh_v = bdf["soh"].dropna().values
    cyc_v = bdf.loc[bdf["soh"].notna(), "cycle_number"].values.astype(float)

    degrees = st.multiselect("Polynomial degrees to fit", [1, 2, 3, 4], default=[1, 2, 3])
    if len(soh_v) < 4:
        st.warning("Not enough cycles for this battery.")
    else:
        results = compare_polynomial_degrees(cyc_v, soh_v, degrees=degrees)
        fig_poly = go.Figure()
        fig_poly.add_trace(go.Scatter(x=cyc_v, y=soh_v, mode="markers",
                                       name="Observed SOH", marker=dict(size=5, color="#bdc3c7")))
        colors_deg = ["#2ecc71", "#3498db", "#e67e22", "#e74c3c"]
        x_ext = np.linspace(cyc_v.min(), cyc_v.max() * 1.25, 400)
        for (deg, res), col in zip(sorted(results.items()), colors_deg):
            fn = res["poly_fn"]
            fig_poly.add_trace(go.Scatter(
                x=x_ext, y=fn(x_ext), mode="lines",
                name=f"deg={deg} R²={res['r_squared']:.3f} RMSE={res['rmse']:.2f}",
                line=dict(color=col, width=2.5),
            ))
        fig_poly.add_hline(y=eol_pct, line_dash="dash", line_color="red",
                            annotation_text=f"EOL {eol_pct:.0f}% (configurable)")
        fig_poly.update_layout(xaxis_title="Cycle Number", yaxis_title="SOH (%)",
                                title=f"Battery {bid} — Polynomial Fits", height=420)
        st.plotly_chart(fig_poly, use_container_width=True)

        st.subheader("Fit Quality Comparison")
        rows = [{"Degree": d, "R²": round(r["r_squared"], 4), "RMSE": round(r["rmse"], 4)}
                for d, r in sorted(results.items())]
        st.table(pd.DataFrame(rows).set_index("Degree"))

        best_deg = max(results, key=lambda d: results[d]["r_squared"])
        st.subheader(f"EOL Prediction (degree-{best_deg} fit)")
        eol = predict_eol_cycle(results[best_deg]["poly_fn"],
                                int(cyc_v[-1]), eol_threshold=eol_pct)
        if eol.get("estimated_eol_cycle"):
            st.metric("Estimated EOL Cycle", eol["estimated_eol_cycle"],
                      f"{eol['cycles_remaining']} cycles remaining")
        st.warning(eol.get("note", ""))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[7]:
    st.title("🔵 K-Means Clustering")
    st.markdown("""
    ### Unit 1: Unsupervised Learning · K-Means · Elbow Method · Silhouette

    K-Means minimises **within-cluster sum of squares (WCSS)**:
    $$J = \\sum_{k=1}^{K} \\sum_{\\mathbf{x} \\in C_k} \\|\\mathbf{x} - \\boldsymbol{\\mu}_k\\|^2$$
    """)

    k = st.slider("Number of clusters K", 2, 8, 3)
    clust_feats = [f for f in ["soh", "capacity", "voltage_mean", "voltage_std",
                                "temperature_mean", "Re_Ohm", "cycle_norm"] if f in df.columns]

    with st.spinner("Running K-Means..."):
        X_scaled, feat_names, valid_idx = prepare_cluster_features(df, features=clust_feats)
        km, labels = fit_kmeans(X_scaled, k=k)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    var_exp = pca.explained_variance_ratio_

    df_valid = df.loc[valid_idx].copy()
    df_valid["cluster"] = labels
    df_valid["PC1"] = X_pca[:, 0]
    df_valid["PC2"] = X_pca[:, 1]

    fig_clust = px.scatter(
        df_valid, x="PC1", y="PC2", color=df_valid["cluster"].astype(str),
        hover_data=["battery_id", "soh", "cycle_number"],
        color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#3498db"],
        title=f"K={k} Clusters in PCA Space  (PC1={var_exp[0]*100:.1f}%  PC2={var_exp[1]*100:.1f}%)",
        opacity=0.55, height=420,
    )
    st.plotly_chart(fig_clust, use_container_width=True)

    st.subheader("Cluster Statistics")
    cluster_stats = interpret_clusters(df_valid, labels, valid_idx, feat_names)
    cluster_names = assign_cluster_names(cluster_stats)
    for _, row in cluster_stats.iterrows():
        cid = int(row["cluster_id"])
        name = cluster_names.get(cid, f"Cluster {cid}")
        soh_m = row.get("soh_mean", float("nan"))
        dom = row.get("dominant_label", "?")
        pur = row.get("label_purity", 0.0)
        st.metric(
            label=f"Cluster {cid}: {name}",
            value=f"n={int(row['n_members'])}",
            delta=f"mean SOH={soh_m:.1f}%  dominant={dom}  purity={pur:.0%}",
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 9 — INDEPENDENCE TESTING
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[8]:
    st.title("⚖️ Independence Testing")
    st.markdown("""
    ### Unit 1: Independence · Conditional Independence · Hypothesis Testing

    **Pearson correlation test:**
    H₀: ρ = 0 (no linear correlation)
    Reject H₀ if p-value < α

    **Key distinction:**
    - Independence → Cov(X,Y) = 0  (and all higher-order relations)
    - Corr(X,Y) = 0 does NOT imply independence (only no *linear* dependence)
    """)

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_cols = [c for c in num_cols if "bin" not in c and "flag" not in c
                    and "uid" not in c and c != "eol_flag"]

    col_a = st.selectbox("Variable X", numeric_cols, index=numeric_cols.index("cycle_number") if "cycle_number" in numeric_cols else 0)
    col_b = st.selectbox("Variable Y", numeric_cols, index=numeric_cols.index("soh") if "soh" in numeric_cols else 1)
    alpha = st.slider("Significance level α", 0.01, 0.10, 0.05, 0.01)

    if col_a != col_b:
        result = independence_test(df[col_a], df[col_b], alpha=alpha)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pearson r", f"{result['pearson_r']:.4f}")
        c2.metric("Pearson p-value", f"{result['pearson_p']:.4f}")
        c3.metric("Spearman r", f"{result['spearman_r']:.4f}")
        c4.metric("Spearman p-value", f"{result['spearman_p']:.4f}")

        if result["pearson_significant"] or result["spearman_significant"]:
            st.error(f"❌ Evidence of DEPENDENCE (p < α={alpha})")
        else:
            st.success(f"✅ Insufficient evidence to reject independence at α={alpha}")

        st.info(result["interpretation"])

        # Scatter plot
        clean = df[[col_a, col_b]].dropna().sample(min(1000, len(df)), random_state=42)
        fig_sc = px.scatter(clean, x=col_a, y=col_b,
                             opacity=0.4, height=350,
                             title=f"Scatter: {col_a} vs {col_b}")
        st.plotly_chart(fig_sc, use_container_width=True)

        st.caption("Note: A significant p-value indicates evidence of a *linear or monotonic* "
                   "relationship, NOT necessarily causation. Nonlinear dependencies may exist "
                   "even when Pearson r ≈ 0.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 10 — MODEL EVALUATION & LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════

elif page == PAGES[9]:
    st.title("✅ Model Evaluation & Limitations")

    st.subheader("Gaussian Naive Bayes — Classification Performance")
    pred_path = PROJECT_ROOT / "reports" / "results" / "bayesian_predictions.csv"
    if pred_path.exists():
        preds_df = pd.read_csv(pred_path)
        from sklearn.metrics import classification_report, confusion_matrix
        y_true = preds_df["degradation_label"].values
        y_pred = preds_df["predicted_class"].values
        mask = np.isin(y_true, CLASSES) & np.isin(y_pred, CLASSES)
        y_true_k, y_pred_k = y_true[mask], y_pred[mask]

        acc = (y_true_k == y_pred_k).mean()
        c1, c2 = st.columns(2)
        c1.metric("Overall Accuracy", f"{acc:.4f}")
        c2.metric("Test Cycles", len(y_true_k))

        cm = confusion_matrix(y_true_k, y_pred_k, labels=CLASSES)
        fig_cm = px.imshow(cm, x=CLASSES, y=CLASSES, text_auto=True,
                            color_continuous_scale="Blues", aspect="auto",
                            title="Confusion Matrix (Test Set)")
        fig_cm.update_xaxes(title="Predicted")
        fig_cm.update_yaxes(title="True")
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("Run `python experiments/experiment_02_bayesian.py` to generate predictions.")

    st.divider()
    st.subheader("Polynomial Model — Fleet R² Distribution")
    poly_path = PROJECT_ROOT / "reports" / "results" / "polynomial_fleet_summary.csv"
    if poly_path.exists():
        fleet_df = pd.read_csv(poly_path)
        fig_r2 = px.histogram(fleet_df, x="r_squared", nbins=20,
                               title="Fleet Polynomial Fit Quality (degree=2)\nR² distribution",
                               color_discrete_sequence=["#3498db"])
        st.plotly_chart(fig_r2, use_container_width=True)

    st.divider()
    st.subheader("⚠️ Documented Limitations")
    limitations = [
        ("EOL Threshold", "70% SOH is a **configurable research parameter** — NOT a universal safety standard. "
         "Real-world EOL depends on application requirements."),
        ("Naive Bayes Assumption", "Features are assumed conditionally independent given the class. "
         "In practice, voltage, current, and temperature are correlated."),
        ("Polynomial Extrapolation", "Predictions beyond observed cycle range have increasing uncertainty. "
         "Polynomial curves may behave non-physically outside training data."),
        ("Risk Score ≠ Probability", "Model outputs are statistical estimates from training data, "
         "NOT physically calibrated failure probabilities."),
        ("Laboratory Data", "NASA test data uses controlled cycling. Real-world operating conditions "
         "(variable C-rates, partial cycles, temperature swings) differ significantly."),
        ("Causation Disclaimer", "Observed correlations (e.g., temperature vs. capacity) reflect "
         "observational data. We do NOT claim causal relationships."),
        ("Safety Disclaimer", "This project is for **educational and research purposes only**. "
         "It is NOT a safety certification system and must NOT be used for safety-critical decisions."),
    ]
    for title_l, desc in limitations:
        with st.expander(f"⚠️ {title_l}"):
            st.markdown(desc)

    st.divider()
    st.subheader("Project File Summary")
    files_info = {
        "src/data_loader.py": "Metadata + cycle CSV loading with physical validity filters",
        "src/preprocessing.py": "SOH computation, impedance mapping, EOL/label assignment",
        "src/feature_engineering.py": "17 derived features with mathematical documentation",
        "src/statistics_engine.py": "Mean/Variance/Quantiles/Covariance/Independence testing",
        "src/probability_engine.py": "KDE/PDF, CDF, conditional distributions",
        "src/bayesian_risk.py": "Gaussian Naive Bayes + Bayes' Rule demonstration",
        "src/polynomial_model.py": "Degrees 1–4 fitting, EOL prediction",
        "src/clustering.py": "K-Means, Elbow method, Silhouette score",
        "src/visualization.py": "18+ publication-quality figures",
        "tests/": "60 pytest tests — all passing",
    }
    for fname, desc in files_info.items():
        st.markdown(f"**`{fname}`** — {desc}")
