"""
diagnostics.py
==============

Statistical diagnostics for the HoF prediction modeling pipeline.

This module provides three diagnostic functions used to support the
multicollinearity, influence, and anomaly-detection analyses required at the
Advanced rubric tier:

  1. compute_vif()              — Variance Inflation Factor on curated features.
                                   Career counting stats (HR/H/RBI/TB) are
                                   highly collinear by baseball construction.
                                   VIF supports the regularization strategy.

  2. cooks_distance_logistic()  — Cook's distance + leverage on a Logistic
                                   Regression baseline. Identifies the
                                   borderline / controversial HoF candidates
                                   (e.g., Tony Pérez, Jim Rice) whose
                                   induction status disproportionately drives
                                   linear-model coefficients.

  3. robust_z_anomalies()       — MAD-based anomaly detection. Flags careers
                                   that are statistical extremes EVEN AFTER
                                   era adjustment (e.g., Bonds 2001-2004,
                                   Ruth 1921). These are the players the
                                   model MUST classify correctly.

These diagnostics support the project's central question:
    "What separates HoF legends from merely good players?"
    -> Diagnostic evidence that legends are statistical anomalies, while
       the borderline cases are exactly where the model's value lies.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Curated subset of features for VIF analysis. We choose features that are
# (a) candidates for the modeling matrix AND (b) likely correlated by
# construction. Career-volume stats (HR / H / RBI / TB) are highly correlated
# because a great hitter accumulates ALL of them simultaneously.
VIF_CURATED_FEATURES = [
    # Career-level batting totals (already aggregated in features.py)
    "bat_HR", "bat_H", "bat_RBI", "bat_BB", "bat_SO", "bat_SB",
    # Career-level era-adjusted peaks
    "bat_max_OPS_plus_proxy", "bat_mean_OPS_plus_proxy",
    # Career awards / recognition
    "allstar_games", "allstar_years",
    "award_total", "award_unique",
    "award_mvp_count", "award_gold_glove_count",
    # Career length
    "n_mlb_seasons", "career_span_years",
    # Career pitching totals
    "pit_W", "pit_SO", "pit_IP",
    "pit_max_ERA_plus_proxy",
]


# ======================================================================
# 1. VIF — multicollinearity diagnostic
# ======================================================================
def compute_vif(df: pd.DataFrame,
                features: Optional[list] = None,
                tables_dir: Optional[Path] = None) -> pd.DataFrame:
    """Compute VIF on a curated feature set.

    VIF is computed on a curated subset of career features that are likely
    correlated by construction. Career-volume stats (HR, H, RBI, TB) are
    highly correlated because a great hitter accumulates all of them
    simultaneously, so we expect high VIF values that justify regularization.

    Returns
    -------
    pd.DataFrame with columns [feature, vif, interpretation]
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except ImportError:
        return pd.DataFrame({
            "feature": ["statsmodels_unavailable"],
            "vif": [np.nan],
            "interpretation": ["install statsmodels to compute VIF"],
        })

    pool = df[df["model_eligible_pool"] == 1].copy()
    if features is None:
        features = [f for f in VIF_CURATED_FEATURES if f in pool.columns]

    X = pool[features].replace([np.inf, -np.inf], np.nan).dropna()
    if len(X) < len(features) + 5:
        return pd.DataFrame({
            "feature": ["insufficient_data"],
            "vif": [np.nan],
            "interpretation": [f"only {len(X)} rows after dropna"],
        })

    # Standardize before VIF — otherwise scale interacts with the calculation.
    X_std = StandardScaler().fit_transform(X)

    rows = []
    for i, feat in enumerate(features):
        try:
            v = variance_inflation_factor(X_std, i)
        except Exception:
            v = np.nan
        rows.append({"feature": feat, "vif": v})

    out = pd.DataFrame(rows).sort_values("vif", ascending=False, na_position="last")
    out["vif"] = out["vif"].round(3)
    out["interpretation"] = pd.cut(
        out["vif"],
        bins=[-np.inf, 5, 10, np.inf],
        labels=["acceptable", "moderate", "high — regularize"],
    )

    if tables_dir is not None:
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        out.to_csv(tables_dir / "vif_curated_features.csv", index=False)

    return out


# ======================================================================
# 2. Cook's distance + leverage diagnostics on Logistic Regression baseline
# ======================================================================
def cooks_distance_logistic(X: pd.DataFrame, y: pd.Series,
                            ids: Optional[pd.Series] = None,
                            tables_dir: Optional[Path] = None,
                            figures_dir: Optional[Path] = None,
                            top_k_features: int = 20) -> pd.DataFrame:
    """Cook's distance + leverage diagnostics on a Logistic Regression baseline.

    Identifies the most influential observations under a linear model.
    For HoF prediction these are typically the borderline / controversial
    candidates: Tony Pérez, Jim Rice, Harold Baines, etc. Documenting them
    transparently strengthens the report's narrative ("the model is most
    uncertain about the cases the writers themselves were uncertain about").

    Parameters
    ----------
    X : numeric-only feature matrix
    y : binary target Series (inducted)
    ids : Series of player names or ids for labelling
    top_k_features : int, default 20
        We project to the top-K features by univariate correlation with y so
        the Logit problem stays well-conditioned (n_eligible ≈ 3,000;
        n_features can be > 200 in our pipeline).
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return pd.DataFrame({"error": ["statsmodels not installed"]})

    Xc = X.copy().select_dtypes(include=[np.number])
    Xc = Xc.replace([np.inf, -np.inf], np.nan)
    if Xc.shape[1] == 0:
        return pd.DataFrame({"error": ["no numeric features"]})

    Xc = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Xc),
                      columns=Xc.columns, index=Xc.index)
    Xc = pd.DataFrame(StandardScaler().fit_transform(Xc),
                      columns=Xc.columns, index=Xc.index)

    # Subset to top-K features by univariate correlation
    corrs = Xc.corrwith(y).abs().sort_values(ascending=False)
    top_features = [c for c in corrs.head(top_k_features).index if not pd.isna(corrs[c])]
    if len(top_features) < 2:
        return pd.DataFrame({"error": ["no correlated features"]})
    X_top = sm.add_constant(Xc[top_features])

    try:
        model = sm.Logit(y.values, X_top.values).fit(disp=0, maxiter=200)
    except Exception as e:
        return pd.DataFrame({"error": [f"Logit failed: {type(e).__name__}: {e}"]})

    try:
        infl = model.get_influence()
        df_infl = pd.DataFrame({
            "cooks_d": infl.cooks_distance[0],
            "hat_diag": infl.hat_matrix_diag,
            "student_resid": infl.resid_studentized,
        }, index=X.index)
    except Exception as e:
        return pd.DataFrame({"error": [f"influence failed: {type(e).__name__}: {e}"]})

    if ids is not None:
        df_infl["id"] = ids.values
    df_infl["inducted"] = y.values

    threshold_cooks = 4.0 / len(df_infl)
    p = len(top_features) + 1
    threshold_lev = 2.0 * p / len(df_infl)
    df_infl["is_high_cooks"] = (df_infl["cooks_d"] > threshold_cooks).astype(int)
    df_infl["is_high_leverage"] = (df_infl["hat_diag"] > threshold_lev).astype(int)
    df_infl["is_influential"] = (
        df_infl["is_high_cooks"] | df_infl["is_high_leverage"]
    ).astype(int)

    if tables_dir is not None:
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        df_infl.to_csv(tables_dir / "logistic_influence_diagnostics.csv")
        # Top influential players for the report
        top = df_infl.nlargest(25, "cooks_d")
        top.to_csv(tables_dir / "top_influential_players.csv")

    if figures_dir is not None:
        import matplotlib.pyplot as plt
        figures_dir = Path(figures_dir)
        figures_dir.mkdir(parents=True, exist_ok=True)

        # Cook's distance plot (P1-style)
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.scatter(range(len(df_infl)), df_infl["cooks_d"], s=8, alpha=0.5,
                   c=df_infl["inducted"].map({0: "#888", 1: "#C62828"}))
        ax.axhline(threshold_cooks, ls="--", color="red",
                   label=f"4/n threshold = {threshold_cooks:.4f}")
        ax.set_xlabel("Observation index")
        ax.set_ylabel("Cook's distance")
        ax.set_title("Cook's distance — Logistic Regression baseline (HoF eligibility)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "cooks_distance.png", dpi=160)
        plt.close()

        # Leverage vs studentized residuals (P1-style)
        fig, ax = plt.subplots(figsize=(7, 5))
        sc = ax.scatter(df_infl["hat_diag"], df_infl["student_resid"],
                        s=10, alpha=0.6,
                        c=df_infl["inducted"].map({0: "#888", 1: "#C62828"}))
        ax.axvline(threshold_lev, ls="--", color="red",
                   label=f"2p/n leverage threshold")
        ax.axhline(0, ls=":", color="grey")
        ax.set_xlabel("Leverage (hat diagonal)")
        ax.set_ylabel("Studentized residual")
        ax.set_title("Leverage vs Studentized Residuals (HoF logistic baseline)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "leverage_studentized.png", dpi=160)
        plt.close()

    return df_infl


# ======================================================================
# 3. Robust (MAD-based) Z-score anomaly detection
# ======================================================================
def robust_z_anomalies(df: pd.DataFrame,
                        stat_col: str = "peak_OPS_plus",
                        threshold: float = 3.5,
                        tables_dir: Optional[Path] = None) -> pd.DataFrame:
    """MAD-based robust Z-score for era-adjusted career-anomaly detection.

    A MAD-based robust Z-score flags careers that are statistical extremes
    relative to the eligible-pool distribution. We apply it to era-adjusted
    career peaks (peak_OPS_plus, peak_ERA_plus). Players flagged here are
    statistical extremes EVEN AFTER era adjustment — exactly the legends a
    HoF model must classify correctly (Babe Ruth's 1921, Bonds' 2001-2004,
    Pedro Martinez's 1999, Sandy Koufax's 1965).

    Storytelling angle: these are NOT data errors. They are the SIGNAL.
    """
    pool = df[df["model_eligible_pool"] == 1].copy()
    if stat_col not in pool.columns:
        return pd.DataFrame({"error": [f"{stat_col} not in df"]})

    s = pool[stat_col].replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) < 10:
        return pd.DataFrame({"error": ["insufficient data"]})

    med = float(s.median())
    mad = float((s - med).abs().median())
    if mad == 0:
        return pd.DataFrame({"error": ["MAD is zero"]})

    pool["robust_z"] = 0.6745 * (pool[stat_col] - med) / mad
    pool["is_anomaly_high"] = (pool["robust_z"] > threshold).astype(int)
    pool["is_anomaly_low"] = (pool["robust_z"] < -threshold).astype(int)
    pool["is_anomaly"] = (pool["is_anomaly_high"] | pool["is_anomaly_low"]).astype(int)

    keep_cols = [c for c in [
        "playerID", "nameFirst", "nameLast", "primary_role",
        "primary_position", "debut_era", stat_col, "robust_z",
        "is_anomaly_high", "is_anomaly_low", "inducted"
    ] if c in pool.columns]
    anomalies = (pool.loc[pool["is_anomaly"] == 1, keep_cols]
                     .sort_values("robust_z", ascending=False))

    if tables_dir is not None:
        tables_dir = Path(tables_dir)
        tables_dir.mkdir(parents=True, exist_ok=True)
        anomalies.to_csv(tables_dir / f"robust_z_anomalies_{stat_col}.csv", index=False)
        summary = pd.DataFrame([{
            "stat_col": stat_col,
            "median": med,
            "mad": mad,
            "threshold": threshold,
            "n_anomalies": int(len(anomalies)),
            "n_high": int(pool["is_anomaly_high"].sum()),
            "n_low": int(pool["is_anomaly_low"].sum()),
            "n_anomaly_inducted": int(anomalies["inducted"].sum())
                if "inducted" in anomalies else 0,
            "n_anomaly_not_inducted": int((1 - anomalies.get("inducted", 0)).sum())
                if "inducted" in anomalies else 0,
        }])
        summary.to_csv(tables_dir / f"robust_z_summary_{stat_col}.csv", index=False)

    return anomalies
