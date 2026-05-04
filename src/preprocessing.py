"""
preprocessing.py
================

Multi-variant preprocessing for downstream supervised modeling.

This module exports four scaled variants of the model matrix (raw, standardized,
min-max, power+standardized) so each model class can use the most
appropriate scaling for its assumptions:

  * Tree-based models (RF, XGBoost, LightGBM) are scale-invariant → use raw
  * Logistic Regression / SVM / distance-based methods → use standardized
  * Bounded-scale algorithms (some neural nets) → use min-max
  * Linear models that assume normality of features → use Yeo-Johnson + std

The variants are applied to the HoF career-level feature matrix. Only continuous columns are scaled; binary indicators / one-hot
columns are kept on their natural 0/1 scale.

Author: Team 6 — Person C (Feature Engineering)
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer

# Columns that should NOT be features (target, ids, leakage fields)
NON_FEATURE_COLUMNS = {
    "playerID", "player_name", "nameFirst", "nameLast",
    "inducted", "is_eligible", "model_eligible_pool",
    "induction_year", "hof_first_ballot_year", "hof_last_ballot_year",
    "hof_ballot_rows",
    "eligibility_year", "first_year", "last_year", "final_year", "debut_year",
}


def _is_binary_or_indicator(s: pd.Series) -> bool:
    """Return True if the series looks like a 0/1 indicator or one-hot column."""
    if not pd.api.types.is_numeric_dtype(s):
        return False
    valid = s.dropna()
    if len(valid) == 0:
        return False
    unique = set(valid.unique())
    return unique.issubset({0, 1, 0.0, 1.0, True, False})


def export_scaled_variants(df: pd.DataFrame,
                           output_dir: Path,
                           target_col: str = "inducted",
                           id_col: str = "playerID") -> Dict[str, Path]:
    """Export 4 scaling variants of the modeling-ready matrix.

    Only the rows where `model_eligible_pool == 1` are exported (training pool).
    Continuous numeric columns are transformed; binary / one-hot columns are
    preserved on their 0/1 scale; ID and target columns are kept unscaled.

    Returns
    -------
    dict mapping variant_name -> output Path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pool = df[df["model_eligible_pool"] == 1].copy()
    if id_col not in pool.columns or target_col not in pool.columns:
        raise ValueError(f"Pool missing {id_col} or {target_col}")

    ids = pool[id_col].copy()
    y = pool[target_col].astype(int).copy()

    # Identify continuous numeric features vs binary indicators
    candidate_cols = [c for c in pool.columns
                      if c not in NON_FEATURE_COLUMNS and c != id_col]
    numeric_cols = [c for c in candidate_cols
                    if pd.api.types.is_numeric_dtype(pool[c])]
    binary_cols = [c for c in numeric_cols if _is_binary_or_indicator(pool[c])]
    continuous_cols = [c for c in numeric_cols if c not in binary_cols]

    # Median-impute continuous, mode-impute binary, replace inf
    X_cont = pool[continuous_cols].replace([np.inf, -np.inf], np.nan)
    if continuous_cols:
        X_cont_imputed = pd.DataFrame(
            SimpleImputer(strategy="median").fit_transform(X_cont),
            columns=continuous_cols, index=pool.index)
    else:
        X_cont_imputed = pd.DataFrame(index=pool.index)

    X_bin = pool[binary_cols].fillna(0).astype(int) if binary_cols else \
        pd.DataFrame(index=pool.index)

    paths: Dict[str, Path] = {}

    # ---- Variant 1: Raw (no scaling) — for tree-based ----
    raw = pd.concat([X_cont_imputed, X_bin], axis=1)
    raw[id_col] = ids.values
    raw[target_col] = y.values
    p1 = output_dir / "model_matrix_raw.csv"
    raw.to_csv(p1, index=False)
    paths["raw"] = p1

    # ---- Variant 2: Standardized (Z-score on continuous only) ----
    if continuous_cols:
        std_arr = StandardScaler().fit_transform(X_cont_imputed)
        X_std = pd.DataFrame(std_arr, columns=continuous_cols, index=pool.index)
    else:
        X_std = X_cont_imputed
    std = pd.concat([X_std, X_bin], axis=1)
    std[id_col] = ids.values
    std[target_col] = y.values
    p2 = output_dir / "model_matrix_standardized.csv"
    std.to_csv(p2, index=False)
    paths["standardized"] = p2

    # ---- Variant 3: Min-Max scaled (on continuous only) ----
    if continuous_cols:
        mm_arr = MinMaxScaler().fit_transform(X_cont_imputed)
        X_mm = pd.DataFrame(mm_arr, columns=continuous_cols, index=pool.index)
    else:
        X_mm = X_cont_imputed
    mm = pd.concat([X_mm, X_bin], axis=1)
    mm[id_col] = ids.values
    mm[target_col] = y.values
    p3 = output_dir / "model_matrix_minmax.csv"
    mm.to_csv(p3, index=False)
    paths["minmax"] = p3

    # ---- Variant 4: Yeo-Johnson + standardize (heavy-tailed normalization) ----
    if continuous_cols:
        try:
            yj_arr = PowerTransformer(method="yeo-johnson",
                                      standardize=True).fit_transform(X_cont_imputed)
            X_yj = pd.DataFrame(yj_arr, columns=continuous_cols, index=pool.index)
        except Exception:
            X_yj = X_cont_imputed
    else:
        X_yj = X_cont_imputed
    yj = pd.concat([X_yj, X_bin], axis=1)
    yj[id_col] = ids.values
    yj[target_col] = y.values
    p4 = output_dir / "model_matrix_power.csv"
    yj.to_csv(p4, index=False)
    paths["power"] = p4

    # Manifest documenting variant counts
    manifest = pd.DataFrame([{
        "variant": k,
        "path": str(v.relative_to(output_dir.parent.parent)),
        "n_rows": len(pool),
        "n_continuous_features": len(continuous_cols),
        "n_binary_features": len(binary_cols),
        "n_total_features": len(continuous_cols) + len(binary_cols),
    } for k, v in paths.items()])
    manifest.to_csv(output_dir / "scaled_variants_manifest.csv", index=False)

    return paths
