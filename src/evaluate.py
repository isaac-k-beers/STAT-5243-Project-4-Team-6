from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from sklearn.calibration import calibration_curve


def save_curves(y_true, proba, figures_dir: Path, prefix: str = "final"):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, proba)
    plt.figure(figsize=(6, 4)); plt.plot(fpr, tpr); plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC Curve")
    plt.tight_layout(); plt.savefig(figures_dir / f"{prefix}_roc_curve.png", dpi=160); plt.close()
    precision, recall, _ = precision_recall_curve(y_true, proba)
    plt.figure(figsize=(6, 4)); plt.plot(recall, precision)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve")
    plt.tight_layout(); plt.savefig(figures_dir / f"{prefix}_pr_curve.png", dpi=160); plt.close()
    frac_pos, mean_pred = calibration_curve(y_true, proba, n_bins=8, strategy="quantile")
    plt.figure(figsize=(6, 4)); plt.plot(mean_pred, frac_pos, marker="o"); plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Observed induction rate"); plt.title("Calibration Curve")
    plt.tight_layout(); plt.savefig(figures_dir / f"{prefix}_calibration_curve.png", dpi=160); plt.close()
    cm = confusion_matrix(y_true, (proba >= 0.5).astype(int), labels=[0, 1])
    plt.figure(figsize=(4, 4)); plt.imshow(cm); plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"]); plt.yticks([0, 1], ["Actual 0", "Actual 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.title("Confusion Matrix @ 0.50")
    plt.tight_layout(); plt.savefig(figures_dir / f"{prefix}_confusion_matrix.png", dpi=160); plt.close()


def stratified_metrics_table(pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col in ["primary_role", "debut_era", "cluster_label"]:
        if group_col not in pred_df.columns:
            continue
        for value, g in pred_df.groupby(group_col):
            if len(g) < 10:
                continue
            rows.append({
                "grouping": group_col,
                "value": value,
                "n": len(g),
                "actual_induction_rate": g["inducted"].mean(),
                "mean_predicted_probability": g["predicted_probability"].mean(),
                "top25_share_if_applicable": np.nan
            })
    return pd.DataFrame(rows)
