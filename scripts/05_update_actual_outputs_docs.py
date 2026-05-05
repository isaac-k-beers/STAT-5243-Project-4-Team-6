from pathlib import Path
import json
from datetime import datetime

import pandas as pd

ROOT = Path(".").resolve()
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)


def read_csv(path):
    p = ROOT / path
    return pd.read_csv(p) if p.exists() else None


def read_json(path):
    p = ROOT / path
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def shape_of(path):
    p = ROOT / path
    if not p.exists():
        return "missing"
    cols = len(pd.read_csv(p, nrows=0).columns)
    rows = sum(1 for _ in open(p, encoding="utf-8", errors="ignore")) - 1
    return f"{rows:,} rows × {cols:,} columns"


def md_table(df, cols=None, n=None):
    if df is None or df.empty:
        return "_Missing or empty._"
    out = df.copy()
    if cols:
        out = out[[c for c in cols if c in out.columns]]
    if n:
        out = out.head(n)
    return out.to_markdown(index=False)


manifest = read_csv("reports/tables/part1_source_manifest.csv")
external = read_csv("reports/tables/part1_external_source_status.csv")
kpis = read_csv("reports/tables/summary_kpis.csv")
clusters = read_csv("reports/tables/cluster_diagnostics.csv")
models = read_csv("reports/tables/model_comparison_holdout.csv")
training = read_json("reports/tables/training_summary.json")
scaled = read_csv("data/processed/scaled_variants_manifest.csv")

best_model = training.get("best_model", "unknown")
best_pr = training.get("best_holdout_pr_auc", "unknown")
best_roc = training.get("best_holdout_roc_auc", "unknown")

content = f"""# Actual Outputs Summary

Generated at: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

This file is generated from the current local output files after rerunning the full pipeline. Use these numbers in the final report and presentation.

## 1. Processed data shapes

| File | Actual shape | Purpose |
|---|---:|---|
| `data/processed/player_features_base.csv` | {shape_of("data/processed/player_features_base.csv")} | Cleaned player-level feature table before clustering |
| `data/processed/player_features_with_archetypes.csv` | {shape_of("data/processed/player_features_with_archetypes.csv")} | Final modeling dataset with archetype features |
| `data/processed/model_predictions_all_eligible.csv` | {shape_of("data/processed/model_predictions_all_eligible.csv")} | Final predictions for eligible players |
| `app/model_predictions.csv` | {shape_of("app/model_predictions.csv")} | Prediction file used by the Shiny app |

## 2. Data source manifest

{md_table(manifest, ["table", "status", "source_used", "rows", "min_year", "max_year"])}

## 3. External source status

{md_table(external)}

## 4. Headline EDA KPIs

{md_table(kpis)}

## 5. Cluster diagnostics

{md_table(clusters, ["k", "silhouette", "davies_bouldin", "inertia"])}

## 6. Preprocessing variants

{md_table(scaled)}

## 7. Model comparison

{md_table(models, ["model", "pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy", "brier_score", "top_25_precision", "top_50_precision"])}

## 8. Training summary

| Quantity | Actual value |
|---|---:|
| Best model | `{best_model}` |
| Best holdout PR AUC | {best_pr} |
| Best holdout ROC AUC | {best_roc} |
| Train rows | {training.get("train_n", "unknown")} |
| Test rows | {training.get("test_n", "unknown")} |
| Train positives | {training.get("train_positives", "unknown")} |
| Test positives | {training.get("test_positives", "unknown")} |
| Features used | {training.get("features_used", "unknown")} |

## 9. Report-writing note

Use the numbers in this file as the source of truth. If older markdown files mention different values, update them to match this file.
"""

(DOCS / "actual_outputs_summary.md").write_text(content)

recheck = DOCS / "recheck_status.md"
old = recheck.read_text() if recheck.exists() else "# Recheck Status\n"
marker = "\n## Latest Actual Output Summary\n"
section = marker + "\n" + content.replace("# Actual Outputs Summary\n", "")
if marker in old:
    old = old.split(marker)[0].rstrip() + section
else:
    old = old.rstrip() + "\n" + section
recheck.write_text(old)

print("Updated docs/actual_outputs_summary.md")
print("Updated docs/recheck_status.md")
