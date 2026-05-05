from pathlib import Path
import sys
import json

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import TABLES, REPORTS, DATA_PROCESSED


def md_table(df: pd.DataFrame, columns=None, n=None) -> str:
    if df is None or df.empty:
        return "_Missing or empty._"
    out = df.copy()
    if columns:
        out = out[[c for c in columns if c in out.columns]]
    if n:
        out = out.head(n)
    return out.to_markdown(index=False)


def csv_shape(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    cols = len(pd.read_csv(path, nrows=0).columns)
    rows = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1
    return rows, cols


if __name__ == "__main__":
    summary_path = TABLES / "training_summary.json"
    metrics_path = TABLES / "model_comparison_holdout.csv"
    cluster_path = TABLES / "cluster_profiles.csv"
    kpi_path = TABLES / "summary_kpis.csv"

    if not summary_path.exists() or not metrics_path.exists():
        raise SystemExit("Run scripts/03_train_models.py first.")

    summary = json.loads(summary_path.read_text())
    metrics = pd.read_csv(metrics_path)
    clusters = pd.read_csv(cluster_path) if cluster_path.exists() else pd.DataFrame()
    kpis = pd.read_csv(kpi_path) if kpi_path.exists() else pd.DataFrame()

    base_rows, base_cols = csv_shape(DATA_PROCESSED / "player_features_base.csv")
    arch_rows, arch_cols = csv_shape(DATA_PROCESSED / "player_features_with_archetypes.csv")

    models_list = ", ".join(metrics["model"].astype(str).tolist())
    best_model = summary.get("best_model", metrics.sort_values("pr_auc", ascending=False).iloc[0]["model"])

    eligible = int(kpis["n_eligible_players"].iloc[0]) if "n_eligible_players" in kpis.columns else "unknown"
    positives = int(kpis["n_inducted"].iloc[0]) if "n_inducted" in kpis.columns else "unknown"
    induction_rate = float(kpis["induction_rate"].iloc[0]) if "induction_rate" in kpis.columns else None

    lines = []
    lines.append("# Final Report Draft — MLB Hall of Fame Prediction + Player Archetype Clustering\n")
    lines.append("## 1. Problem Statement\n")
    lines.append("We ask whether career performance, awards, biographical attributes, era-adjusted statistics, and unsupervised player archetypes can predict whether an eligible MLB player is inducted into the Baseball Hall of Fame. The target is `inducted` (1 = inducted, 0 = eligible but not inducted).\n")
    lines.append("The project story is: **what separates Cooperstown legends from very good players?**\n")

    lines.append("## 2. Data Collection and Preparation\n")
    lines.append("The project combines four sources: the full SABR/Lahman 1871-2025 CSV release as the primary relational baseball database, Kaggle Baseball Databank as an audit and backup source, pybaseball ID and Statcast samples as programmatic baseball-data acquisition examples, and a BeautifulSoup scrape of the Hall of Fame Future Eligibles page as a semi-structured web source.\n")
    lines.append("The acquisition step maps `People.csv` to the project-compatible `Master.csv` name, joins relational tables mainly on `playerID`, and uses `Appearances.csv` as the canonical source for primary-position construction. The main data challenge is that raw baseball data are fragmented across player-team-season rows, so the pipeline aggregates team stints and constructs one career-level row per player.\n")
    lines.append(f"The final base player feature table contains **{base_rows:,} players and {base_cols:,} columns**. The eligible modeling pool contains **{eligible:,} players**, including **{positives:,} inducted positives**.\n")

    lines.append("## 3. EDA and Unsupervised Learning\n")
    if induction_rate is not None:
        lines.append(f"EDA includes dataset-level summary statistics, class imbalance, missingness analysis, role and era comparisons, selected correlations, hypothesis tests, VIF diagnostics, influence diagnostics, and class-conditional achievement distributions. The headline induction rate in the eligible modeling pool is **{induction_rate:.4f}**.\n")
    else:
        lines.append("EDA includes dataset-level summary statistics, class imbalance, missingness analysis, role and era comparisons, selected correlations, hypothesis tests, VIF diagnostics, influence diagnostics, and class-conditional achievement distributions.\n")
    lines.append("K-means clustering, hierarchical clustering diagnostics, and PCA are used to discover player archetypes. These archetype features are merged into the supervised modeling dataset.\n")
    if not clusters.empty:
        lines.append("### Archetype profile summary\n")
        lines.append(clusters.to_markdown(index=False))
        lines.append("\n")

    lines.append("## 4. Feature Engineering and Preprocessing\n")
    lines.append(f"Feature groups include career volume, rate statistics, peak 3-year and 5-year windows, awards, All-Star counts, postseason summaries, era-adjusted OPS/ERA proxies, biographical variables, role and position variables, cluster IDs, PCA coordinates, and distance-to-centroid features. Preprocessing includes imputation, one-hot encoding, standardization, min-max scaling, power transformation, and leakage-aware feature exclusions. The final archetype modeling table contains **{arch_rows:,} rows and {arch_cols:,} columns**.\n")

    lines.append("## 5. Model Development\n")
    lines.append(f"Validation strategy: **{summary['split_note']}**. The training set contains **{summary['train_n']:,} players** with **{summary['train_positives']:,} positives**. The holdout set contains **{summary['test_n']:,} players** with **{summary['test_positives']:,} positives**.\n")
    lines.append(f"The actual reported models are: **{models_list}**. Hyperparameter tuning is evaluated with cross-validated average precision, and model selection emphasizes PR AUC because Hall of Fame induction is a rare positive class.\n")

    lines.append("## 6. Model Comparison and Selection\n")
    lines.append(md_table(metrics, ["model", "pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy", "brier_score", "top_25_precision", "top_50_precision"]))
    lines.append("\n")
    lines.append(f"The selected model is **{best_model}**, chosen primarily by holdout PR AUC. It achieved **PR AUC {summary['best_holdout_pr_auc']:.6f}** and **ROC AUC {summary['best_holdout_roc_auc']:.6f}** on the time-based holdout set.\n")

    lines.append("## 7. Interpretation and Communication\n")
    lines.append("The most important feature groups include awards, All-Star appearances, peak performance, career totals, cluster features, and era-adjusted metrics. This suggests that Hall of Fame induction is associated with both statistical production and historical recognition. The Shiny app, **Cooperstown Calculator**, reads `app/model_predictions.csv` and lets users inspect player-level induction probabilities.\n")

    lines.append("## 8. Limitations and Future Work\n")
    lines.append("This is an observational prediction project, so the results describe associations rather than causal effects. Hall voting behavior changes across eras, early baseball records can be noisier, salary coverage starts in 1985, and modern advanced metrics such as WAR/JAWS are not fully integrated. Future work could incorporate richer WAR/JAWS sources, expand pybaseball Statcast features, and model time from retirement to induction with survival analysis.\n")

    lines.append("## 9. Member Contributions\n")
    lines.append("Replace this section with actual team member contributions before final submission.\n")

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "final_report_draft.md").write_text("\n".join(lines))
    print("Saved reports/final_report_draft.md")
