from pathlib import Path
import sys, json
sys.path.append(str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.config import DATA_PROCESSED, TABLES, REPORTS

if __name__ == "__main__":
    summary_path = TABLES / "training_summary.json"
    metrics_path = TABLES / "model_comparison_holdout.csv"
    cluster_path = TABLES / "cluster_profiles.csv"
    if not summary_path.exists() or not metrics_path.exists():
        raise SystemExit("Run scripts/03_train_models.py first.")
    summary = json.loads(summary_path.read_text())
    metrics = pd.read_csv(metrics_path)
    clusters = pd.read_csv(cluster_path) if cluster_path.exists() else pd.DataFrame()
    lines = []
    lines.append("# Final Report Draft — MLB Hall of Fame Prediction + Player Archetype Clustering\n")
    lines.append("## 1. Problem Statement\n")
    lines.append("We ask whether career performance, awards, biographical attributes, era-adjusted statistics, and unsupervised player archetypes can predict whether an eligible MLB player is inducted into the Baseball Hall of Fame. The target is `inducted` (1 = inducted, 0 = eligible but not inducted).\n")
    lines.append("## 2. Data Collection and Preparation\n")
    lines.append("The project combines four sources: the full SABR / Lahman 1871–2025 CSV release as the primary relational baseball database, the Kaggle Baseball Databank as an audit / backup source, pybaseball ID and Statcast samples as a programmatic baseball-data source, and a BeautifulSoup scrape of the Hall of Fame Future Eligibles page as a semi-structured web source. We map `People.csv` to the project-compatible `Master.csv` name, join 15 relational tables on `playerID`, and use `Appearances.csv` as the canonical source for primary-position construction.\n")
    lines.append("## 3. EDA and Unsupervised Learning\n")
    lines.append("EDA includes class imbalance, missingness, era patterns, role distributions, selected correlations, and achievement distributions. K-means clustering, hierarchical clustering, and PCA are used to discover player archetypes and engineer cluster features.\n")
    if not clusters.empty:
        lines.append("### Archetype profile summary\n")
        lines.append(clusters.to_markdown(index=False))
        lines.append("\n")
    lines.append("## 4. Feature Engineering and Preprocessing\n")
    lines.append("Feature groups include career volume, rate statistics, peak 3-year and 5-year windows, awards, All-Star counts, postseason summaries, era-adjusted OPS/ERA proxies, biographical variables, role/position variables, cluster IDs, PCA coordinates, and distance-to-centroid features. Preprocessing uses median imputation, categorical imputation, one-hot encoding, standardization, and leakage-aware feature exclusions.\n")
    lines.append("## 5. Model Development\n")
    lines.append(f"Validation strategy: {summary['split_note']}. Training set n = {summary['train_n']} with {summary['train_positives']} positives; holdout n = {summary['test_n']} with {summary['test_positives']} positives. Models include tuned Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and a Stacking ensemble.\n")
    lines.append("## 6. Model Comparison and Selection\n")
    lines.append(metrics[["model","pr_auc","roc_auc","precision","recall","f1","balanced_accuracy","brier_score","top_25_precision"]].to_markdown(index=False))
    lines.append("\n")
    lines.append(f"The selected model is **{summary['best_model']}**, chosen primarily by holdout PR AUC and secondarily by ROC AUC, calibration, interpretability, and robustness across roles/eras.\n")
    lines.append("## 7. Interpretation and Communication\n")
    lines.append("The report should emphasize the story: what separates legends from very good players? Interpret the selected model using permutation importance, PR/ROC curves, calibration, cluster profiles, and borderline-player examples.\n")
    lines.append("## 8. Limitations and Future Work\n")
    lines.append("Key limitations include incomplete modern data in Kaggle, era bias, historical changes in Hall voting behavior, salary missingness before 1985, and target timing for recently eligible players. Future work can incorporate latest SABR/Lahman tables, pybaseball WAR/Statcast features, and survival analysis for time-to-induction.\n")
    lines.append("## 9. Member Contributions\n")
    lines.append("Replace this section with your actual team contributions before submission.\n")
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "final_report_draft.md").write_text("\n".join(lines))
    print("Saved reports/final_report_draft.md")
