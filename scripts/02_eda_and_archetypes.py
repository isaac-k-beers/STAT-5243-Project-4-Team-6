from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import DATA_PROCESSED, FIGURES, TABLES, MODELS, RANDOM_STATE
from src.clustering import fit_archetypes
from src.eda import (
    make_eda_outputs, make_summary_statistics,
    make_inducted_vs_not_boxplots, make_annotated_scatter,
    make_hof_rate_with_ci_by_era, make_hypothesis_tests,
)
from src.diagnostics import compute_vif, robust_z_anomalies
from src.preprocessing import export_scaled_variants
import pandas as pd
import joblib
import matplotlib.pyplot as plt

if __name__ == "__main__":
    base_path = DATA_PROCESSED / "player_features_base.csv"
    if not base_path.exists():
        raise SystemExit("Run scripts/01_build_player_features.py first.")
    df = pd.read_csv(base_path, low_memory=False)

    # Basic summary statistics required for the Advanced EDA rubric tier.
    print("Producing summary statistics tables ...")
    summary_outputs = make_summary_statistics(df, TABLES)
    print(f"  -> {len(summary_outputs)} summary tables written to {TABLES}")
    print("\nHeadline KPIs:")
    print(summary_outputs["kpis"].T.to_string(header=False))

    print("\nProducing EDA figures ...")
    make_eda_outputs(df, FIGURES, TABLES)

    # Player archetype clustering (k=3..7 for proper diagnostics).
    print("\nFitting player archetypes ...")
    out, diagnostics, km, prep, pca, cluster_features = fit_archetypes(df, k_range=range(3, 8), random_state=RANDOM_STATE)
    out.to_csv(DATA_PROCESSED / "player_features_with_archetypes.csv", index=False)
    diagnostics.to_csv(TABLES / "cluster_diagnostics.csv", index=False)
    pd.DataFrame({"cluster_feature": cluster_features}).to_csv(TABLES / "cluster_features_used.csv", index=False)
    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kmeans": km, "preprocess": prep, "pca": pca, "features": cluster_features}, MODELS / "archetype_pipeline.joblib")
    # Figures for clustering.
    plt.figure(figsize=(7, 4))
    plt.plot(diagnostics["k"], diagnostics["inertia"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("K-means inertia / WSS")
    plt.title("Elbow Plot for Player Archetypes")
    plt.tight_layout(); plt.savefig(FIGURES / "06_cluster_elbow.png", dpi=160); plt.close()
    plt.figure(figsize=(7, 4))
    plt.plot(diagnostics["k"], diagnostics["silhouette"], marker="o")
    plt.xlabel("Number of clusters k")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette Scores for Player Archetypes")
    plt.tight_layout(); plt.savefig(FIGURES / "07_cluster_silhouette.png", dpi=160); plt.close()
    pool = out[out["model_eligible_pool"] == 1]
    plt.figure(figsize=(7, 5))
    for cid, g in pool.groupby("cluster_id"):
        plt.scatter(g["pca1"], g["pca2"], s=16, alpha=0.65, label=f"Cluster {int(cid)}")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("Player Archetype Clusters in PCA Space")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout(); plt.savefig(FIGURES / "08_cluster_pca.png", dpi=160); plt.close()
    cluster_profile = pool.groupby(["cluster_id", "cluster_label"]).agg(
        players=("playerID", "count"), hof_rate=("inducted", "mean"),
        median_hr=("bat_HR", "median"), median_ops=("bat_OPS", "median"),
        median_wins=("pit_W", "median"), median_saves=("pit_SV", "median"),
        median_awards=("award_total", "median"), median_allstar=("allstar_games", "median")
    ).reset_index()
    cluster_profile.to_csv(TABLES / "cluster_profiles.csv", index=False)
    print("Saved EDA figures, cluster diagnostics, and data/processed/player_features_with_archetypes.csv")
    print(diagnostics.to_string(index=False))

    # =================================================================
    # Extended EDA, statistical diagnostics, and scaled-variant exports
    # =================================================================
    print("\nClass-conditional boxplots ...")
    make_inducted_vs_not_boxplots(out, FIGURES)

    print("Annotated scatter plots with Pearson r ...")
    make_annotated_scatter(out, FIGURES, TABLES)

    print("HoF rate by era with 95% Wilson CI ...")
    make_hof_rate_with_ci_by_era(out, FIGURES, TABLES)

    print("Hypothesis tests (KS, Mann-Whitney, chi-squared) ...")
    ht = make_hypothesis_tests(out, TABLES)
    if not ht.empty:
        print(ht[["feature", "ks_p_value", "mw_p_value"]].to_string(index=False))

    print("Robust Z-score anomaly detection on peak_OPS_plus ...")
    robust_z_anomalies(out, stat_col="peak_OPS_plus",
                       threshold=3.5, tables_dir=TABLES)
    if "peak_ERA_plus_proxy" in out.columns:
        print("Robust Z-score anomaly detection on peak_ERA_plus_proxy ...")
        robust_z_anomalies(out, stat_col="peak_ERA_plus_proxy",
                           threshold=3.5, tables_dir=TABLES)

    print("VIF analysis on curated career features ...")
    vif_table = compute_vif(out, tables_dir=TABLES)
    if not vif_table.empty:
        print(vif_table.head(10).to_string(index=False))

    print("Exporting 4 scaled variants of the modeling matrix ...")
    paths = export_scaled_variants(out, output_dir=DATA_PROCESSED)
    for k, v in paths.items():
        print(f"  {k:15s} -> {v}")

    print("\nExtended EDA and diagnostic tables written.")
