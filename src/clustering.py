from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

CLUSTER_FEATURES = [
    "bat_PA", "bat_HR", "bat_HR_rate", "bat_OPS", "bat_BA", "bat_OBP", "bat_SLG", "bat_SB", "bat_RBI",
    "pit_IP", "pit_W", "pit_SV", "pit_SO", "pit_ERA", "pit_WHIP", "pit_K9", "pit_start_share", "pit_save_share",
    "career_span_years", "n_mlb_seasons", "height", "weight", "bmi", "fld_fielding_pct", "primary_position_games",
    "award_total", "allstar_games"
]

def available_cluster_features(df: pd.DataFrame) -> list[str]:
    return [c for c in CLUSTER_FEATURES if c in df.columns]

def fit_archetypes(df: pd.DataFrame, k_range=range(3, 9), random_state: int = 42):
    pool = df[df["model_eligible_pool"] == 1].copy()
    features = available_cluster_features(pool)
    X_raw = pool[features].replace([np.inf, -np.inf], np.nan)
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X = prep.fit_transform(X_raw)
    rows = []
    best_k = None
    best_score = -np.inf
    best_model = None
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, max_iter=300, random_state=random_state)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels, sample_size=min(1000, X.shape[0]), random_state=random_state)
        db = davies_bouldin_score(X, labels)
        rows.append({"k": k, "silhouette": sil, "davies_bouldin": db, "inertia": km.inertia_})
        if sil > best_score:
            best_score, best_k = sil, k
            best_model = km
    final_km = best_model
    km_labels = final_km.labels_
    # Hierarchical clustering is used as a robustness check on the same scaled feature space.
    ag_labels = AgglomerativeClustering(n_clusters=best_k, linkage="ward").fit_predict(X)
    pca = PCA(n_components=2, random_state=random_state)
    coords = pca.fit_transform(X)
    pool["cluster_id"] = km_labels
    pool["hier_cluster_id"] = ag_labels
    pool["pca1"] = coords[:, 0]
    pool["pca2"] = coords[:, 1]
    # Distance-to-centroid features: useful supervised features beyond hard cluster ID.
    distances = final_km.transform(X)
    for j in range(distances.shape[1]):
        pool[f"cluster_distance_{j}"] = distances[:, j]
    labels_map = name_clusters(pool)
    pool["cluster_label"] = pool["cluster_id"].map(labels_map)
    out = df.merge(pool[["playerID", "cluster_id", "hier_cluster_id", "pca1", "pca2", "cluster_label"] + [f"cluster_distance_{j}" for j in range(distances.shape[1])]], on="playerID", how="left")
    diagnostics = pd.DataFrame(rows)
    return out, diagnostics, final_km, prep, pca, features

def name_clusters(pool: pd.DataFrame) -> dict[int, str]:
    labels = {}
    summaries = pool.groupby("cluster_id").agg(
        n=("playerID", "count"),
        hitter_share=("primary_role", lambda s: (s == "Hitter").mean()),
        pitcher_share=("primary_role", lambda s: (s == "Pitcher").mean()),
        hr=("bat_HR", "median"),
        ops=("bat_OPS", "median"),
        sb=("bat_SB", "median"),
        ip=("pit_IP", "median"),
        sv=("pit_SV", "median"),
        w=("pit_W", "median"),
        career=("career_span_years", "median"),
    ).reset_index()
    for _, r in summaries.iterrows():
        if r["pitcher_share"] > 0.65 and r["sv"] >= max(40, summaries["sv"].quantile(0.70)):
            label = "Dominant Reliever / Closer"
        elif r["pitcher_share"] > 0.65 and r["ip"] >= summaries["ip"].quantile(0.70):
            label = "Workhorse Starting Pitcher"
        elif r["hitter_share"] > 0.55 and r["hr"] >= summaries["hr"].quantile(0.75):
            label = "Power Hitter"
        elif r["hitter_share"] > 0.55 and r["sb"] >= summaries["sb"].quantile(0.75):
            label = "Speed / Contact Profile"
        elif r["hitter_share"] > 0.55 and r["ops"] >= summaries["ops"].quantile(0.65):
            label = "On-base / Contact Hitter"
        elif r["career"] >= summaries["career"].quantile(0.70):
            label = "Long-Career Compiler"
        else:
            label = "Role Player / Low-Volume Profile"
        labels[int(r["cluster_id"])] = label
    # Ensure labels are unique enough for display.
    counts = {}
    unique = {}
    for k, v in labels.items():
        counts[v] = counts.get(v, 0) + 1
        unique[k] = v if counts[v] == 1 else f"{v} {counts[v]}"
    return unique
