from pathlib import Path
import sys, json
sys.path.append(str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from src.config import DATA_PROCESSED, MODELS, TABLES, FIGURES, RANDOM_STATE, HOLDOUT_ELIGIBILITY_YEAR
from src.models import split_features, build_model_searches, metric_dict, make_stacking_model
from src.evaluate import save_curves, stratified_metrics_table
from src.diagnostics import cooks_distance_logistic

if __name__ == "__main__":
    data_path = DATA_PROCESSED / "player_features_with_archetypes.csv"
    if not data_path.exists():
        raise SystemExit("Run scripts/02_eda_and_archetypes.py first.")
    df = pd.read_csv(data_path, low_memory=False)
    pool, X, y, feature_cols = split_features(df)
    MAX_OBSERVED_ELIGIBILITY_YEAR = 2026
    valid_year_mask = pool["eligibility_year"] <= MAX_OBSERVED_ELIGIBILITY_YEAR
    if (~valid_year_mask).sum() > 0:
        print(
            f"Dropping {(~valid_year_mask).sum()} players with eligibility_year "
            f"> {MAX_OBSERVED_ELIGIBILITY_YEAR} to avoid labeling future-eligible players as negatives."
        )

    pool = pool.loc[valid_year_mask].reset_index(drop=True)
    X = X.loc[valid_year_mask].reset_index(drop=True)
    y = y.loc[valid_year_mask].reset_index(drop=True)

    leaky_archetype_cols = [
    c for c in X.columns
    if c in ["cluster_id", "hier_cluster_id", "pca1", "pca2", "cluster_label"]
    or c.startswith("cluster_distance_")
]

if leaky_archetype_cols:
    print("Dropping pre-split archetype features to avoid leakage:")
    print(leaky_archetype_cols)
    X = X.drop(columns=leaky_archetype_cols)
    feature_cols = [c for c in feature_cols if c not in leaky_archetype_cols]
    holdout_mask = pool["eligibility_year"] >= HOLDOUT_ELIGIBILITY_YEAR
    # Fallback for unusual datasets; not triggered for this Kaggle dataset.
    if holdout_mask.sum() < 100 or y[holdout_mask].sum() < 5:
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(np.arange(len(pool)), test_size=0.25, stratify=y, random_state=RANDOM_STATE)
        train_mask = np.zeros(len(pool), dtype=bool); train_mask[train_idx] = True
        holdout_mask = np.zeros(len(pool), dtype=bool); holdout_mask[test_idx] = True
        split_note = "Stratified random holdout fallback"
    else:
        train_mask = ~holdout_mask
        split_note = f"Time-based holdout: train eligibility_year < {HOLDOUT_ELIGIBILITY_YEAR}; test eligibility_year >= {HOLDOUT_ELIGIBILITY_YEAR}"
    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_test, y_test = X.loc[holdout_mask], y.loc[holdout_mask]
    test_pool = pool.loc[holdout_mask].copy()
    MODELS.mkdir(parents=True, exist_ok=True); TABLES.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    # ---------------------------------------------------------------
    # Influence diagnostics on a Logistic Regression baseline.
    # Identifies borderline / controversial HoF candidates whose
    # induction status disproportionately drives linear-model fit.
    # ---------------------------------------------------------------
    print("Computing Cook's distance + leverage diagnostics on LR baseline ...")
    try:
        ids_train = pool.loc[train_mask, "player_name"] if "player_name" in pool.columns else None
        infl_df = cooks_distance_logistic(
            X_train, y_train, ids=ids_train,
            tables_dir=TABLES, figures_dir=FIGURES, top_k_features=20)
        if not infl_df.empty and "error" not in infl_df.columns:
            n_inf = int(infl_df.get("is_influential", pd.Series([0])).sum())
            print(f"  -> flagged {n_inf} influential players "
                  f"(see logistic_influence_diagnostics.csv)")
    except Exception as exc:  # noqa: BLE001
        print(f"  influence diagnostics failed: {exc}; continuing.")

    searches, cv = build_model_searches(X_train, y_train, RANDOM_STATE)
    best_estimators = {}
    metrics_rows = []
    cv_rows = []
    for name, search in searches.items():
        print(f"Training {name} ...")
        search.fit(X_train, y_train)
        best_estimators[name] = search.best_estimator_
        proba = search.predict_proba(X_test)[:, 1]
        m = metric_dict(y_test.to_numpy(), proba)
        m.update({"model": name, "best_cv_pr_auc": float(search.best_score_), "best_params": json.dumps(search.best_params_, default=str)})
        metrics_rows.append(m)
        cv_result = pd.DataFrame(search.cv_results_)
        cv_result["model"] = name
        cv_rows.append(cv_result[["model", "rank_test_score", "mean_test_score", "std_test_score", "params"]])
        joblib.dump(search.best_estimator_, MODELS / f"{name}.joblib")
        print(f"Finished {name}: holdout PR AUC={m['pr_auc']:.3f}, ROC AUC={m['roc_auc']:.3f}")
    # Stacking ensemble. Re-uses the best estimators above as base learners,
    # with regularized logistic regression as the meta-learner. This gives us
    # a fifth supervised model and clearly demonstrates Lecture 8 Part II.
    print("Training stacking ensemble ...")
    stacking = make_stacking_model(best_estimators, RANDOM_STATE)
    if stacking is not None:
        try:
            stacking.fit(X_train, y_train)
            best_estimators["stacking"] = stacking
            proba = stacking.predict_proba(X_test)[:, 1]
            m = metric_dict(y_test.to_numpy(), proba)
            m.update({"model": "stacking", "best_cv_pr_auc": float("nan"),
                      "best_params": json.dumps({"base_learners": list(best_estimators.keys())})})
            metrics_rows.append(m)
            joblib.dump(stacking, MODELS / "stacking.joblib")
            print(f"Finished stacking: holdout PR AUC={m['pr_auc']:.3f}, ROC AUC={m['roc_auc']:.3f}")
        except Exception as exc:  # noqa: BLE001
            print(f"Stacking ensemble failed ({type(exc).__name__}: {exc}); continuing without it.")

    print("Building model comparison tables ...")
    metrics = pd.DataFrame(metrics_rows).sort_values(["pr_auc", "roc_auc"], ascending=False)
    metrics.to_csv(TABLES / "model_comparison_holdout.csv", index=False)
    if cv_rows:
        pd.concat(cv_rows, ignore_index=True).to_csv(TABLES / "cv_search_results.csv", index=False)
    best_name = metrics.iloc[0]["model"]
    best_model = best_estimators[best_name]
    joblib.dump(best_model, MODELS / "best_holdout_model.joblib")
    print("Saving holdout predictions and figures ...")
    # Save holdout predictions and figures.
    holdout_proba = best_model.predict_proba(X_test)[:, 1]
    holdout_pred = test_pool[["playerID", "player_name", "primary_role", "debut_era", "cluster_label", "inducted", "eligibility_year"]].copy()
    holdout_pred["predicted_probability"] = holdout_proba
    holdout_pred["predicted_label_050"] = (holdout_proba >= 0.5).astype(int)
    holdout_pred.sort_values("predicted_probability", ascending=False).to_csv(TABLES / "holdout_predictions_best_model.csv", index=False)
    save_curves(y_test.to_numpy(), holdout_proba, FIGURES, prefix="best_holdout")
    stratified_metrics_table(holdout_pred).to_csv(TABLES / "stratified_holdout_summary.csv", index=False)
    print("Computing fast global feature importance ...")
    # Fast global interpretation: model-native feature importance or coefficients.
    try:
        prep = best_model.named_steps.get("prep")
        model = best_model.named_steps.get("model")
        try:
            names = prep.get_feature_names_out()
        except Exception:
            names = feature_cols
        if hasattr(model, "feature_importances_"):
            vals = model.feature_importances_
            label = "model_native_importance"
        elif hasattr(model, "coef_"):
            vals = abs(model.coef_).ravel()
            label = "absolute_coefficient"
        else:
            vals = []
            label = "importance"
        if len(vals):
            imp = pd.DataFrame({"feature": list(names)[:len(vals)], label: vals}).sort_values(label, ascending=False)
            imp.to_csv(TABLES / "feature_importance_best_model.csv", index=False)
            top = imp.head(20).iloc[::-1]
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 6)); plt.barh(top["feature"], top[label]); plt.xlabel(label); plt.title("Global Feature Importance")
            plt.tight_layout(); plt.savefig(FIGURES / "best_holdout_feature_importance.png", dpi=160); plt.close()
    except Exception as e:
        (TABLES / "feature_importance_error.txt").write_text(str(e))

    print("Refitting selected model on all eligible players for app/demo ...")
    # Refit selected model type on all eligible players for the app/demo and future/borderline ranking.
    final_model = clone(best_model)
    final_model.fit(X, y)
    joblib.dump(final_model, MODELS / "final_model_retrained_all_eligible.joblib")
    all_proba = final_model.predict_proba(X)[:, 1]
    app_df = pool[["playerID", "player_name", "primary_role", "primary_position", "debut_era", "debut_year", "final_year", "eligibility_year", "cluster_id", "cluster_label", "inducted", "bat_HR", "bat_H", "bat_OPS", "pit_W", "pit_SV", "pit_ERA", "award_total", "allstar_games"]].copy()
    app_df["predicted_probability"] = all_proba
    app_df["predicted_label_050"] = (all_proba >= 0.5).astype(int)
    app_df.sort_values("predicted_probability", ascending=False).to_csv(DATA_PROCESSED / "model_predictions_all_eligible.csv", index=False)
    # Also place app CSV where Shiny can find it.
    app_dir = Path(__file__).resolve().parents[1] / "app"
    app_dir.mkdir(exist_ok=True)
    app_df.sort_values("predicted_probability", ascending=False).to_csv(app_dir / "model_predictions.csv", index=False)
    summary = {
        "split_note": split_note,
        "train_n": int(train_mask.sum()), "test_n": int(holdout_mask.sum()),
        "train_positives": int(y_train.sum()), "test_positives": int(y_test.sum()),
        "best_model": str(best_name),
        "best_holdout_pr_auc": float(metrics.iloc[0]["pr_auc"]),
        "best_holdout_roc_auc": float(metrics.iloc[0]["roc_auc"]),
        "features_used": len(feature_cols)
    }
    print("Writing training summary ...")
    (TABLES / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print("Training complete.")
    print(json.dumps(summary, indent=2))
    print(metrics[["model", "pr_auc", "roc_auc", "precision", "recall", "f1", "balanced_accuracy", "brier_score", "top_25_precision"]].to_string(index=False))
