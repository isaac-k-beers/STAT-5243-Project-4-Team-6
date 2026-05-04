from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, brier_score_loss, log_loss, confusion_matrix
)

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None

EXCLUDE_FEATURES = {
    "playerID", "player_name", "nameFirst", "nameLast", "inducted", "induction_year",
    "is_eligible", "model_eligible_pool", "hof_ballot_rows", "hof_first_ballot_year",
    "hof_last_ballot_year", "eligibility_year", "first_year", "last_year", "final_year",
    "debut_year"  # era is represented separately; raw year is not used as a direct target-time feature.
}


def split_features(df: pd.DataFrame):
    pool = df[df["model_eligible_pool"] == 1].copy()
    y = pool["inducted"].astype(int)
    feature_cols = [c for c in pool.columns if c not in EXCLUDE_FEATURES]
    X = pool[feature_cols].copy()
    # Convert object columns with mixed values to strings for stable one-hot encoding.
    for c in X.select_dtypes(include=["object", "category"]).columns:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    return pool, X, y, feature_cols


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    return ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols)
    ])


def metric_dict(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (proba >= threshold).astype(int)
    out = {
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "brier_score": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1]),
    }
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    order = np.argsort(proba)[::-1]
    for k in [10, 25, 50]:
        kk = min(k, len(order))
        out[f"top_{k}_precision"] = float(np.mean(np.array(y_true)[order[:kk]])) if kk else np.nan
    return out



class ManualSearch:
    """Small deterministic grid search used to keep the project reproducible and fast."""
    def __init__(self, estimator, param_grid: dict, cv, scoring: str = "average_precision"):
        from itertools import product
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.scoring = scoring
        keys = list(param_grid.keys())
        self.param_list = [dict(zip(keys, values)) for values in product(*[param_grid[k] for k in keys])]

    def fit(self, X, y):
        from sklearn.base import clone
        from sklearn.model_selection import cross_val_score
        rows = []
        best_score = -np.inf
        best_params = None
        best_est = None
        for params in self.param_list:
            est = clone(self.estimator).set_params(**params)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring, n_jobs=1)
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            rows.append({"params": params, "mean_test_score": mean, "std_test_score": std})
            if mean > best_score:
                best_score = mean
                best_params = params
                best_est = clone(self.estimator).set_params(**params)
        best_est.fit(X, y)
        # Add ranks.
        ordered = sorted(enumerate(rows), key=lambda x: x[1]["mean_test_score"], reverse=True)
        ranks = [None] * len(rows)
        for rank, (idx, _) in enumerate(ordered, start=1):
            ranks[idx] = rank
        for row, rank in zip(rows, ranks):
            row["rank_test_score"] = rank
        self.cv_results_ = rows
        self.best_score_ = best_score
        self.best_params_ = best_params
        self.best_estimator_ = best_est
        return self

    def predict_proba(self, X):
        return self.best_estimator_.predict_proba(X)


def build_model_searches(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42):
    prep = make_preprocessor(X_train)
    pos = y_train.sum()
    neg = len(y_train) - pos
    scale_pos_weight = float(neg / max(pos, 1))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    searches = {}
    # 1) Logistic regression baseline.
    logit = Pipeline([
        ("prep", prep),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"))
    ])
    searches["logistic_regression"] = ManualSearch(
        logit,
        {"model__C": [0.01, 0.1, 1.0, 10.0]},
        cv=cv, scoring="average_precision"
    )
    # 2) Random forest.
    rf = Pipeline([
        ("prep", prep),
        ("model", RandomForestClassifier(random_state=random_state, class_weight="balanced_subsample", n_jobs=1))
    ])
    searches["random_forest"] = ManualSearch(
        rf,
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 4],
            "model__max_features": ["sqrt"],
        },
        cv=cv, scoring="average_precision"
    )
    # 3) Gradient boosting with proper hyperparameter search.
    gb = Pipeline([
        ("prep", prep),
        ("model", GradientBoostingClassifier(random_state=random_state))
    ])
    searches["gradient_boosting"] = ManualSearch(
        gb,
        {
            "model__n_estimators": [50, 100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        cv=cv, scoring="average_precision"
    )
    # 4) XGBoost with proper hyperparameter search.
    if XGBClassifier is not None:
        xgb = Pipeline([
            ("prep", prep),
            ("model", XGBClassifier(
                random_state=random_state, objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", n_jobs=1, scale_pos_weight=scale_pos_weight
            ))
        ])
        searches["xgboost"] = ManualSearch(
            xgb,
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
                "model__colsample_bytree": [0.8, 1.0],
                "model__reg_lambda": [1.0],
            },
            cv=cv, scoring="average_precision"
        )
    return searches, cv

def make_stacking_model(best_estimators: dict, random_state: int = 42):
    # Stacking uses already tuned, diverse base learners. The meta learner is regularized logistic regression.
    candidates = []
    for name in ["logistic_regression", "random_forest", "gradient_boosting", "xgboost", "lightgbm"]:
        if name in best_estimators:
            candidates.append((name, best_estimators[name]))
    if len(candidates) < 2:
        return None
    return StackingClassifier(
        estimators=candidates,
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        stack_method="predict_proba",
        cv=5,
        n_jobs=1,
        passthrough=False
    )
