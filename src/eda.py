from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# =====================================================================
# BASIC SUMMARY STATISTICS — Project 4 Rubric 2 (EDA) Advanced tier
# =====================================================================
SUMMARY_NUMERIC_FEATURES = [
    "n_mlb_seasons", "career_span_years", "debut_age", "final_age",
    "bat_PA", "bat_AB", "bat_H", "bat_HR", "bat_RBI", "bat_SB",
    "bat_BA", "bat_OBP", "bat_SLG", "bat_OPS",
    "bat_mean_OPS_plus_proxy", "bat_max_OPS_plus_proxy",
    "pit_IP", "pit_W", "pit_L", "pit_SO", "pit_SV",
    "pit_ERA", "pit_WHIP", "pit_K9",
    "pit_mean_ERA_plus_proxy", "pit_max_ERA_plus_proxy",
    "award_total", "allstar_games", "salary_total",
    "height", "weight", "bmi",
]


def make_summary_statistics(df: pd.DataFrame, tables_dir: Path) -> dict:
    """Produce basic descriptive statistics required for Rubric 2 Advanced tier.

    Outputs:
        - tables_dir/summary_overall.csv   — overall describe() across all eligible players
        - tables_dir/summary_by_induction.csv — split by inducted vs not-inducted
        - tables_dir/summary_by_role.csv   — split by primary_role
        - tables_dir/summary_by_era.csv    — split by debut_era
        - tables_dir/categorical_counts.csv — value counts for the key categoricals
    Returns a dict of the produced DataFrames for inline use in notebooks.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)
    pool = df[df["model_eligible_pool"] == 1].copy()
    feats = [c for c in SUMMARY_NUMERIC_FEATURES if c in pool.columns]

    out = {}

    # 1) Overall summary statistics (mean / median / std / quantiles / missing)
    overall = pool[feats].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    overall["missing_rate"] = pool[feats].isna().mean()
    overall["missing_count"] = pool[feats].isna().sum()
    overall.to_csv(tables_dir / "summary_overall.csv")
    out["overall"] = overall

    # 2) Split by induction status
    by_ind = (pool.groupby("inducted")[feats]
                  .agg(["count", "mean", "median", "std"])
                  .round(3))
    by_ind.to_csv(tables_dir / "summary_by_induction.csv")
    out["by_induction"] = by_ind

    # 3) Split by primary role
    if "primary_role" in pool.columns:
        by_role = (pool.groupby("primary_role")[feats]
                       .agg(["count", "mean", "median"])
                       .round(3))
        by_role.to_csv(tables_dir / "summary_by_role.csv")
        out["by_role"] = by_role

    # 4) Split by debut era
    if "debut_era" in pool.columns:
        by_era = (pool.groupby("debut_era")[feats]
                      .agg(["count", "mean", "median"])
                      .round(3))
        by_era.to_csv(tables_dir / "summary_by_era.csv")
        out["by_era"] = by_era

    # 5) Categorical counts
    cat_rows = []
    for col in ["primary_role", "primary_position", "bats", "throws",
                "international_flag", "debut_era"]:
        if col not in pool.columns:
            continue
        vc = pool[col].value_counts(dropna=False).head(20)
        for value, count in vc.items():
            cat_rows.append({"column": col, "value": value, "count": int(count),
                             "share": round(count / len(pool), 4)})
    cat_df = pd.DataFrame(cat_rows)
    cat_df.to_csv(tables_dir / "categorical_counts.csv", index=False)
    out["categorical"] = cat_df

    # 6) Headline KPIs (single-row table for the report)
    kpis = pd.DataFrame([{
        "n_players_total":          int(len(df)),
        "n_eligible_players":       int(len(pool)),
        "n_inducted":               int(pool["inducted"].sum()),
        "induction_rate":           round(float(pool["inducted"].mean()), 4),
        "year_range_min":           int(pool["debut_year"].min()) if "debut_year" in pool else None,
        "year_range_max":           int(pool["final_year"].max()) if "final_year" in pool else None,
        "median_career_seasons":    float(pool["n_mlb_seasons"].median()),
        "median_career_span_years": float(pool["career_span_years"].median()),
        "n_pitchers":               int((pool["primary_role"] == "Pitcher").sum()) if "primary_role" in pool else None,
        "n_hitters":                int((pool["primary_role"] == "Hitter").sum()) if "primary_role" in pool else None,
        "n_two_way":                int((pool["primary_role"] == "Two-Way").sum()) if "primary_role" in pool else None,
    }])
    kpis.to_csv(tables_dir / "summary_kpis.csv", index=False)
    out["kpis"] = kpis

    return out

def make_eda_outputs(df: pd.DataFrame, figures_dir: Path, tables_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    pool = df[df["model_eligible_pool"] == 1].copy()
    # Class imbalance.
    counts = pool["inducted"].value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    plt.bar(["Not inducted", "Inducted"], [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Hall of Fame Target Class Imbalance")
    plt.ylabel("Eligible players")
    save_fig(figures_dir / "01_class_imbalance.png")
    # HoF rate by era.
    era = pool.groupby("debut_era").agg(players=("playerID", "count"), hof_rate=("inducted", "mean")).reset_index()
    era = era[era["players"] >= 20].sort_values("debut_era")
    era.to_csv(tables_dir / "hof_rate_by_era.csv", index=False)
    plt.figure(figsize=(9, 4))
    plt.bar(era["debut_era"], era["hof_rate"])
    plt.xticks(rotation=35, ha="right")
    plt.title("Hall of Fame Induction Rate by Debut Era")
    plt.ylabel("Induction rate")
    save_fig(figures_dir / "02_hof_rate_by_era.png")
    # Role distribution.
    role = pool.groupby(["primary_role", "inducted"]).size().unstack(fill_value=0)
    role.to_csv(tables_dir / "hof_counts_by_role.csv")
    plt.figure(figsize=(7, 4))
    x = np.arange(len(role.index))
    plt.bar(x - 0.2, role.get(0, 0), width=0.4, label="Not inducted")
    plt.bar(x + 0.2, role.get(1, 0), width=0.4, label="Inducted")
    plt.xticks(x, role.index, rotation=20, ha="right")
    plt.ylabel("Players")
    plt.title("Eligible Players by Role and Induction Status")
    plt.legend()
    save_fig(figures_dir / "03_role_by_induction.png")
    # Missingness.
    miss = df.isna().mean().sort_values(ascending=False).head(25).reset_index()
    miss.columns = ["feature", "missing_rate"]
    miss.to_csv(tables_dir / "top_missingness.csv", index=False)
    plt.figure(figsize=(8, 6))
    plt.barh(miss["feature"][::-1], miss["missing_rate"][::-1])
    plt.xlabel("Missing rate")
    plt.title("Top 25 Missingness Rates")
    save_fig(figures_dir / "04_missingness.png")
    # Correlation heatmap for selected numeric features.
    selected = [c for c in ["bat_HR", "bat_H", "bat_OPS", "bat_SB", "pit_W", "pit_SV", "pit_SO", "pit_ERA", "award_total", "allstar_games", "career_span_years", "bmi", "inducted"] if c in pool.columns]
    corr = pool[selected].corr(numeric_only=True)
    corr.to_csv(tables_dir / "selected_correlation_matrix.csv")
    plt.figure(figsize=(8, 7))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.index)), corr.index)
    plt.title("Selected Feature Correlation Heatmap")
    save_fig(figures_dir / "05_correlation_heatmap.png")
    # Distributions of key achievements by induction.
    for feature in ["bat_HR", "bat_OPS", "pit_W", "pit_SV", "award_total", "allstar_games"]:
        if feature in pool.columns:
            plt.figure(figsize=(7, 4))
            a = pool.loc[pool.inducted == 0, feature].replace([np.inf, -np.inf], np.nan).dropna()
            b = pool.loc[pool.inducted == 1, feature].replace([np.inf, -np.inf], np.nan).dropna()
            plt.hist(a, bins=35, alpha=0.5, label="Not inducted")
            plt.hist(b, bins=35, alpha=0.5, label="Inducted")
            plt.title(f"Distribution of {feature} by Induction Status")
            plt.legend()
            save_fig(figures_dir / f"dist_{feature}.png")


# =====================================================================
# Extended EDA helpers — class-conditional comparisons, annotated
# scatter plots, era rates with confidence intervals, hypothesis tests,
# and career-trajectory visualizations.
# Each plot is one beat in "what makes a Hall of Famer".
# =====================================================================
def make_inducted_vs_not_boxplots(df: pd.DataFrame, figures_dir: Path) -> None:
    """Boxplots of key features split by induction status (P1-style)."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    pool = df[df["model_eligible_pool"] == 1].copy()
    candidates = [
        ("bat_HR", "Career Home Runs"),
        ("bat_H", "Career Hits"),
        ("pit_W", "Career Pitcher Wins"),
        ("pit_SO", "Career Pitcher Strikeouts"),
        ("award_mvp_count", "MVP Awards"),
        ("allstar_games", "Career All-Star Selections"),
        ("bat_max_OPS_plus_proxy", "Peak OPS+ (era-adjusted)"),
        ("n_mlb_seasons", "Career Length (seasons)"),
    ]
    panels = [(f, label) for f, label in candidates if f in pool.columns]
    if not panels:
        return
    n = len(panels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.0))
    axes = axes.flatten() if n > 1 else [axes]
    for ax, (f, label) in zip(axes, panels):
        data = [
            pool.loc[pool["inducted"] == 0, f].replace([np.inf, -np.inf], np.nan).dropna().values,
            pool.loc[pool["inducted"] == 1, f].replace([np.inf, -np.inf], np.nan).dropna().values,
        ]
        ax.boxplot(data, labels=["Not", "Inducted"], showfliers=True)
        ax.set_title(label, fontsize=10)
    for ax in axes[len(panels):]:
        ax.axis("off")
    plt.suptitle("Class-conditional distributions of key HoF predictors",
                 fontweight="bold")
    save_fig(figures_dir / "09_class_conditional_boxplots.png")


def make_annotated_scatter(df: pd.DataFrame, figures_dir: Path,
                           tables_dir: Path) -> None:
    """Annotated scatter plots with Pearson r and p-value."""
    from scipy.stats import pearsonr
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    pool = df[df["model_eligible_pool"] == 1].copy()
    pairs = [
        ("bat_HR", "award_mvp_count", "Career HR vs MVP awards"),
        ("bat_max_OPS_plus_proxy", "allstar_games", "Peak OPS+ vs All-Star total"),
        ("n_mlb_seasons", "bat_H", "Career length vs total hits"),
    ]
    rows = []
    n_panels = len(pairs)
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4.5, 4))
    if n_panels == 1:
        axes = [axes]
    for ax, (x, y, title) in zip(axes, pairs):
        if x not in pool.columns or y not in pool.columns:
            ax.axis("off"); continue
        d = pool[[x, y, "inducted"]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(d) < 30:
            ax.axis("off"); continue
        try:
            r, p = pearsonr(d[x], d[y])
        except Exception:
            r, p = np.nan, np.nan
        colors = d["inducted"].map({0: "#888", 1: "#C62828"})
        ax.scatter(d[x], d[y], s=10, alpha=0.4, c=colors)
        ax.set_xlabel(x); ax.set_ylabel(y)
        ax.set_title(f"{title}\n(r = {r:.3f}, p = {p:.2e})", fontsize=10)
        rows.append({"x": x, "y": y, "r": r, "p_value": p, "n": len(d)})
    plt.tight_layout()
    save_fig(figures_dir / "10_annotated_scatter.png")
    if rows:
        pd.DataFrame(rows).to_csv(tables_dir / "annotated_scatter_correlations.csv", index=False)


def make_hof_rate_with_ci_by_era(df: pd.DataFrame, figures_dir: Path,
                                  tables_dir: Path) -> None:
    """HoF induction rate by debut era WITH 95% Wilson confidence intervals."""
    from scipy import stats
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    pool = df[df["model_eligible_pool"] == 1].copy()
    if "debut_era" not in pool.columns:
        return
    grp = pool.groupby("debut_era").agg(
        n=("playerID", "count"),
        n_inducted=("inducted", "sum"),
    ).reset_index()
    grp["rate"] = grp["n_inducted"] / grp["n"]
    # Wilson 95% CI (correct formula).
    z = 1.96
    n = grp["n"].astype(float)
    p_hat = grp["rate"].astype(float)
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    grp["ci_low"] = (centre - spread).clip(lower=0)
    grp["ci_high"] = (centre + spread).clip(upper=1)
    # Guarantee non-negative error bars for matplotlib.
    grp["err_low"] = (grp["rate"] - grp["ci_low"]).clip(lower=0)
    grp["err_high"] = (grp["ci_high"] - grp["rate"]).clip(lower=0)
    grp.to_csv(tables_dir / "hof_rate_with_ci_by_era.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(grp))
    ax.bar(x, grp["rate"], color="#2E7D32", alpha=0.85)
    ax.errorbar(x, grp["rate"],
                yerr=[grp["err_low"].values, grp["err_high"].values],
                fmt="none", ecolor="black", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(grp["debut_era"], rotation=20, ha="right")
    ax.set_ylabel("Induction rate")
    ax.set_title("HoF induction rate by debut era (with 95% Wilson CI)")
    save_fig(figures_dir / "11_hof_rate_with_ci_by_era.png")


def make_hypothesis_tests(df: pd.DataFrame, tables_dir: Path) -> pd.DataFrame:
    """KS, Mann-Whitney, and chi-squared tests reported as a table."""
    from scipy import stats
    tables_dir.mkdir(parents=True, exist_ok=True)
    pool = df[df["model_eligible_pool"] == 1].copy()

    rows = []
    numeric_cols = [
        "bat_HR", "bat_H",
        "pit_W", "pit_SO",
        "bat_max_OPS_plus_proxy", "award_mvp_count", "allstar_games",
        "n_mlb_seasons",
    ]
    for col in numeric_cols:
        if col not in pool.columns:
            continue
        a = pool.loc[pool["inducted"] == 0, col].dropna()
        b = pool.loc[pool["inducted"] == 1, col].dropna()
        if len(a) < 5 or len(b) < 5:
            continue
        try:
            ks_stat, ks_p = stats.ks_2samp(a, b)
        except Exception:
            ks_stat, ks_p = np.nan, np.nan
        try:
            mw_stat, mw_p = stats.mannwhitneyu(a, b, alternative="two-sided")
        except Exception:
            mw_stat, mw_p = np.nan, np.nan
        rows.append({
            "feature": col,
            "n_not_inducted": len(a),
            "n_inducted": len(b),
            "ks_statistic": round(ks_stat, 4) if not np.isnan(ks_stat) else None,
            "ks_p_value": ks_p,
            "mw_statistic": mw_stat,
            "mw_p_value": mw_p,
        })

    # Chi-squared: induction vs primary_position
    cat_rows = []
    for cat_col in ["primary_role", "primary_position", "debut_era"]:
        if cat_col not in pool.columns:
            continue
        ct = pd.crosstab(pool[cat_col], pool["inducted"])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            continue
        try:
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            cat_rows.append({
                "categorical": cat_col,
                "chi2_statistic": round(chi2, 4),
                "p_value": p,
                "degrees_of_freedom": dof,
                "n_levels": ct.shape[0],
            })
        except Exception:
            continue

    out_num = pd.DataFrame(rows)
    out_cat = pd.DataFrame(cat_rows)
    out_num.to_csv(tables_dir / "hypothesis_tests_numeric.csv", index=False)
    out_cat.to_csv(tables_dir / "hypothesis_tests_categorical.csv", index=False)
    return out_num


def make_career_trajectory_plots(player_year_df: pd.DataFrame,
                                  figures_dir: Path,
                                  legends: list = None) -> None:
    """Career-trajectory plots for famous players (presentation-ready).

    Storytelling: "Babe Ruth's 1921 isn't an outlier — it's the signal."
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    if legends is None:
        legends = ["ruthba01", "mayswi01", "aaronha01", "koufasa01",
                   "trouutmi01", "pujolal01"]
    if "playerID" not in player_year_df.columns or "yearID" not in player_year_df.columns:
        return
    df = player_year_df[player_year_df["playerID"].isin(legends)].copy()
    if df.empty:
        return

    plot_metric = None
    for cand in ["OPS_plus_proxy", "OPS_z", "OPS"]:
        if cand in df.columns:
            plot_metric = cand
            break
    if plot_metric is None:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    for pid, g in df.groupby("playerID"):
        g = g.sort_values("yearID")
        name = (g["nameFirst"].iloc[0] + " " + g["nameLast"].iloc[0]
                if "nameFirst" in g.columns else pid)
        ax.plot(g["yearID"], g[plot_metric], marker="o", label=name, alpha=0.85)
    ax.axhline(100, ls=":", color="grey", label="League average (100)")
    ax.set_xlabel("Season")
    ax.set_ylabel(plot_metric)
    ax.set_title(f"Career trajectory ({plot_metric}) — selected legends")
    ax.legend(fontsize=8, ncol=2)
    save_fig(figures_dir / "12_career_trajectories.png")
