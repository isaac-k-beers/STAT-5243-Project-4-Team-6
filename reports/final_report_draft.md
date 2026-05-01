# Final Report Draft — MLB Hall of Fame Prediction + Player Archetype Clustering

**STAT 5243 Project 4 (Final) — Team 6.**

## 1. Problem Statement

We ask whether career performance, awards, biographical attributes, era-adjusted statistics, and unsupervised player archetypes can predict whether an eligible MLB player is inducted into the Baseball Hall of Fame. The target is `inducted` (1 = inducted, 0 = eligible but not inducted). The modeling pool is the set of players who satisfy the BBWAA eligibility filter — at least 10 MLB seasons played and retired at least 5 calendar years before the latest available Hall of Fame ballot — plus already-inducted players from earlier eras (older Negro-Leagues / pre-modern inductees who do not satisfy the 10-season modern threshold are still kept as positives). The headline result on a held-out cohort of 1,299 eligible players (49 inducted) is **PR AUC 0.721 / ROC AUC 0.990** with the selected Gradient Boosting classifier, and a top-25 precision of 0.76.

## 2. Data Collection and Preparation

The project combines four sources to satisfy the Advanced data-collection tier. The primary structured source is **SABR / Lahman 2025**: the full 27-CSV release lives in `data/raw/sabr_lahman/`, and the 18 tables actually consumed by the pipeline are reconciled into `data/raw/analysis_input/` (Master ← People, Batting, Pitching, Fielding, FieldingOF, FieldingPost, BattingPost, PitchingPost, SeriesPost, HallOfFame, AwardsPlayers, AwardsSharePlayers, AllstarFull, Salaries, Teams, TeamsFranchises, TeamsHalf, Appearances). The original **Kaggle Baseball Databank** (14 CSVs in `data/raw/kaggle/`) is preserved as an audit / backup copy and is only read when SABR / Lahman files are not available. **pybaseball** contributes a small reproducible supplement — a `playerid_lookup` example (4 rows) and a Statcast pitch-level sample (314 rows for one 2015 game day). A focused **BeautifulSoup scrape** of `https://baseballhall.org/hall-of-fame/future-eligibles` produces 142 rows of currently active or recently retired candidates that the Shiny app uses for live predictions. `reports/tables/part1_source_manifest.csv` records exactly which file came from which source for every reconciled table.

`Appearances.csv` is part of the SABR / Lahman 2025 release and is used as the canonical source for primary-position construction. `src/features.py` aggregates per-position game counts (`G_p`, `G_c`, `G_1b`, …, `G_dh`) and selects the position with the most games for each player; it falls back to `Fielding.csv` only if `Appearances.csv` is unavailable.

The cleaned player-level feature table holds **24,270 players** spanning debut years 1871 through 2025 (Hall of Fame voting through the 2026 ballot). Target construction uses `eligibility_year = final_year + 5` (the BBWAA five-calendar-year rule) and an `is_eligible` flag for `n_mlb_seasons ≥ 10` AND `eligibility_year ≤ max(HallOfFame.yearid)`. The eligible modeling pool contains **3,738 players** of which **281 are inducted**, an induction rate of **7.5%**.

## 3. EDA and Unsupervised Learning

EDA produces six summary-statistics tables (overall, by induction, by role, by era, categorical counts, KPIs) and eleven figures: class imbalance, HoF rate by debut era, role × induction, missingness, correlation heatmap, cluster elbow, cluster silhouette, PCA scatter, class-conditional boxplots, annotated scatter with Pearson r and p-value, and HoF rate by era with 95% Wilson confidence intervals. Six per-stat distribution histograms split by induction status (HR, OPS, Wins, Saves, Awards, All-Star) are saved as `reports/figures/dist_*.png`. Hypothesis tests (KS, Mann-Whitney, chi-squared) all reject the null of equal distribution for the strongest predictors — for example `n_mlb_seasons` KS p ≈ 3 × 10⁻⁸² and Mann-Whitney p ≈ 1 × 10⁻⁹⁰, `allstar_games` KS p ≈ 2.5 × 10⁻⁶⁶, `award_mvp_count` Mann-Whitney p ≈ 9 × 10⁻⁵⁸ — and are saved to `hypothesis_tests_numeric.csv` and `hypothesis_tests_categorical.csv`.

Unsupervised learning is integrated directly into the EDA stage. K-means clustering is fit on a curated set of cluster features (career batting / pitching volume, era-adjusted peaks, career length, BMI, awards, All-Star count) for k in {3, 4, 5, 6, 7}. Hierarchical (Agglomerative) clustering is used as a robustness check on the same scaled feature space, and PCA is applied for two-component visualization. The silhouette score selects **k = 5**.

| k | Silhouette | Davies-Bouldin | Inertia |
|---|---|---|---|
| 3 | 0.225 | 1.621 | 69,955 |
| 4 | 0.229 | 1.604 | 63,641 |
| **5** | **0.233** | **1.463** | 59,234 |
| 6 | 0.150 | 1.635 | 55,749 |
| 7 | 0.172 | 1.559 | 52,413 |

The five resulting archetypes are named by `src/clustering.py: name_clusters` from the cluster centroids and written to `reports/tables/cluster_profiles.csv`. The cluster ID, hierarchical-cluster ID, PCA components, and per-centroid distances are merged back into `player_features_with_archetypes.csv` so the supervised models use the unsupervised structure as features. Robust Z-score (MAD-based) anomaly detection on `peak_OPS_plus` flags era-adjusted statistical extremes — exactly the legends a HoF model must classify correctly.

## 4. Feature Engineering and Preprocessing

The base feature table holds **260 columns** at player level; once the unsupervised cluster columns are merged the table holds approximately **268 columns**. Eight feature groups are organized around HoF-relevant signals:

1. **Eligibility flags** (`n_mlb_seasons`, `eligibility_year`, `is_eligible`, `model_eligible_pool`)
2. **Career batting volume** (counting + rate)
3. **Career pitching volume** (counting + rate)
4. **Rolling-window peaks** — 3-year and 5-year peak HR / H / RBI / OPS for batters, W / SO / SV / IP / ERA-z for pitchers
5. **Era-adjusted features** — OPS+ and ERA+ proxies computed against per-year league benchmarks so a 1968 hitter and a 1999 hitter are directly comparable
6. **Awards / All-Star / postseason aggregates** (MVP / Cy Young / Gold Glove / Silver Slugger / WS-MVP / Triple Crown counts; All-Star games; postseason batting and pitching aggregates)
7. **Biographical** (height, weight, BMI, debut age, debut era, international flag, handedness, cyclical birth-month encoding)
8. **Unsupervised features** (cluster ID, hierarchical-cluster ID, PCA1, PCA2, distances to the first three centroids)

Layered on top: IQR outlier flags (`*_outlier_high`, `*_outlier_low`) computed without capping (because the high tail is precisely the legends the model must identify), Yeo-Johnson power transforms (`*_yj`) for heavy-tailed counting stats, concentration / diversity metrics (`award_entropy`, `award_hhi`, `position_versatility_entropy`, `position_concentration_hhi`), and five baseball-specific interaction terms — `peak_x_longevity`, `allstar_x_pos_scarcity`, `mvp_x_postseason_HR`, `is_pitcher_x_K`, and `steroid_era_x_HR` (HRs accumulated 1994 – 2005, used by the model to learn the historical voter penalty).

Preprocessing inside the modeling pipeline uses scikit-learn `ColumnTransformer`s fitted on training folds only: median imputation and standardization for numeric features, most-frequent imputation and one-hot encoding for categorical features. Four scaled model-matrix variants are exported for downstream model-class-specific scaling: `model_matrix_raw.csv`, `model_matrix_standardized.csv`, `model_matrix_minmax.csv`, `model_matrix_power.csv` (Yeo-Johnson + standardize). After leakage exclusions (HoF voting fields, `eligibility_year`, salary), the modeling pipeline uses **254 features** on the **3,738-row** eligible pool.

## 5. Model Development

The validation strategy is a time-based holdout: train on `eligibility_year < 2000` (**2,439 eligible players, 232 inducted**) and test on `eligibility_year ≥ 2000` (**1,299 eligible players, 49 inducted**). All models are trained inside scikit-learn pipelines with `class_weight = "balanced"` (or `scale_pos_weight` for XGBoost) to handle the imbalanced positive class, with stratified 5-fold cross-validation using `average_precision` (PR AUC) scoring.

Four models are trained in the default run: **Logistic Regression** (regularized baseline, `C` in {0.1, 1}), **Random Forest** (`n_estimators = 100`, `max_depth = None`, `min_samples_leaf` in {1, 4}, `max_features = "sqrt"`), **Gradient Boosting** (`learning_rate = 0.1`, `max_depth = 2`, `n_estimators = 50`), and a **Stacking ensemble** with a Logistic Regression meta-learner over the LR + RF + GB base learners. **XGBoost** is trained additionally when the `xgboost` package is installed in the environment.

## 6. Model Comparison and Selection

Holdout leaderboard (eligibility_year ≥ 2000, 1,299 eligible players, 49 inducted):

| Model | PR AUC | ROC AUC | F1 | Top-25 precision | Brier |
|---|---|---|---|---|---|
| **Gradient Boosting (selected)** | **0.721** | **0.990** | 0.739 | 0.76 | 0.020 |
| Stacking ensemble | 0.706 | 0.992 | 0.488 | 0.76 | 0.061 |
| Random Forest | 0.695 | 0.991 | 0.734 | 0.76 | 0.026 |
| Logistic Regression | 0.576 | 0.985 | 0.464 | 0.64 | 0.069 |

The selected model is **Gradient Boosting**, chosen on the basis of holdout PR AUC (the headline metric for the imbalanced positive class), supported by ROC AUC, calibration on the test cohort, and stable per-cohort behavior. Gradient Boosting is then retrained on the full eligible pool and persisted as `models/final_model_retrained_all_eligible.joblib` for the Shiny app.

Diagnostics on the curated 20-feature career subset confirm the multicollinearity expected from career counting stats: **VIF** is 108.9 for `bat_RBI`, 47.0 for `bat_H`, 41.1 for `pit_IP`, 39.6 for `pit_W`, 37.1 for `bat_HR`, 27.3 for `allstar_years`, 26.2 for `allstar_games`, and 15.5 for `bat_BB` (all flagged "high — regularize"); `n_mlb_seasons` (9.0) and `career_span_years` (7.7) are "moderate". This supports the regularization / tree-based modeling story. **Cook's distance + leverage** on the Logistic Regression baseline flags **349 influential players** above the 4 / n threshold — these are the borderline / controversial HoF candidates and are surfaced in `reports/tables/top_influential_players.csv`.

## 7. Interpretation and Communication

The story is what separates legends from very good players. The selected Gradient Boosting model is interpreted using model-native feature importance, PR / ROC curves, calibration curves, named cluster profiles, and stratified breakdowns by primary role, debut era, and cluster label. The R Shiny **Cooperstown Calculator** is the primary communication artifact: it lets a user pick any eligible player and see the predicted induction probability, the player's archetype, the top model drivers, and a list of comparable historical players. The 142 rows scraped from the Hall of Fame Future Eligibles page are surfaced in the same app so readers can see predictions for currently active candidates the model has not seen in the historical training data.

## 8. Limitations and Future Work

Historical changes in HoF voting behavior (the 1990s expansion of the writers' pool, the steroid-era penalty, contextual factors like postseason narratives) are only partly captured. Salary is missing-by-design before 1985 and is excluded from supervised modeling because it would leak post-career information. Target timing is fragile for very recently eligible players, so the holdout cohort uses `eligibility_year ≥ 2000`. Statcast-era pitch tracking is only sampled from pybaseball; future work could use the full Statcast historical pull (2015 – present) to add command / stuff features for modern pitchers, plus survival analysis for time-to-induction (some players are elected on the first ballot, others wait a decade), and Bayesian hierarchical models to share strength across position-by-era cohorts.

## 9. Reproducibility

`requirements.txt` pins all dependencies. `RANDOM_STATE = 42` is set in `src/config.py`. All preprocessing is fitted inside scikit-learn pipelines on training folds. Raw data downloads are documented in `data/raw/source_manifest.json`. The pipeline runs end-to-end via `python run_all.py` and prints a numbered transcript for each stage.

## 10. Member Contributions

See `docs/member_contributions_template.md`. Replace the placeholder names with the actual team members before submission.
