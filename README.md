# MLB Hall of Fame Prediction + Player Archetype Clustering

**STAT 5243 Project 4 (Final) — Team 6.** End-to-end machine-learning pipeline that predicts whether an MLB player will be inducted into the Baseball Hall of Fame, and discovers latent player archetypes that underlie that decision.

---

## Predictive question

> Among MLB players who satisfy BBWAA-style Hall of Fame eligibility (played ≥ 10 MLB seasons and retired ≥ 5 calendar years), can we predict whether a player will be inducted into the Baseball Hall of Fame using career performance, awards, biographical attributes, era-adjusted statistics, and unsupervised player archetypes?

**Target variable.** `inducted` — binary classification (`1 = elected`, `0 = eligible but not elected`).

**Actual rerun headline result.** The final rerun selected **gradient_boosting** by holdout PR AUC. On the time-based holdout cohort of **1,299 eligible players** with **49 inducted positives**, the selected model achieved **PR AUC 0.721**, **ROC AUC 0.990**, **top-25 precision 0.76**, and **top-50 precision 0.74**.

---

## Data sources

Four sources are combined to satisfy the Advanced data-collection rubric:

| Source | Coverage | Role |
|---|---|---|
| **SABR / Lahman 1871–2025** (full 27-table CSV release) | 1871–2025; Hall of Fame table through 2026 | Primary structured source. `People.csv` is mapped to `Master.csv`; 18 reconciled analysis tables are created. |
| **Kaggle Baseball Databank** | 1871–2015 | Audit / backup copy of the older schema. |
| **pybaseball** | player ID lookup + 2015 Statcast sample | Programmatic acquisition example for modern baseball data and pitch-level Statcast data. |
| **BeautifulSoup web scrape** | live Hall of Fame Future Eligibles page | Semi-structured web-scraping evidence and potential future-candidate source. The current Shiny app uses precomputed model predictions, not the scraped page directly. |

---

## How to run

```bash
pip install -r requirements.txt

python scripts/00_verify_data.py
python scripts/01a_data_acquisition_part1.py
python scripts/01_build_player_features.py
python scripts/02_eda_and_archetypes.py
python scripts/03_train_models.py
python scripts/04_make_report_assets.py
```

For the R Shiny app:

```r
setwd("app")
shiny::runApp(".")
```

---

## Pipeline parts

| Part | Files | Purpose |
|---|---|---|
| Part 0 | `README.md`, `requirements.txt`, `docs/` | Project documentation and reproducibility setup |
| Part 1 | `scripts/00_verify_data.py`, `scripts/01a_data_acquisition_part1.py`, `scripts/01_build_player_features.py`, `src/data_loader.py`, `src/features.py`, `notebooks/01_data_acquisition.ipynb` | Multi-source acquisition, relational joins, target construction, player-level feature table |
| Part 2 | `scripts/02_eda_and_archetypes.py`, `src/eda.py`, `src/clustering.py`, `src/diagnostics.py` | Summary statistics, distribution and correlation analysis, hypothesis tests, VIF diagnostic, anomaly/influence diagnostics, K-means / hierarchical clustering, PCA |
| Part 3 | `src/features.py`, `src/preprocessing.py` | Feature engineering, IQR outlier flagging, Yeo-Johnson power transforms, concentration/diversity metrics, interaction terms, scaled model-matrix variants |
| Part 4 | `scripts/03_train_models.py`, `src/models.py` | Logistic Regression, Random Forest, Gradient Boosting, and Stacking ensemble in the current run; `src/models.py` also contains conditional XGBoost support if available in the local environment |
| Part 5 | `src/evaluate.py`, `reports/tables/`, `reports/figures/` | Multi-metric evaluation, calibration, stratified breakdowns, influence diagnostics, feature importance |
| Part 6 | `app/app.R` | R Shiny **Cooperstown Calculator** — interactive web app for Hall-of-Fame predictions |
| Part 7 | `reports/final_report_draft.md`, `presentation/slides_outline.md` | Written report and oral presentation deliverables |

---

## Leakage controls

- `HallOfFame.csv` is used only to build the `inducted` target. Hall voting fields (`votes`, `ballots`, `needed`, vote percentage proxies) are excluded from the supervised feature matrix.
- `eligibility_year` (`final_year + 5`) is used for the time-based train/test split, **not** as a modeling feature.
- Salary columns are excluded from supervised modeling because salary is historically incomplete and can leak era/post-career information.
- All preprocessing is fitted inside scikit-learn pipelines on training folds only.

---

## Main outputs from the actual rerun

| File | Actual shape / role | Used by |
|---|---|---|
| `data/processed/player_features_base.csv` | 24,270 rows × 260 columns | EDA, summary statistics, feature audit |
| `data/processed/player_features_with_archetypes.csv` | 24,270 rows × 270 columns | **main final modeling dataset** |
| `data/processed/model_matrix_*.csv` | 3,738 eligible players × 248 columns | preprocessing variants |
| `data/processed/model_predictions_all_eligible.csv` | 3,738 rows × 21 columns | report, ranked candidates, Shiny app |
| `models/final_model_retrained_all_eligible.joblib` | selected model refit on all eligible players | inference/app |
| `app/model_predictions.csv` | copy of prediction table | live Shiny demo |
| `reports/tables/model_comparison_holdout.csv` | actual model leaderboard | model selection memo |

---

## Data source summary numbers

| Quantity | Value |
|---|---:|
| Total players in master table | 24,270 |
| Year coverage | 1871–2021 |
| Eligible modeling pool | 3,738 |
| Inducted positives in pool | 281 |
| Induction rate | 7.5% |
| Pitchers in pool | 1,227 |
| Hitters in pool | 2,309 |
| Two-way players in pool | 202 |
| Time-based holdout cohort | 1,299 players (49 inducted) |

## Actual model leaderboard

| model               |   pr_auc |   roc_auc |   precision |   recall |       f1 |   balanced_accuracy |   brier_score |   top_25_precision |   top_50_precision |
|:--------------------|---------:|----------:|------------:|---------:|---------:|--------------------:|--------------:|-------------------:|-------------------:|
| gradient_boosting   | 0.72117  |  0.989584 |    0.628571 | 0.897959 | 0.739496 |            0.93858  |     0.0196428 |               0.76 |               0.74 |
| stacking            | 0.70571  |  0.991771 |    0.322368 | 1        | 0.487562 |            0.9588   |     0.0611595 |               0.76 |               0.78 |
| random_forest       | 0.69497  |  0.991265 |    0.594937 | 0.959184 | 0.734375 |            0.966792 |     0.0261383 |               0.76 |               0.76 |
| logistic_regression | 0.576372 |  0.985061 |    0.302469 | 1        | 0.464455 |            0.9548   |     0.0688102 |               0.64 |               0.62 |

The selected final model is **gradient_boosting** because it has the highest holdout PR AUC in the actual rerun.

---

## Team and contributions

| Member | Role |
|---|---|
| Person A | Data Engineer & Pipeline Lead — Part 1 |
| Person B | EDA & Unsupervised Learning Lead — Part 2 |
| Person C | Feature Engineering & Modeling Lead — Parts 3, 4 |
| Person D | Evaluation, Communication & Bonus App Lead — Parts 5, 6, 7 |

Replace `Person A–D` with actual names before final submission.
