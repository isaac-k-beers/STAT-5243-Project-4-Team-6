# MLB Hall of Fame Prediction + Player Archetype Clustering

## Project overview

This project asks whether an eligible Major League Baseball player will be inducted into the Baseball Hall of Fame. The target variable is `inducted`, where `1` means the player was inducted and `0` means the player was eligible but not inducted.

The project combines two ideas into one coherent story:

1. **Prediction:** estimate a player's Hall of Fame induction probability from career statistics, awards, era-adjusted metrics, and biographical information.
2. **Interpretation:** cluster players into career archetypes so the model is not only predictive, but also easier to explain.

The central story is: **what separates Cooperstown legends from very good players?**

## Actual final rerun result

The final reproducible rerun selected **`gradient_boosting`** by holdout PR AUC. The time-based holdout set contained **1,299 eligible players** with **49 inducted positives**. The selected model achieved:

| Metric | Actual value |
|---|---:|
| Holdout PR AUC | 0.727281 |
| Holdout ROC AUC | 0.989845 |
| Precision at 0.5 | 0.696429 |
| Recall at 0.5 | 0.795918 |
| F1 at 0.5 | 0.742857 |
| Balanced accuracy | 0.891159 |
| Brier score | 0.016512 |
| Top-25 precision | 0.84 |
| Top-50 precision | 0.72 |

## Data sources

| Source | Coverage / output | Role in project |
|---|---|---|
| **SABR / Lahman 1871-2025** | 27 CSV files; Hall of Fame table through 2026 | Primary structured baseball database. `People.csv` is mapped to `Master.csv`. |
| **Kaggle Baseball Databank** | Older 1871-2015 mirror | Backup and audit copy. |
| **pybaseball** | 4 player-ID lookup rows and 314 Statcast sample rows | Programmatic external-data acquisition example. |
| **BeautifulSoup scrape** | 142 text rows from the Hall of Fame Future Eligibles page | Semi-structured web-scraping evidence and future-candidate extension source. |

The actual source manifest is saved in `reports/tables/part1_source_manifest.csv`.

## How to run

Create or activate the environment:

```bash
conda activate project4
pip install -r requirements.txt
```

Run step by step:

```bash
python scripts/00_verify_data.py
python scripts/01a_data_acquisition_part1.py
python scripts/01_build_player_features.py
python scripts/02_eda_and_archetypes.py
python scripts/03_train_models.py
python scripts/04_make_report_assets.py
python scripts/05_update_actual_outputs_docs.py
```

Or run the full pipeline:

```bash
python run_all.py
```

## Main generated datasets

| File | Actual shape | Purpose |
|---|---:|---|
| `data/processed/player_features_base.csv` | 24,270 rows × 260 columns | Cleaned player-level career table before clustering. |
| `data/processed/player_features_with_archetypes.csv` | 24,270 rows × 270 columns | Final modeling dataset with archetype/PCA features. |
| `data/processed/model_matrix_raw.csv` | 3,738 rows × 248 columns | Raw encoded modeling matrix. |
| `data/processed/model_matrix_standardized.csv` | 3,738 rows × 248 columns | Standardized model matrix. |
| `data/processed/model_matrix_minmax.csv` | 3,738 rows × 248 columns | Min-max scaled model matrix. |
| `data/processed/model_matrix_power.csv` | 3,738 rows × 248 columns | Power-transformed model matrix. |
| `data/processed/model_predictions_all_eligible.csv` | 3,738 rows × 21 columns | Final predictions for the eligible player pool. |
| `app/model_predictions.csv` | 3,738 rows × 21 columns | Prediction file used by the Shiny app. |

## Pipeline structure

| Part | Script | Main output |
|---|---|---|
| Data acquisition | `scripts/01a_data_acquisition_part1.py` | `data/raw/analysis_input/`, source manifest, external-source status |
| Feature table construction | `scripts/01_build_player_features.py` | `player_features_base.csv` |
| EDA and archetypes | `scripts/02_eda_and_archetypes.py` | summary tables, figures, clusters, PCA features |
| Model development | `scripts/03_train_models.py` | trained models and holdout predictions |
| Report assets | `scripts/04_make_report_assets.py` | `reports/final_report_draft.md` |
| Output sync | `scripts/05_update_actual_outputs_docs.py` | `docs/actual_outputs_summary.md`, `docs/recheck_status.md` |

## Actual model comparison

| model               |   pr_auc |   roc_auc |   precision |   recall |       f1 |   balanced_accuracy |   brier_score |   top_25_precision |   top_50_precision |
|:--------------------|---------:|----------:|------------:|---------:|---------:|--------------------:|--------------:|-------------------:|-------------------:|
| gradient_boosting   | 0.727281 |  0.989845 |    0.696429 | 0.795918 | 0.742857 |            0.891159 |     0.0165117 |               0.84 |               0.72 |
| stacking            | 0.72118  |  0.992098 |    0.295181 | 1        | 0.455814 |            0.9532   |     0.0665042 |               0.76 |               0.82 |
| random_forest       | 0.705554 |  0.991478 |    0.619718 | 0.897959 | 0.733333 |            0.93818  |     0.0252146 |               0.72 |               0.8  |
| logistic_regression | 0.571217 |  0.982465 |    0.231132 | 1        | 0.375479 |            0.9348   |     0.0998615 |               0.68 |               0.6  |

The final selected model is **`gradient_boosting`**, because it achieved the highest holdout PR AUC in the actual rerun.

## R Shiny app

After model training, run:

```bash
R -e 'shiny::runApp("app")'
```

The app reads `app/model_predictions.csv` and displays player-level Hall of Fame probabilities.

## Final report note

Use `docs/actual_outputs_summary.md` as the source of truth for all final numbers. If older expected values appear anywhere, replace them with the actual values recorded there.

## Team and contributions

| Member | Role |
|---|---|
| Isaac Beers | Data acquisition, source verification, and data documentation |
| Yolanda He | Exploratory data analysis, visualizations, and unsupervised learning |
| Qiuting He | Feature engineering, feature dictionary, and feature-set design |
| Rui Lin | Data preprocessing, leakage-safe pipeline, and modeling support |
