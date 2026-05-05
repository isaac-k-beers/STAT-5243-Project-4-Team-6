# Recheck Status

Rechecked against the actual output files in the final local rerun.

## Commands represented by the current outputs

```bash
python scripts/00_verify_data.py
python scripts/01a_data_acquisition_part1.py
python scripts/01_build_player_features.py
python scripts/02_eda_and_archetypes.py
python scripts/03_train_models.py
python scripts/04_make_report_assets.py
python scripts/05_update_actual_outputs_docs.py
```

## Data and feature outputs

| Output | Actual result |
|---|---:|
| Base player feature table | 24,270 rows × 260 columns |
| Final archetype modeling table | 24,270 rows × 270 columns |
| Eligible prediction table | 3,738 rows × 21 columns |
| Model matrix files | 3,738 rows × 248 columns each |
| Eligible modeling pool | 3,738 players |
| Inducted positives | 281 players |

## Source coverage

| table                  | status   | source_used   |   rows |   min_year |   max_year |
|:-----------------------|:---------|:--------------|-------:|-----------:|-----------:|
| Master.csv             | ready    | SABR/Lahman   |  24270 |        nan |        nan |
| Batting.csv            | ready    | SABR/Lahman   | 128598 |       1871 |       2025 |
| Pitching.csv           | ready    | SABR/Lahman   |  57630 |       1871 |       2025 |
| Fielding.csv           | ready    | SABR/Lahman   | 174332 |       1871 |       2025 |
| HallOfFame.csv         | ready    | SABR/Lahman   |   6426 |       1936 |       2026 |
| AllstarFull.csv        | ready    | SABR/Lahman   |   6425 |       1933 |       2025 |
| AwardsPlayers.csv      | ready    | SABR/Lahman   |  12667 |       1877 |       2025 |
| AwardsSharePlayers.csv | ready    | SABR/Lahman   |   7613 |       1911 |       2025 |
| Salaries.csv           | ready    | SABR/Lahman   |  26428 |       1985 |       2016 |
| Teams.csv              | ready    | SABR/Lahman   |   3614 |       1871 |       2025 |
| BattingPost.csv        | ready    | SABR/Lahman   |  18687 |       1884 |       2025 |
| PitchingPost.csv       | ready    | SABR/Lahman   |   7474 |       1884 |       2025 |
| FieldingOF.csv         | ready    | SABR/Lahman   |  12028 |       1871 |       1955 |
| SeriesPost.csv         | ready    | SABR/Lahman   |    440 |       1884 |       2025 |
| Appearances.csv        | ready    | SABR/Lahman   | 128512 |       1871 |       2025 |
| FieldingPost.csv       | ready    | SABR/Lahman   |  17934 |       1903 |       2025 |
| TeamsFranchises.csv    | ready    | SABR/Lahman   |    203 |        nan |        nan |
| TeamsHalf.csv          | ready    | SABR/Lahman   |    142 |       1892 |       1981 |

## External-source status

| timestamp_utc             | source                            | success   | message                                                                                      |
|:--------------------------|:----------------------------------|:----------|:---------------------------------------------------------------------------------------------|
| 2026-05-05T22:39:55+00:00 | SABR/Lahman                       | True      | SABR/Lahman CSVs already present (27 files).                                                 |
| 2026-05-05T22:39:55+00:00 | BeautifulSoup Hall of Fame scrape | True      | BeautifulSoup scrape saved 142 text rows.                                                    |
| 2026-05-05T22:39:59+00:00 | pybaseball                        | True      | Saved pybaseball ID lookup examples (4 rows). | Saved pybaseball Statcast sample (314 rows). |

## Model comparison

| model               |   pr_auc |   roc_auc |   precision |   recall |       f1 |   balanced_accuracy |   brier_score |   top_25_precision |   top_50_precision |
|:--------------------|---------:|----------:|------------:|---------:|---------:|--------------------:|--------------:|-------------------:|-------------------:|
| gradient_boosting   | 0.727281 |  0.989845 |    0.696429 | 0.795918 | 0.742857 |            0.891159 |     0.0165117 |               0.84 |               0.72 |
| stacking            | 0.72118  |  0.992098 |    0.295181 | 1        | 0.455814 |            0.9532   |     0.0665042 |               0.76 |               0.82 |
| random_forest       | 0.705554 |  0.991478 |    0.619718 | 0.897959 | 0.733333 |            0.93818  |     0.0252146 |               0.72 |               0.8  |
| logistic_regression | 0.571217 |  0.982465 |    0.231132 | 1        | 0.375479 |            0.9348   |     0.0998615 |               0.68 |               0.6  |

## Selected model

The selected model is **`gradient_boosting`**.

| Metric | Actual value |
|---|---:|
| Holdout PR AUC | 0.727281 |
| Holdout ROC AUC | 0.989845 |
| Train rows | 2,439 |
| Test rows | 1,299 |
| Train positives | 232 |
| Test positives | 49 |
| Features used | 244 |

## Recheck conclusion

The markdown files should refer to the actual current outputs above. The current final run uses full SABR/Lahman as the primary data source, includes pybaseball and BeautifulSoup acquisition evidence, creates EDA and clustering outputs, trains four reported supervised models, and selects Gradient Boosting by holdout PR AUC.

## Latest Actual Output Summary


Generated at: 2026-05-05 19:15:40

This file is generated from the current local output files after rerunning the full pipeline. Use these numbers in the final report and presentation.

## 1. Processed data shapes

| File | Actual shape | Purpose |
|---|---:|---|
| `data/processed/player_features_base.csv` | 24,270 rows × 260 columns | Cleaned player-level feature table before clustering |
| `data/processed/player_features_with_archetypes.csv` | 24,270 rows × 270 columns | Final modeling dataset with archetype features |
| `data/processed/model_predictions_all_eligible.csv` | 3,738 rows × 21 columns | Final predictions for eligible players |
| `app/model_predictions.csv` | 3,738 rows × 21 columns | Prediction file used by the Shiny app |

## 2. Data source manifest

| table                  | status   | source_used   |   rows |   min_year |   max_year |
|:-----------------------|:---------|:--------------|-------:|-----------:|-----------:|
| Master.csv             | ready    | SABR/Lahman   |  24270 |        nan |        nan |
| Batting.csv            | ready    | SABR/Lahman   | 128598 |       1871 |       2025 |
| Pitching.csv           | ready    | SABR/Lahman   |  57630 |       1871 |       2025 |
| Fielding.csv           | ready    | SABR/Lahman   | 174332 |       1871 |       2025 |
| HallOfFame.csv         | ready    | SABR/Lahman   |   6426 |       1936 |       2026 |
| AllstarFull.csv        | ready    | SABR/Lahman   |   6425 |       1933 |       2025 |
| AwardsPlayers.csv      | ready    | SABR/Lahman   |  12667 |       1877 |       2025 |
| AwardsSharePlayers.csv | ready    | SABR/Lahman   |   7613 |       1911 |       2025 |
| Salaries.csv           | ready    | SABR/Lahman   |  26428 |       1985 |       2016 |
| Teams.csv              | ready    | SABR/Lahman   |   3614 |       1871 |       2025 |
| BattingPost.csv        | ready    | SABR/Lahman   |  18687 |       1884 |       2025 |
| PitchingPost.csv       | ready    | SABR/Lahman   |   7474 |       1884 |       2025 |
| FieldingOF.csv         | ready    | SABR/Lahman   |  12028 |       1871 |       1955 |
| SeriesPost.csv         | ready    | SABR/Lahman   |    440 |       1884 |       2025 |
| Appearances.csv        | ready    | SABR/Lahman   | 128512 |       1871 |       2025 |
| FieldingPost.csv       | ready    | SABR/Lahman   |  17934 |       1903 |       2025 |
| TeamsFranchises.csv    | ready    | SABR/Lahman   |    203 |        nan |        nan |
| TeamsHalf.csv          | ready    | SABR/Lahman   |    142 |       1892 |       1981 |

## 3. External source status

| timestamp_utc             | source                            | success   | message                                                                                      |
|:--------------------------|:----------------------------------|:----------|:---------------------------------------------------------------------------------------------|
| 2026-05-05T23:11:13+00:00 | SABR/Lahman                       | True      | SABR/Lahman CSVs already present (27 files).                                                 |
| 2026-05-05T23:11:14+00:00 | BeautifulSoup Hall of Fame scrape | True      | BeautifulSoup scrape saved 142 text rows.                                                    |
| 2026-05-05T23:11:17+00:00 | pybaseball                        | True      | Saved pybaseball ID lookup examples (4 rows). | Saved pybaseball Statcast sample (314 rows). |

## 4. Headline EDA KPIs

|   n_players_total |   n_eligible_players |   n_inducted |   induction_rate |   year_range_min |   year_range_max |   median_career_seasons |   median_career_span_years |   n_pitchers |   n_hitters |   n_two_way |
|------------------:|---------------------:|-------------:|-----------------:|-----------------:|-----------------:|------------------------:|---------------------------:|-------------:|------------:|------------:|
|             24270 |                 3738 |          281 |           0.0752 |             1871 |             2021 |                      13 |                         13 |         1227 |        2309 |         202 |

## 5. Cluster diagnostics

|   k |   silhouette |   davies_bouldin |   inertia |
|----:|-------------:|-----------------:|----------:|
|   3 |     0.224728 |          1.62095 |   69955.4 |
|   4 |     0.228528 |          1.60396 |   63640.9 |
|   5 |     0.23264  |          1.46294 |   59234.2 |
|   6 |     0.1501   |          1.63494 |   55749.1 |
|   7 |     0.171865 |          1.55912 |   52413.4 |

## 6. Preprocessing variants

| variant      | path                                         |   n_rows |   n_continuous_features |   n_binary_features |   n_total_features |
|:-------------|:---------------------------------------------|---------:|------------------------:|--------------------:|-------------------:|
| raw          | data/processed/model_matrix_raw.csv          |     3738 |                     216 |                  30 |                246 |
| standardized | data/processed/model_matrix_standardized.csv |     3738 |                     216 |                  30 |                246 |
| minmax       | data/processed/model_matrix_minmax.csv       |     3738 |                     216 |                  30 |                246 |
| power        | data/processed/model_matrix_power.csv        |     3738 |                     216 |                  30 |                246 |

## 7. Model comparison

| model               |   pr_auc |   roc_auc |   precision |   recall |       f1 |   balanced_accuracy |   brier_score |   top_25_precision |   top_50_precision |
|:--------------------|---------:|----------:|------------:|---------:|---------:|--------------------:|--------------:|-------------------:|-------------------:|
| gradient_boosting   | 0.727281 |  0.989845 |    0.696429 | 0.795918 | 0.742857 |            0.891159 |     0.0165117 |               0.84 |               0.72 |
| stacking            | 0.72118  |  0.992098 |    0.295181 | 1        | 0.455814 |            0.9532   |     0.0665042 |               0.76 |               0.82 |
| random_forest       | 0.705554 |  0.991478 |    0.619718 | 0.897959 | 0.733333 |            0.93818  |     0.0252146 |               0.72 |               0.8  |
| logistic_regression | 0.571217 |  0.982465 |    0.231132 | 1        | 0.375479 |            0.9348   |     0.0998615 |               0.68 |               0.6  |

## 8. Training summary

| Quantity | Actual value |
|---|---:|
| Best model | `gradient_boosting` |
| Best holdout PR AUC | 0.7272813794700826 |
| Best holdout ROC AUC | 0.9898448979591837 |
| Train rows | 2439 |
| Test rows | 1299 |
| Train positives | 232 |
| Test positives | 49 |
| Features used | 244 |

## 9. Report-writing note

Use the numbers in this file as the source of truth. If older markdown files mention different values, update them to match this file.
