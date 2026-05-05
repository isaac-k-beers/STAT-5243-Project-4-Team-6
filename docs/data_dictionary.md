# Data Dictionary

The project uses a player-level modeling design. The raw SABR/Lahman tables are first reconciled into `data/raw/analysis_input/`, then aggregated into one row per player.

## Primary processed files

| File | Actual shape | Description |
|---|---:|---|
| `data/processed/player_features_base.csv` | 24,270 rows × 260 columns | Cleaned player-level feature table before clustering. |
| `data/processed/player_features_with_archetypes.csv` | 24,270 rows × 270 columns | Final modeling dataset with cluster labels, PCA coordinates, and cluster-distance features. |
| `data/processed/model_predictions_all_eligible.csv` | 3,738 rows × 21 columns | Final predictions for all players in the eligible modeling pool. |
| `app/model_predictions.csv` | 3,738 rows × 21 columns | Copy of prediction output used by the R Shiny app. |

## Target and eligibility fields

| Field | Meaning |
|---|---|
| `inducted` | Binary target: 1 if the player was inducted into the Hall of Fame, 0 otherwise. |
| `is_eligible` | Eligibility indicator based on career length and retirement timing. |
| `model_eligible_pool` | Modeling-pool indicator; includes eligible non-inducted players and inducted players. |
| `eligibility_year` | Approximate first eligibility year, defined using the five-year retirement rule. |

## Main feature groups

| Group | Examples | Interpretation |
|---|---|---|
| Biographical | `height`, `weight`, `bmi`, `debut_age`, `final_age`, `birthCountry`, `bats`, `throws` | Player background and physical profile. |
| Career batting | `bat_H`, `bat_HR`, `bat_RBI`, `bat_PA`, `bat_OPS`, `bat_mean_OPS_plus_proxy` | Offensive production and era-adjusted batting quality. |
| Career pitching | `pit_W`, `pit_SO`, `pit_SV`, `pit_ERA`, `pit_K9`, `pit_mean_ERA_plus_proxy` | Pitching volume, run prevention, strikeout ability, and era-adjusted pitching quality. |
| Peak performance | 3-year and 5-year peak features | Captures short-run dominance, not only career accumulation. |
| Awards and recognition | `award_total`, `award_all_star_mvp_count`, `awardshare_rows`, `allstar_years` | Historical recognition by writers, leagues, and fans. |
| Postseason | postseason batting and pitching summaries | Playoff performance signals. |
| Position and role | `primary_position`, `primary_role`, position game counts | Separates hitters, pitchers, and two-way players. |
| Archetype features | `cluster_id`, `cluster_label`, `pca1`, `pca2`, `cluster_dist_*` | Unsupervised career-type features created from clustering. |

## Preprocessing matrices

| File | Actual shape | Notes |
|---|---:|---|
| `model_matrix_raw.csv` | 3,738 rows × 248 columns | Encoded raw modeling matrix. |
| `model_matrix_standardized.csv` | 3,738 rows × 248 columns | Numeric variables standardized. |
| `model_matrix_minmax.csv` | 3,738 rows × 248 columns | Numeric variables min-max scaled. |
| `model_matrix_power.csv` | 3,738 rows × 248 columns | Power-transformed numeric variables. |

The `scaled_variants_manifest.csv` reports 246 engineered feature columns because the CSV matrix files also include identifier/target columns.
