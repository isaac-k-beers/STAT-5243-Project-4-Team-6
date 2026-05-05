# Feature Dictionary

The feature table is player-level: one row per `playerID`. The base table contains **260 columns**; after adding archetype and PCA features, the final modeling dataset contains **270 columns**.

## Final dataset shapes

| File | Rows | Columns | Purpose |
|---|---:|---:|---|
| `data/processed/player_features_base.csv` | 24,270 | 260 | Base player-level table before clustering. |
| `data/processed/player_features_with_archetypes.csv` | 24,270 | 270 | Main final modeling dataset. |
| `data/processed/model_predictions_all_eligible.csv` | 3,738 | 21 | Final predictions for eligible players. |

## Feature groups

### Biographical and career timing features

Examples include `debut_age`, `final_age`, `career_span_years`, `height`, `weight`, `bmi`, `birthCountry`, `bats`, and `throws`. These features describe player background, career timing, and physical profile.

### Batting production features

Examples include `bat_PA`, `bat_AB`, `bat_H`, `bat_HR`, `bat_RBI`, `bat_SB`, `bat_BA`, `bat_OBP`, `bat_SLG`, `bat_OPS`, `bat_mean_OPS_plus_proxy`, and `bat_max_OPS_plus_proxy`. These features capture both career volume and rate-based offensive quality.

### Pitching production features

Examples include `pit_IP`, `pit_W`, `pit_L`, `pit_SO`, `pit_SV`, `pit_ERA`, `pit_WHIP`, `pit_K9`, `pit_K_BB`, `pit_mean_ERA_plus_proxy`, and `pit_max_ERA_plus_proxy`. These features capture pitcher workload, effectiveness, and era-adjusted quality.

### Peak-performance features

The pipeline creates peak-window features, including 3-year and 5-year peak summaries. These represent career dominance at a player's best, not only total accumulation.

### Awards and recognition features

Examples include `award_total`, `award_all_star_mvp_count`, `awardshare_rows`, `allstar_games`, and `allstar_years`. These features capture historical recognition and league/media acknowledgment.

### Position and role features

`primary_position` is derived from `Appearances.csv` when available. The script aggregates position game counts and selects the position with the most career appearances. `primary_role` separates hitters, pitchers, and two-way players.

### Unsupervised archetype features

The clustering step adds `cluster_id`, `cluster_label`, `pca1`, `pca2`, and distance-to-centroid features. These features summarize player career archetypes such as power hitters, workhorse starters, and dominant relievers.

## Leakage controls

The following fields are excluded from model predictors because they would leak the target or split logic: `inducted`, `induction_year`, Hall voting fields, `is_eligible`, `model_eligible_pool`, `eligibility_year`, and direct identifiers such as `playerID` and `player_name`.

## Most important feature groups in the final model

The selected Gradient Boosting model's top features include awards, All-Star appearances, peak/career production, pitcher wins, batting average, runs, OPS, career length, and era-adjusted pitching quality. This suggests that Hall of Fame induction is associated with both statistical production and historical recognition.
