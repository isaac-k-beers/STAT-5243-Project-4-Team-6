# Feature Dictionary

The feature table is player-level: one row per `playerID`. The base feature table holds 260 columns; once the cluster features from the unsupervised step are merged the table holds approximately 268 columns.

## Target and eligibility

- `inducted`: 1 if the player appears as inducted in `HallOfFame.csv` with category `Player`; otherwise 0.
- `n_mlb_seasons`: number of unique years across `Batting.csv`, `Pitching.csv`, and `Fielding.csv`.
- `eligibility_year`: `final_year + 5`, an implementation of the BBWAA rule that a player must be retired for at least 5 calendar years before their first ballot. Used for the time-based train / test split, never as a model feature.
- `is_eligible`: 1 if `n_mlb_seasons >= 10` AND `eligibility_year <= max(HallOfFame.yearid)`; otherwise 0.
- `model_eligible_pool`: 1 for eligible players or already-inducted players (older Negro-Leagues / pre-modern inductees who do not satisfy the 10-season modern threshold are still kept as positives).
- `induction_year`, `hof_first_ballot_year`, `hof_last_ballot_year`, `hof_ballot_rows`: target-construction audit fields, excluded from the modeling feature matrix.

## Career batting features

`bat_G`, `bat_PA`, `bat_AB`, `bat_R`, `bat_H`, `bat_2B`, `bat_3B`, `bat_HR`, `bat_RBI`, `bat_SB`, `bat_CS`, `bat_BB`, `bat_SO`, `bat_HBP`, `bat_SF`, `bat_GIDP`, `bat_TB`, `bat_BA`, `bat_OBP`, `bat_SLG`, `bat_OPS`, `bat_ISO`, `bat_HR_rate`, `bat_BB_rate`, `bat_SO_rate`, `bat_SB_success`.

## Career pitching features

`pit_G`, `pit_GS`, `pit_W`, `pit_L`, `pit_CG`, `pit_SHO`, `pit_SV`, `pit_IPouts`, `pit_IP`, `pit_H`, `pit_ER`, `pit_HR`, `pit_BB`, `pit_SO`, `pit_ERA`, `pit_WHIP`, `pit_K_BB`, `pit_K9`, `pit_BB9`, `pit_HR9`, `pit_win_pct`, `pit_start_share`, `pit_complete_game_share`, `pit_save_share`.

## Peak-window features

3-year and 5-year rolling-window peaks for the strongest counting and rate stats: `bat_peak3_HR`, `bat_peak3_H`, `bat_peak3_RBI`, `bat_peak3_SB`, `bat_peak3_OPS`, `bat_peak3_OPS_z`, `bat_peak3_OPS_plus_proxy`, and 5-year analogues; same for pitching with `pit_peak3_W`, `pit_peak3_SO`, `pit_peak3_SV`, `pit_peak3_IP`, `pit_peak3_ERA_plus_proxy`, `pit_peak3_ERA_z_good`, and 5-year versions.

## Era-adjusted features

`bat_mean_OPS_z`, `bat_max_OPS_z`, `bat_mean_OPS_plus_proxy`, `bat_max_OPS_plus_proxy` for batters; `pit_mean_ERA_z_good`, `pit_max_ERA_z_good`, `pit_mean_ERA_plus_proxy`, `pit_max_ERA_plus_proxy` for pitchers. League benchmarks are computed per-year so a 1968 hitter and a 1999 hitter are directly comparable on the same scale.

## Awards and recognition

`award_total`, `award_unique`, `award_years`, `award_mvp_count`, `award_cyyoung_count`, `award_gold_glove_count`, `award_silver_slugger_count`, `award_rookie_count`, `award_triple_crown_count`, `award_world_series_mvp_count`, `award_all_star_mvp_count`. From `AwardsSharePlayers.csv`: `awardshare_rows`, `awardshare_pointsWon_sum`, `awardshare_pointsMax_sum`, `awardshare_votesFirst_sum`, `awardshare_max_share`, `awardshare_mean_share`. From `AllstarFull.csv`: `allstar_games`, `allstar_years`, `allstar_appearances`, `allstar_starts`.

## Postseason features

`batpost_*` aggregates from `BattingPost.csv` and `pitpost_*` aggregates from `PitchingPost.csv`, including derived `batpost_OPS`, `pitpost_ERA`, `pitpost_WHIP`.

## Fielding and role

`primary_position` is derived from `Appearances.csv` when available — the script aggregates per-position game counts (`G_p`, `G_c`, `G_1b`, …, `G_dh`) and selects the position with the most games for each player. Falls back to `Fielding.csv` only if `Appearances.csv` is unavailable. `primary_position_games` is the career sum of games at the primary position. `primary_role` (Hitter / Pitcher / Two-Way) is decided by PA and IP thresholds. `fld_fielding_pct`, `pos_games_*` (one column per fielding position observed in the data), `of_Glf`, `of_Gcf`, `of_Grf` from `FieldingOF.csv`. `primary_pos_scarcity` is an ordinal scarcity weight: catcher and shortstop receive higher weights, since they are scarcer in HoF inductees relative to position players at first base or DH.

## Biographical features

`birthYear`, `birthMonth`, `birthCountry`, `weight`, `height`, `bmi`, `debut_age`, `final_age`, `career_span_years`, `bats`, `throws`, `bats_throws`, `international_flag`, `birth_month_sin`, `birth_month_cos`, `debut_era`, `debut_decade`. `*_missing` flag columns mark fields with structural missingness.

## Salary

`salary_total`, `salary_mean`, `salary_median`, `salary_max`, `salary_years`, `salary_first_year`, `salary_last_year`. Salary is missing-by-design before 1985 and is excluded from supervised modeling because it leaks post-career information.

## IQR outlier flags

For each heavy-tailed counting stat, the table holds `*_outlier_high` and `*_outlier_low` flags computed from the `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]` rule. The flags are documentation-only — outlier values are NOT capped, because the players in the high tail are precisely the legends the HoF model must identify. Example columns: `bat_HR_outlier_high`, `bat_H_outlier_high`, `pit_W_outlier_high`, `pit_SO_outlier_high`. The aggregate `n_iqr_outlier_high` counts how many career-stat dimensions place a player in the legendary tail.

## Yeo-Johnson transformed features

For heavy-tailed counting stats, a parallel `*_yj` column holds the Yeo-Johnson power-transformed value, providing approximately normal distributions for any linear or distance-based downstream model. Example: `bat_HR_yj`, `pit_SO_yj`, `bat_H_yj`, `bat_RBI_yj`, `pit_W_yj`, `pit_SV_yj`.

## Concentration and diversity metrics

- `award_entropy` and `award_hhi` quantify whether a player's career awards are diverse (5-tool legend) or concentrated in one award type (specialist). Computed across the `award_*_count` columns.
- `position_versatility_entropy` and `position_concentration_hhi` quantify whether the player played one position (specialist) or many positions (utility / two-way). Computed across the `pos_games_*` columns.
- `career_award_total_norm` is the row-sum of award counts used as the normalization base.

## Interaction features

- `peak_x_longevity`: `bat_max_OPS_plus_proxy × n_mlb_seasons`. Captures sustained excellence.
- `allstar_x_pos_scarcity`: `allstar_games × primary_pos_scarcity`. Captures rare-position superstars.
- `mvp_x_postseason_HR`: `award_mvp_count × batpost_HR` (or `mvp_x_career_HR` as a fallback when postseason batting is unavailable).
- `is_pitcher_x_K`: pitcher-specific strikeout signal.
- `steroid_era_x_HR`: HRs accumulated during the steroid era (1994 – 2005), used by the model to learn the historical voter penalty.

## Unsupervised features

After K-means clustering with k chosen by silhouette score (k = 5 in the run), the modeling table receives 8 additional columns: `cluster_id`, `cluster_label`, `hier_cluster_id`, `pca1`, `pca2`, `cluster_distance_0`, `cluster_distance_1`, `cluster_distance_2` (distances to the first three centroids; the full distance matrix per row is computed but only the first three are exposed by the merge). These convert the unsupervised structure into supervised inputs.

## Output files

| File | Rows | Cols | Notes |
|---|---|---|---|
| `data/processed/player_features_base.csv` | 24,270 | 260 | one row per player; primary EDA target |
| `data/processed/player_features_with_archetypes.csv` | 24,270 | ~268 | base + cluster features; primary modeling input |
| `data/processed/model_predictions_all_eligible.csv` | 3,738 | 21 | predictions for the eligible pool; consumed by the Shiny app |
