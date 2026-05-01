# Part 1 — Data Acquisition & Preparation

This project uses a four-source acquisition strategy to satisfy the Advanced data-collection tier on the Project 4 rubric.

## Sources

1. **SABR / Lahman 2025** — primary structured source. The full 27-CSV release lives in `data/raw/sabr_lahman/`; the 18 tables actually consumed by the modeling pipeline are reconciled into `data/raw/analysis_input/`. `People.csv` is renamed to `Master.csv` for project-name compatibility. `Appearances.csv` is part of this release and is used as the canonical source for primary-position construction.
2. **Kaggle Baseball Databank** — the original 14-CSV Kaggle release is preserved in `data/raw/kaggle/` as an audit / backup copy. The pipeline only reads from this folder when SABR / Lahman files are not available.
3. **pybaseball** — programmatic API to Baseball-Reference / FanGraphs / Baseball Savant. The acquisition script produces a `playerid_lookup` example (4 rows) and a Statcast pitch-level sample (314 rows for one 2015 game day) in `data/raw/pybaseball/`.
4. **BeautifulSoup scrape** — focused scrape of `https://baseballhall.org/hall-of-fame/future-eligibles` producing 142 rows in `data/raw/scraped/`. These rows are surfaced by the R Shiny app as the live candidate list.

## Tables consumed by the pipeline

The 18 tables loaded by `src/features.py` and joined on `playerID` and `(playerID, yearID)`, all sourced from SABR / Lahman 2025:

- `Master.csv` (← SABR `People.csv`)
- `Batting.csv`, `Pitching.csv`, `Fielding.csv`, `FieldingOF.csv`, `FieldingPost.csv`
- `BattingPost.csv`, `PitchingPost.csv`, `SeriesPost.csv`
- `HallOfFame.csv` (target construction, 1936 – 2026)
- `AwardsPlayers.csv`, `AwardsSharePlayers.csv`, `AllstarFull.csv`
- `Salaries.csv` (descriptive only; excluded from modeling features)
- `Teams.csv`, `TeamsFranchises.csv`, `TeamsHalf.csv`
- `Appearances.csv` (canonical source for primary position)

`reports/tables/part1_source_manifest.csv` records exactly which file came from which source for every reconciled table.


The raw input is not a single simple table. It is 18 relational tables spanning 1871 – 2025, with multi-stint player-season-team rows, structural missingness in biographical and salary fields, era differences, and a non-trivial target-construction story (BBWAA eligibility of ≥ 10 MLB seasons plus retirement ≥ 5 calendar years before the latest ballot). Adding the Kaggle audit copy, pybaseball Statcast sample, and the BeautifulSoup scrape of Future Eligibles brings together structured bulk data and live web data, exactly the multi-source pattern the rubric calls for.

## Primary-position construction

`src/features.py` uses `Appearances.csv` as the canonical source for primary position. The script aggregates per-position game counts (`G_p`, `G_c`, `G_1b`, …, `G_dh`) and selects the position with the most games for each player. It falls back to `Fielding.csv` only if `Appearances.csv` is unavailable.

## Target and eligibility

- `inducted = 1` if the player appears as inducted in `HallOfFame.csv` with category `Player`; else `0`.
- `n_mlb_seasons` is the count of unique years across `Batting.csv`, `Pitching.csv`, and `Fielding.csv`.
- `eligibility_year = final_year + 5` implements the BBWAA rule that a player must be retired for at least 5 calendar years before their first ballot. Used for the time-based train / test split, never as a model feature.
- `is_eligible = 1` if `n_mlb_seasons ≥ 10` AND `eligibility_year ≤ max(HallOfFame.yearid)`; else `0`.
- `model_eligible_pool = 1` for eligible players or already-inducted players (older Negro-Leagues / pre-modern inductees who do not satisfy the 10-season modern threshold are still kept as positives).

## How to run Part 1

From the project root:

```bash
pip install -r requirements.txt
python scripts/00_verify_data.py
python scripts/01a_data_acquisition_part1.py
python scripts/01_build_player_features.py
```

## Key outputs

```text
data/raw/analysis_input/                         # the 18 reconciled tables consumed by the pipeline
reports/tables/part1_source_manifest.csv         # table-by-table source audit
reports/tables/part1_external_source_status.csv  # external-source success / failure log
data/raw/source_manifest.json                    # machine-readable source log
data/processed/player_features_base.csv          # 24,270 players × 260 columns
```

## Leakage controls

`HallOfFame.csv` is used only to build the `inducted` target. Hall voting fields (`votes`, `ballots`, `needed`, vote percentages) are explicitly excluded from the feature matrix. `eligibility_year` is used for the time-based train / test split and is excluded from the model feature matrix. Salary columns are excluded from supervised modeling because salary is missing-by-design before 1985 and leaks post-career information. All preprocessing (imputation, scaling, one-hot encoding) is fitted inside scikit-learn pipelines on the training folds only.
