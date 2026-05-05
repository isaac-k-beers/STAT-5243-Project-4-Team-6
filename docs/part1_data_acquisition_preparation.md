# Part 1 — Data Acquisition and Preparation

## Goal

Convert fragmented Lahman/Baseball Databank tables into one clean player-level career table for Hall of Fame modeling.

## Sources used in the final rerun

| Source | Actual output |
|---|---|
| SABR / Lahman 1871-2025 | 27 CSV files present; main source for all reconciled analysis tables. |
| Kaggle Baseball Databank | Backup and audit source. |
| pybaseball | 4 player-ID lookup rows and 314 Statcast sample rows. |
| BeautifulSoup | 142 scraped text rows from the Hall of Fame Future Eligibles page. |

## Reconciled analysis tables

The acquisition script writes the analysis-ready raw tables to `data/raw/analysis_input/`. `People.csv` is mapped to `Master.csv` for compatibility with the rest of the project.

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

## Cleaning and alignment problems

The raw tables are fragmented across player-team-season rows. A player can appear multiple times in one year if he played for multiple teams. The project therefore aggregates team stints, aligns career-level features, and builds one row per player.

Important cleaning tasks include:

- mapping `People.csv` to `Master.csv`;
- aggregating batting, pitching, fielding, awards, and postseason tables;
- deriving primary position from `Appearances.csv`;
- creating Hall of Fame labels from `HallOfFame.csv`;
- constructing eligibility-aligned modeling labels;
- handling historical missingness and salary coverage gaps;
- preventing leakage from Hall voting variables.

## Main output

| Output | Actual shape |
|---|---:|
| `data/processed/player_features_base.csv` | 24,270 rows × 260 columns |

This table is the input for EDA, clustering, feature audit, and supervised modeling.
