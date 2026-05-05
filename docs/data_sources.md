# Data Sources

## Overview

The final rerun uses four source types: full SABR/Lahman, Kaggle, pybaseball, and BeautifulSoup. The source manifest confirms that the main modeling tables are taken from SABR/Lahman in the final rerun.

## 1. SABR / Lahman 1871-2025 primary source

**Location:** `data/raw/sabr_lahman/`

The full SABR/Lahman release contains 27 CSV files. In the final rerun, `scripts/01a_data_acquisition_part1.py` builds `data/raw/analysis_input/` from the SABR/Lahman tables. `People.csv` is mapped to `Master.csv` so the rest of the project can use a consistent table name.

Important source rows from the final manifest:

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

## 2. Kaggle Baseball Databank backup source

**Location:** `data/raw/kaggle/`

The Kaggle data are preserved as an audit and backup source. The current final run does not rely on Kaggle for the main analysis tables because full SABR/Lahman files are available.

## 3. pybaseball external acquisition

**Location:** `data/raw/pybaseball/`

The final run saved pybaseball examples: a player-ID lookup output and a small Statcast sample. These files demonstrate programmatic acquisition from a modern baseball data package. They are used as supplemental data-collection evidence rather than as core predictors in the final Hall of Fame model.

## 4. BeautifulSoup web scrape

**Location:** `data/raw/scraped/`

The final run saved a focused scrape of the Baseball Hall of Fame Future Eligibles page. The scrape produced 142 text rows. It is used as semi-structured web-scraping evidence and a future-extension source. The Shiny app itself reads `app/model_predictions.csv`, not the scraped page directly.

## Source manifest files

| File | Purpose |
|---|---|
| `reports/tables/part1_source_manifest.csv` | Shows which source was used for each analysis table. |
| `reports/tables/part1_external_source_status.csv` | Shows pybaseball and BeautifulSoup acquisition status. |
| `data/raw/source_manifest.json` | JSON version of the source manifest. |
