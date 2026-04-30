# Data Dictionary

## Data Source

This project uses the Baseball Databank / Lahman baseball dataset downloaded from Kaggle.

## Raw Tables Used

| Table | Purpose |
|---|---|
| `Master.csv` | Player demographic information |
| `Batting.csv` | Player batting statistics by season/team |
| `Salaries.csv` | Player salary information |
| `HallOfFame.csv` | Hall of Fame voting and induction data |
| `Pitching.csv` | Loaded for future modeling/team use |
| `Fielding.csv` | Loaded for future modeling/team use |
| `Teams.csv` | Loaded for future team/context features |
| `AwardsPlayers.csv` | Loaded for future award-related features |
| `AllstarFull.csv` | Loaded for future All-Star related features |

## Unit of Analysis

The cleaned dataset is structured at the **player-year level**. Each row represents one MLB player’s aggregated performance in one season.

## Cleaning and Preparation Decisions

- Batting statistics were aggregated across multiple team stints within the same season.
- Players with missing or zero at-bats were removed because batting rate statistics would be undefined.
- Players with fewer than 50 at-bats in a season were removed to reduce noise from extremely small samples.
- Missing salary values were retained because salary data is not available for all historical seasons.
- Hall of Fame induction was converted into a binary indicator.
- Infinite values from rate-stat calculations were replaced with missing values.

## Engineered Features

| Feature | Definition |
|---|---|
| `age` | `yearID - birthYear` |
| `1B` | Singles, calculated as `H - 2B - 3B - HR` |
| `BA` | Batting average, `H / AB` |
| `OBP_proxy` | Approximate on-base percentage |
| `SLG` | Slugging percentage |
| `OPS_proxy` | `OBP_proxy + SLG` |
| `HR_rate` | Home runs per at-bat |
| `BB_rate` | Walks per at-bat |
| `SO_rate` | Strikeouts per at-bat |
| `RBI_rate` | Runs batted in per at-bat |
| `SB_rate` | Stolen bases per at-bat |
| `career_season_number` | Player’s season count up to that year |
| `career_longevity` | Number of seasons in the cleaned dataset |
| `career_AB`, `career_H`, `career_HR`, etc. | Cumulative career totals through the current season |

## Target Variables

| Target | Type | Definition |
|---|---|---|
| `BA_next` | Regression | Player’s batting average in the following season |
| `HR_next` | Regression | Player’s home runs in the following season |
| `OPS_next` | Regression | Player’s OPS proxy in the following season |
| `power_hitter_next` | Classification | `1` if player hits at least 20 HR next season, else `0` |

## Output Files

| File | Description |
|---|---|
| `data/processed/master_player_features.parquet` | Full cleaned player-year dataset |
| `data/processed/modeling_player_features.parquet` | Modeling-ready subset with valid next-season targets |

## Leakage Prevention

Next-season targets were created using player-level group shifting. This means features from season `t` are used to predict outcomes from season `t+1`, preventing same-season target leakage.