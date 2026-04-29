# Data Dictionary

## Data Sources
- Lahman Baseball Database (Kaggle)

## Tables Used
- Master: player demographics
- Batting: offensive statistics
- Salaries: player salary data
- HallOfFame: induction status

## Unit of Analysis
- Player-year

## Key Features
- BA: Batting Average
- OBP_proxy: On-base percentage approximation
- SLG: Slugging percentage
- OPS_proxy: Combined offensive metric
- HR_rate, BB_rate, SO_rate

## Targets
- BA_next: next-season batting average (regression)
- HR_next: next-season home runs (regression)
- power_hitter_next: 1 if HR_next ≥ 20 (classification)

## Notes
- Aggregated across multiple stints per season
- Missing values retained for modeling stage handling
- Targets created using lagged grouping (no data leakage)