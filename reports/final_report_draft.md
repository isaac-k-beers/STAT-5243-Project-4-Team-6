# Final Report Draft — MLB Hall of Fame Prediction + Player Archetype Clustering

## 1. Problem Statement

We ask whether career performance, awards, biographical attributes, era-adjusted statistics, and unsupervised player archetypes can predict whether an eligible MLB player is inducted into the Baseball Hall of Fame. The target is `inducted` (1 = inducted, 0 = eligible but not inducted).

## 2. Data Collection and Preparation

The project combines four sources: the full SABR / Lahman 1871–2025 CSV release as the primary relational baseball database, the Kaggle Baseball Databank as an audit / backup source, pybaseball ID and Statcast samples as a programmatic baseball-data source, and a BeautifulSoup scrape of the Hall of Fame Future Eligibles page as a semi-structured web source. We map `People.csv` to the project-compatible `Master.csv` name, join 15 relational tables on `playerID`, and use `Appearances.csv` as the canonical source for primary-position construction.

## 3. EDA and Unsupervised Learning

EDA includes class imbalance, missingness, era patterns, role distributions, selected correlations, and achievement distributions. K-means clustering, hierarchical clustering, and PCA are used to discover player archetypes and engineer cluster features.

### Archetype profile summary

|   cluster_id | cluster_label              |   players |   hof_rate |   median_hr |   median_ops |   median_wins |   median_saves |   median_awards |   median_allstar |
|-------------:|:---------------------------|----------:|-----------:|------------:|-------------:|--------------:|---------------:|----------------:|-----------------:|
|            0 | Power Hitter               |       529 |  0.300567  |         224 |     0.805708 |             0 |              0 |               8 |                4 |
|            1 | Dominant Reliever / Closer |       463 |  0.0194384 |           0 |     0.285714 |            43 |             47 |               1 |                2 |
|            2 | Speed / Contact Profile    |        30 |  0.0333333 |          37 |     0.71389  |             0 |              0 |               3 |                2 |
|            3 | Power Hitter 2             |      1912 |  0.0209205 |          38 |     0.703664 |             1 |              0 |               1 |                1 |
|            4 | Workhorse Starting Pitcher |       804 |  0.0895522 |           1 |     0.399224 |           121 |              5 |               2 |                2 |


## 4. Feature Engineering and Preprocessing

Feature groups include career volume, rate statistics, peak 3-year and 5-year windows, awards, All-Star counts, postseason summaries, era-adjusted OPS/ERA proxies, biographical variables, role/position variables, cluster IDs, PCA coordinates, and distance-to-centroid features. Preprocessing uses median imputation, categorical imputation, one-hot encoding, standardization, and leakage-aware feature exclusions.

## 5. Model Development

Validation strategy: Time-based holdout: train eligibility_year < 2000; test eligibility_year >= 2000. Training set n = 2439 with 232 positives; holdout n = 1299 with 49 positives. Models include tuned Logistic Regression, Random Forest, Gradient Boosting, XGBoost, and a Stacking ensemble.

## 6. Model Comparison and Selection

| model               |   pr_auc |   roc_auc |   precision |   recall |       f1 |   balanced_accuracy |   brier_score |   top_25_precision |
|:--------------------|---------:|----------:|------------:|---------:|---------:|--------------------:|--------------:|-------------------:|
| gradient_boosting   | 0.727281 |  0.989845 |    0.696429 | 0.795918 | 0.742857 |            0.891159 |     0.0165117 |               0.84 |
| stacking            | 0.72118  |  0.992098 |    0.295181 | 1        | 0.455814 |            0.9532   |     0.0665042 |               0.76 |
| random_forest       | 0.705554 |  0.991478 |    0.619718 | 0.897959 | 0.733333 |            0.93818  |     0.0252146 |               0.72 |
| logistic_regression | 0.571217 |  0.982465 |    0.231132 | 1        | 0.375479 |            0.9348   |     0.0998615 |               0.68 |


The selected model is **gradient_boosting**, chosen primarily by holdout PR AUC and secondarily by ROC AUC, calibration, interpretability, and robustness across roles/eras.

## 7. Interpretation and Communication

The report should emphasize the story: what separates legends from very good players? Interpret the selected model using permutation importance, PR/ROC curves, calibration, cluster profiles, and borderline-player examples.

## 8. Limitations and Future Work

Key limitations include incomplete modern data in Kaggle, era bias, historical changes in Hall voting behavior, salary missingness before 1985, and target timing for recently eligible players. Future work can incorporate latest SABR/Lahman tables, pybaseball WAR/Statcast features, and survival analysis for time-to-induction.

## 9. Member Contributions

Replace this section with your actual team contributions before submission.
