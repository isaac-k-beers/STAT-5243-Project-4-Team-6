# Final Report Draft — MLB Hall of Fame Prediction + Player Archetype Clustering

## 1. Problem Statement

We ask whether career performance, awards, biographical attributes, era-adjusted statistics, and unsupervised player archetypes can predict whether an eligible MLB player is inducted into the Baseball Hall of Fame. The target is `inducted` (1 = inducted, 0 = eligible but not inducted).

The project story is: **what separates Cooperstown legends from very good players?**

## 2. Data Collection and Preparation

The project combines four sources: the full SABR/Lahman 1871-2025 CSV release as the primary relational baseball database, Kaggle Baseball Databank as an audit and backup source, pybaseball ID and Statcast samples as programmatic baseball-data acquisition examples, and a BeautifulSoup scrape of the Hall of Fame Future Eligibles page as a semi-structured web source.

The acquisition step maps `People.csv` to the project-compatible `Master.csv` name, joins relational tables mainly on `playerID`, and uses `Appearances.csv` as the canonical source for primary-position construction. The main data challenge is that raw baseball data are fragmented across player-team-season rows, so the pipeline aggregates team stints and constructs one career-level row per player.

The final base player feature table contains **24,270 players and 260 columns**. The eligible modeling pool contains **3,738 players**, including **281 inducted positives**.

## 3. EDA and Unsupervised Learning

EDA includes dataset-level summary statistics, class imbalance, missingness analysis, role and era comparisons, selected correlations, hypothesis tests, VIF diagnostics, influence diagnostics, and class-conditional achievement distributions. The headline induction rate in the eligible modeling pool is **0.0752**.

K-means clustering, hierarchical clustering diagnostics, and PCA are used to discover player archetypes. These archetype features are merged into the supervised modeling dataset.

### Archetype profile summary

|   cluster_id | cluster_label              |   players |   hof_rate |   median_hr |   median_ops |   median_wins |   median_saves |   median_awards |   median_allstar |
|-------------:|:---------------------------|----------:|-----------:|------------:|-------------:|--------------:|---------------:|----------------:|-----------------:|
|            0 | Power Hitter               |       529 |  0.300567  |         224 |     0.805708 |             0 |              0 |               8 |                4 |
|            1 | Dominant Reliever / Closer |       463 |  0.0194384 |           0 |     0.285714 |            43 |             47 |               1 |                2 |
|            2 | Speed / Contact Profile    |        30 |  0.0333333 |          37 |     0.71389  |             0 |              0 |               3 |                2 |
|            3 | Power Hitter 2             |      1912 |  0.0209205 |          38 |     0.703664 |             1 |              0 |               1 |                1 |
|            4 | Workhorse Starting Pitcher |       804 |  0.0895522 |           1 |     0.399224 |           121 |              5 |               2 |                2 |


## 4. Feature Engineering and Preprocessing

Feature groups include career volume, rate statistics, peak 3-year and 5-year windows, awards, All-Star counts, postseason summaries, era-adjusted OPS/ERA proxies, biographical variables, role and position variables, cluster IDs, PCA coordinates, and distance-to-centroid features. Preprocessing includes imputation, one-hot encoding, standardization, min-max scaling, power transformation, and leakage-aware feature exclusions. The final archetype modeling table contains **24,270 rows and 270 columns**.

## 5. Model Development

Validation strategy: **Time-based holdout: train eligibility_year < 2000; test eligibility_year >= 2000**. The training set contains **2,439 players** with **232 positives**. The holdout set contains **1,299 players** with **49 positives**.

The actual reported models are: **gradient_boosting, stacking, random_forest, logistic_regression**. Hyperparameter tuning is evaluated with cross-validated average precision, and model selection emphasizes PR AUC because Hall of Fame induction is a rare positive class.

## 6. Model Comparison and Selection

| model               |   pr_auc |   roc_auc |   precision |   recall |       f1 |   balanced_accuracy |   brier_score |   top_25_precision |   top_50_precision |
|:--------------------|---------:|----------:|------------:|---------:|---------:|--------------------:|--------------:|-------------------:|-------------------:|
| gradient_boosting   | 0.727281 |  0.989845 |    0.696429 | 0.795918 | 0.742857 |            0.891159 |     0.0165117 |               0.84 |               0.72 |
| stacking            | 0.72118  |  0.992098 |    0.295181 | 1        | 0.455814 |            0.9532   |     0.0665042 |               0.76 |               0.82 |
| random_forest       | 0.705554 |  0.991478 |    0.619718 | 0.897959 | 0.733333 |            0.93818  |     0.0252146 |               0.72 |               0.8  |
| logistic_regression | 0.571217 |  0.982465 |    0.231132 | 1        | 0.375479 |            0.9348   |     0.0998615 |               0.68 |               0.6  |


The selected model is **gradient_boosting**, chosen primarily by holdout PR AUC. It achieved **PR AUC 0.727281** and **ROC AUC 0.989845** on the time-based holdout set.

## 7. Interpretation and Communication

The most important feature groups include awards, All-Star appearances, peak performance, career totals, cluster features, and era-adjusted metrics. This suggests that Hall of Fame induction is associated with both statistical production and historical recognition. The Shiny app, **Cooperstown Calculator**, reads `app/model_predictions.csv` and lets users inspect player-level induction probabilities.

## 8. Limitations and Future Work

This is an observational prediction project, so the results describe associations rather than causal effects. Hall voting behavior changes across eras, early baseball records can be noisier, salary coverage starts in 1985, and modern advanced metrics such as WAR/JAWS are not fully integrated. Future work could incorporate richer WAR/JAWS sources, expand pybaseball Statcast features, and model time from retirement to induction with survival analysis.

## 9. Member Contributions

Replace this section with actual team member contributions before final submission.
