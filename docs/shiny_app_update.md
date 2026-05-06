# Shiny App Update

The Shiny app was upgraded from a simple single-player probability viewer into a multi-tab interactive dashboard.

## Main improvements

1. **Overview dashboard**
   - Eligible-player count.
   - Inductee count.
   - Best model and PR AUC.
   - Model leaderboard plot.
   - Source-coverage table.

2. **Player Explorer**
   - Searchable player selector.
   - Predicted Hall of Fame probability.
   - Actual Hall of Fame status.
   - Player role, position, era, and archetype.
   - Career percentile chart.
   - Similar historical players based on career profile.

3. **Rankings**
   - Filter by role, archetype, actual induction status, and predicted probability.
   - Display top-ranked eligible players after filtering.
   - Prediction-distribution chart.

4. **Archetype Explorer**
   - Cluster-profile table.
   - Top players in each archetype.
   - Prediction distribution within archetype.
   - PCA archetype figure from the Python pipeline.

5. **What-if Similarity**
   - Users can enter a custom statistical profile.
   - The app finds nearest historical players and reports the average model probability and actual induction rate among comparable players.
   - This is an interpretation and similarity tool, not live model inference.

6. **Model & Methods**
   - Model comparison table.
   - Feature-importance table.
   - Precision-recall and calibration figures.
   - Short interpretation notes.

## Report wording

Suggested report sentence:

> We also built an enhanced R Shiny dashboard, the Cooperstown Calculator, to communicate the model in an interactive format. The app allows users to search individual players, compare Hall of Fame probabilities, inspect archetype-based similar players, filter ranked candidates, and explore model-performance outputs such as PR AUC, calibration, and feature importance. This turns the final prediction table into a user-facing communication product rather than a static output.
