# Evaluation Report - California Housing Price Prediction

> Template. Fill in every `[ ]` with your actual numbers from `notebooks/modeling.ipynb`
> after running Milestone 6. Do not fill this in with placeholder/invented numbers -
> leave a field blank and mark `TODO` if you haven't run that step yet.

## 1. Problem framing
Predict median house value (in $100,000 units) for a California census block group, from 8 census-derived features (income, age, rooms, occupancy, location). Source: `sklearn.datasets.fetch_california_housing()`, 1990 U.S. Census data (Pace & Barry, 1997).

## 2. Data notes
- 20,640 rows, 8 features, no missing values in the base sklearn version (unlike the raw StatLib CSV some other tutorials use, which does have missing `total_bedrooms` values - worth knowing these are not identical files).
- Target is right-censored at the top-coded Census value - [ ] document how many test-set rows sit exactly at the maximum target value, since the model cannot outperform this data limitation.

## 3. Models compared

| Model | CV RMSE (mean) | CV RMSE (std) | Best hyperparameters |
|---|---|---|---|
| Baseline (LinearRegression) | [ ] | - | - |
| Ridge | [ ] | [ ] | [ ] |
| RandomForestRegressor | [ ] | [ ] | [ ] |
| HistGradientBoostingRegressor | [ ] | [ ] | [ ] |

## 4. Selected model
**[ TODO: name of best model ]**, selected on cross-validated RMSE.

## 5. Final test-set metrics (scored exactly once)
- RMSE: **[ ]** ($100k units - multiply by 100,000 for dollar error)
- MAE: **[ ]**
- R²: **[ ]**

## 6. Residual analysis
[ ] Insert `reports/residual_plot.png` here or describe it.
[ ] Where does the model do worst? (e.g. very high-value block groups, coastal areas, censored-target rows?)

## 7. Learning curve diagnosis
[ ] Under-fitting / over-fitting / well-fit? What would you try next with more time (more features, more data, different model family)?

## 8. Honest limitations
- This is 1990 Census data - not representative of the current California housing market. State this explicitly if this project is shown to anyone as a "prediction" tool.
- [ ] Any other limitations you found during the project.
