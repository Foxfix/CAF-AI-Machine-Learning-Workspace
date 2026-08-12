# Project 3 - Classical Machine Learning: California Housing Price Prediction


## Dataset - source

- **Name:** California Housing dataset
- **Loader:** `sklearn.datasets.fetch_california_housing()` - part of scikit-learn itself, no manual download, no license ambiguity.
- **Verified via scikit-learn's own documentation** (https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html and the dataset description page https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/descr/california_housing.rst):
  - 20,640 samples, 8 numeric features: `MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude`.
  - Target: median house value per California census block group, in units of **$100,000** (a target of `4.526` means $452,600).
  - Derived from the 1990 U.S. Census (Pace & Barry, 1997), one row per census block group (typically 600–3,000 people).
  - `AveRooms`/`AveBedrms` are **per household**, not per block group - block groups with few households (e.g. vacation areas) can show unusually large values here. This is a real documented quirk, not a bug you need to "fix."
  - The target is right-censored: the original 1990 Census data top-codes house values, which shows up as a spike of block groups exactly at the maximum target value. Document this in EDA rather than silently treating it as a normal data point.
- Same underlying dataset used throughout Aurélien Géron's *Hands-On Machine Learning* (3rd ed.) - chosen deliberately so this project's structure can be compared against that reference without copying its code.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
jupyter lab notebooks/modeling.ipynb
```
Or run the pipeline as scripts:
```bash
python -m src.train      # trains + tunes models, saves the best one
python -m src.evaluate   # scores the held-out test set exactly once
```

## Repository structure
```
project2-classical-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── preprocess.py     # stratified split + feature engineering transformer
│   ├── train.py          # pipeline + GridSearchCV over 3 model families
│   └── evaluate.py       # one-time held-out test evaluation + residual plot
├── notebooks/
│   └── modeling.ipynb
└── reports/
    └── evaluation_report.md   # template - fill in with your actual numbers
```

## Deliverables checklist
- [ ] Confirmed `fetch_california_housing()` runs locally (see network note above)
- [ ] Stratified train/test split implemented (income-bucket stratification, not a naive random split)
- [ ] Baseline `LinearRegression` RMSE recorded
- [ ] Feature engineering (rooms_per_household, bedrooms_per_room, population_per_household) implemented and justified
- [ ] At least 3 model families compared via cross-validated search (Ridge, RandomForest, gradient boosting)
- [ ] Learning curve plotted and interpreted for the best model
- [ ] Held-out test set scored exactly once, `reports/evaluation_report.md` filled in with real numbers and a residual plot
