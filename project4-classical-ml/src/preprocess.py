"""
Stratified split + feature engineering for the California Housing project.

Dataset: sklearn.datasets.fetch_california_housing()
Source verified: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html
See README.md for the "no internet in this sandbox" execution note.
https://codeaiflow.cloud/
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

TARGET_COL = "MedHouseVal"


def load_dataset() -> pd.DataFrame:
    """Load California Housing as a single DataFrame (features + target).

    Requires internet on first call (scikit-learn downloads and caches
    the raw data under ~/scikit_learn_data/). Subsequent calls read from
    the local cache.
    """
    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame  # already includes the target column MedHouseVal
    assert TARGET_COL in df.columns, (
        f"Expected target column '{TARGET_COL}' not found — scikit-learn "
        "may have changed the frame layout; check your installed version."
    )
    return df


def make_income_bucket(df: pd.DataFrame) -> pd.Series:
    """Bucket MedInc into 5 strata for a leakage-safe stratified split.

    Bin edges chosen so each bucket has a meaningful number of rows for
    this dataset's MedInc distribution (units of $10,000s, e.g. 3.5 = $35k).
    """
    return pd.cut(
        df["MedInc"],
        bins=[0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )


def stratified_split(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return X_train, X_test, y_train, y_test using an income-stratified
    split. A naive random split under-samples high-income block groups
    relative to their real-world proportion — this is the leakage/bias
    trap Module 3.1 (Validation Methodology) specifically calls out.
    """
    strata = make_income_bucket(df)
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=strata, random_state=random_state
    )
    X_train = train_df.drop(columns=[target_col])
    X_test = test_df.drop(columns=[target_col])
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    return X_train, X_test, y_train, y_test


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds three derived ratio features:
      - rooms_per_household   = AveRooms / AveOccup
      - bedrooms_per_room     = AveBedrms / AveRooms
      - population_per_household = Population / AveOccup  (approximation;
        AveOccup is itself population/households, so this recovers an
        implied household count scale — documented in the eval report
        as a modeling choice, not a precise household count).

    Stateless (no .fit() computation needed) so it's safe to place inside
    a sklearn.Pipeline without leaking test-set statistics — the ratios
    are computed per-row from existing columns, not fit on training data.
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # Avoid division by zero on the (rare) AveOccup == 0 rows.
        safe_occup = X["AveOccup"].replace(0, np.nan)
        safe_rooms = X["AveRooms"].replace(0, np.nan)

        X["rooms_per_household"] = X["AveRooms"] / safe_occup
        X["bedrooms_per_room"] = X["AveBedrms"] / safe_rooms
        X["population_per_household"] = X["Population"] / safe_occup

        # Any row where the ratio couldn't be computed (division by zero)
        # gets the column median rather than being dropped — dropping
        # here would silently shrink train/test set sizes inconsistently.
        for col in ["rooms_per_household", "bedrooms_per_room",
                    "population_per_household"]:
            X[col] = X[col].fillna(X[col].median())
        return X
