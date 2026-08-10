"""
Reusable load/clean functions for the NYC Airbnb EDA project.

Dataset: New York City Airbnb Open Data (AB_NYC_2019.csv)
Source:  https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
License: CC BY 4.0 (data), see README.md for download instructions.

This module is imported from notebooks/analysis.ipynb rather than having
its logic duplicated in notebook cells, so it's testable and reusable.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw.csv"

EXPECTED_COLUMNS = [
    "id", "name", "host_id", "host_name", "neighbourhood_group",
    "neighbourhood", "latitude", "longitude", "room_type", "price",
    "minimum_nights", "number_of_reviews", "last_review",
    "reviews_per_month", "calculated_host_listings_count",
    "availability_365",
]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw Airbnb CSV.

    Raises a clear, actionable error if the file hasn't been downloaded
    yet, per the README instructions, instead of a generic pandas
    FileNotFoundError with no context.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found.\n"
            "Download the dataset per README.md ('How to download the data') "
            "and place it at data/raw.csv before running this notebook."
        )
    df = pd.read_csv(path)

    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"raw.csv is missing expected columns: {sorted(missing_cols)}. "
            "This usually means the wrong file was downloaded — verify it's "
            "AB_NYC_2019.csv from the Kaggle page listed in README.md."
        )
    return df


def basic_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Return a one-row-per-column profiling table: dtype, % missing, n_unique."""
    return pd.DataFrame({
        "dtype": df.dtypes,
        "pct_missing": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
    })


def clean_listings(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the cleaning decisions documented in notebooks/analysis.ipynb
    (Milestone 2). Kept here — not duplicated in the notebook — so the
    logic is testable in isolation.

    Decisions made (documented, not silent):
    1. reviews_per_month is NaN exactly when number_of_reviews == 0
       (a listing with zero reviews has no review rate to report).
       We fill these with 0, since "no reviews yet" is a real, meaningful
       state, not missing information.
    2. Rows with price == 0 are treated as data-entry errors (a real
       listing cannot be free) and dropped.
    3. minimum_nights values above 365 are treated as outliers/likely
       data errors for a nightly-rental platform and dropped, rather
       than silently kept, which would distort minimum_nights analysis.
    4. neighbourhood_group values are stripped of whitespace and cast
       to a consistent category dtype.
    5. last_review is parsed to a real datetime; rows are otherwise
       untouched if this field is missing (a missing last_review is
       consistent with number_of_reviews == 0 and is not itself an error).
    """
    df = df.copy()

    # (1) reviews_per_month: NaN <=> zero reviews. Verify the assumption,
    # then fill.
    zero_review_mask = df["number_of_reviews"] == 0
    nan_reviews_per_month_mask = df["reviews_per_month"].isna()
    mismatch = (zero_review_mask != nan_reviews_per_month_mask).sum()
    if mismatch > 0:
        # Not fatal — just means the assumption isn't 100% clean in this
        # snapshot of the data. We still fill NaNs with 0, but this is
        # surfaced so the notebook can report it rather than hide it.
        print(
            f"[clean_listings] Note: {mismatch} rows have "
            "reviews_per_month NaN status inconsistent with "
            "number_of_reviews == 0. Filling reviews_per_month NaNs with "
            "0 regardless; documented in analysis notebook."
        )
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)

    # (2) Drop price == 0 (free listings are not realistic on this platform).
    n_before = len(df)
    df = df[df["price"] > 0]
    n_dropped_price = n_before - len(df)

    # (3) Drop minimum_nights outliers above 365.
    n_before = len(df)
    df = df[df["minimum_nights"] <= 365]
    n_dropped_min_nights = n_before - len(df)

    print(
        f"[clean_listings] Dropped {n_dropped_price} rows with price == 0, "
        f"{n_dropped_min_nights} rows with minimum_nights > 365."
    )

    # (4) Normalize neighbourhood_group.
    df["neighbourhood_group"] = (
        df["neighbourhood_group"].astype(str).str.strip().astype("category")
    )
    df["room_type"] = df["room_type"].astype(str).str.strip().astype("category")

    # (5) Parse last_review as a real datetime; missing stays missing.
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")

    return df.reset_index(drop=True)


def price_band(price: pd.Series, bins=(0, 75, 150, 300, np.inf),
                labels=("budget", "mid", "high", "luxury")) -> pd.Series:
    """Bucket price into readable bands for grouped charts (e.g. the
    price-by-borough boxplot in the notebook)."""
    return pd.cut(price, bins=bins, labels=labels)
