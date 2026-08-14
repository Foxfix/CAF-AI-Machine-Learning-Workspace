"""
Customer segmentation utilities for the Mall Customer Segmentation Data
(https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python).

Expected raw columns: CustomerID, Gender, Age, Annual Income (k$),
Spending Score (1-100). See README.md for download instructions and the
network-access caveat.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "Gender", "Age", "Annual Income (k$)", "Spending Score (1-100)",
]

FEATURE_COLUMNS = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw Mall Customer CSV, with a clear error if it hasn't
    been downloaded yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the dataset per README.md "
            "before running this notebook."
        )
    df = pd.read_csv(path)
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"raw.csv is missing expected columns: {sorted(missing)}. "
            "Verify you downloaded Mall_Customers.csv from the Kaggle "
            "page listed in README.md, not a different file."
        )
    return df


def scale_features(df: pd.DataFrame, feature_cols: list[str] = FEATURE_COLUMNS):
    """Return (X_scaled, fitted_scaler). Scaling matters here because
    Annual Income (tens of thousands) and Spending Score (1-100) are on
    very different numeric scales — unscaled K-Means would let income
    dominate the distance metric almost entirely."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler


def scan_k(X_scaled: np.ndarray, k_range=range(2, 11)):
    """Return (list_of_k, inertias, silhouette_scores) for each k in
    k_range, so the notebook can plot the elbow method and silhouette
    score side by side rather than relying on only one heuristic."""
    ks, inertias, sil_scores = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        ks.append(k)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))
    return ks, inertias, sil_scores


def fit_kmeans(X_scaled: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    km.fit(X_scaled)
    return km


def profile_clusters(df: pd.DataFrame, labels: np.ndarray,
                      feature_cols: list[str] = FEATURE_COLUMNS) -> pd.DataFrame:
    """Return a groupby summary table (mean per feature per cluster, plus
    cluster size) for interpretation. This is the table the notebook
    uses to name each cluster with a business persona."""
    profile_df = df.copy()
    profile_df["cluster"] = labels
    summary = profile_df.groupby("cluster")[feature_cols].mean().round(1)
    summary["n_customers"] = profile_df.groupby("cluster").size()
    if "Gender" in profile_df.columns:
        summary["pct_female"] = (
            profile_df.groupby("cluster")["Gender"]
            .apply(lambda s: (s == "Female").mean() * 100)
            .round(1)
        )
    return summary.sort_index()
