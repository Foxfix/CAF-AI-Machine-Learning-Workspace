"""
Tests run against SYNTHETIC data matching the real Mall Customer
Segmentation Data schema (200 rows, CustomerID/Gender/Age/Annual Income
(k$)/Spending Score (1-100)) - used only to prove the clustering_utils
logic actually runs correctly end to end. This is NOT a substitute for
running the notebook against the real downloaded dataset.

Run: python -m pytest tests/ -v
   or: python tests/test_clustering_utils.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clustering_utils import (
    EXPECTED_COLUMNS,
    FEATURE_COLUMNS,
    fit_kmeans,
    profile_clusters,
    scale_features,
    scan_k,
)


def make_synthetic_mall_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic data with the SAME column names/ranges as the real
    dataset (Age 18-70, Annual Income 15-140 k$, Spending Score 1-100),
    with 3 built-in blob-like groups so clustering has something real
    to find - this validates the pipeline runs, it does not validate
    real-world cluster interpretations."""
    rng = np.random.default_rng(seed)
    groups = []
    centers = [(25, 30, 80), (45, 60, 40), (35, 100, 20)]  # age, income, spending
    per_group = n // len(centers)
    for age_c, inc_c, spend_c in centers:
        ages = rng.normal(age_c, 5, per_group).clip(18, 70)
        incomes = rng.normal(inc_c, 10, per_group).clip(15, 140)
        spending = rng.normal(spend_c, 10, per_group).clip(1, 100)
        groups.append(pd.DataFrame({"Age": ages, "Annual Income (k$)": incomes,
                                     "Spending Score (1-100)": spending}))
    df = pd.concat(groups, ignore_index=True)
    df["CustomerID"] = range(1, len(df) + 1)
    df["Gender"] = rng.choice(["Male", "Female"], size=len(df))
    return df[EXPECTED_COLUMNS]


def test_schema_matches():
    df = make_synthetic_mall_data()
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 198  # 66 * 3, from integer division of n=200
    print("test_schema_matches: OK")


def test_scale_features():
    df = make_synthetic_mall_data()
    X_scaled, scaler = scale_features(df)
    assert X_scaled.shape == (len(df), len(FEATURE_COLUMNS))
    # scaled features should have ~mean 0, ~std 1
    assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-8)
    print("test_scale_features: OK, mean~0 std~1 confirmed")


def test_scan_k_returns_sane_values():
    df = make_synthetic_mall_data()
    X_scaled, _ = scale_features(df)
    ks, inertias, sil_scores = scan_k(X_scaled, k_range=range(2, 7))
    assert len(ks) == len(inertias) == len(sil_scores) == 5
    # inertia must strictly decrease as k increases (more clusters -> tighter fit)
    assert all(inertias[i] > inertias[i + 1] for i in range(len(inertias) - 1)), \
        "inertia should monotonically decrease with k"
    # silhouette scores must all be valid (-1, 1)
    assert all(-1 <= s <= 1 for s in sil_scores)
    print(f"test_scan_k_returns_sane_values: OK, inertias={[round(i) for i in inertias]}")
    print(f"  silhouette scores={[round(s, 3) for s in sil_scores]}")


def test_kmeans_recovers_known_groups():
    """Since the synthetic data was built with 3 separated blobs, KMeans
    with k=3 should recover roughly the group sizes we built in
    (66/66/66) - proves the clustering pipeline actually clusters,
    not just runs without crashing."""
    df = make_synthetic_mall_data()
    X_scaled, _ = scale_features(df)
    km = fit_kmeans(X_scaled, k=3)
    _, counts = np.unique(km.labels_, return_counts=True)
    counts_sorted = sorted(counts)
    print(f"test_kmeans_recovers_known_groups: cluster sizes = {counts_sorted}")
    # allow some slack since the blobs have overlapping tails
    assert all(40 <= c <= 90 for c in counts_sorted), \
        f"expected roughly balanced ~66/66/66 clusters, got {counts_sorted}"
    print("test_kmeans_recovers_known_groups: OK")


def test_profile_clusters():
    df = make_synthetic_mall_data()
    X_scaled, _ = scale_features(df)
    km = fit_kmeans(X_scaled, k=3)
    profile = profile_clusters(df, km.labels_)
    assert len(profile) == 3
    assert "n_customers" in profile.columns
    assert profile["n_customers"].sum() == len(df)
    assert "pct_female" in profile.columns
    print("test_profile_clusters: OK")
    print(profile)


if __name__ == "__main__":
    test_schema_matches()
    test_scale_features()
    test_scan_k_returns_sane_values()
    test_kmeans_recovers_known_groups()
    test_profile_clusters()
    print("\nALL TESTS PASSED on synthetic data matching the real schema.")
