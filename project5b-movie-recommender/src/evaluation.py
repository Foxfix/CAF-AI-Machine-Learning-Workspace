"""
Evaluation: precision@k / recall@k for top-N recommendations, plus a
simple train/test split helper for ratings data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def train_test_split_ratings(ratings: pd.DataFrame, test_frac: float = 0.2,
                              random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random per-row split of the ratings table. Simple and sufficient
    for a learning project; a stricter alternative (leave-one-out per
    user, or a time-based split using `timestamp`) is a valid extension
    but not required here."""
    rng = np.random.default_rng(random_state)
    shuffled = ratings.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    n_test = int(len(shuffled) * test_frac)
    test = shuffled.iloc[:n_test]
    train = shuffled.iloc[n_test:]
    return train.reset_index(drop=True), test.reset_index(drop=True)


def build_relevant_sets(test_ratings: pd.DataFrame, relevance_threshold: int = 4) -> dict:
    """{user_id: set(item_ids the user rated >= relevance_threshold in
    the TEST split)}. Users with no such items are simply absent from
    the dict (they contribute nothing to precision/recall - handled by
    precision_recall_at_k, which skips users with no relevant items)."""
    relevant = test_ratings[test_ratings["rating"] >= relevance_threshold]
    return relevant.groupby("user_id")["item_id"].apply(set).to_dict()


def precision_recall_at_k(recommended: dict, relevant: dict, k: int = 10):
    """
    recommended: {user_id: [ranked list of recommended item_ids]}
    relevant:    {user_id: set of item_ids the user actually rated
                  >= threshold in the test set}
    Returns (mean_precision_at_k, mean_recall_at_k) across users who
    have at least 1 relevant item in the test set (users with none
    contribute no information to either metric and are excluded, not
    counted as zero, which would understate real performance).
    """
    precisions, recalls = [], []
    for user, rec_list in recommended.items():
        rel = relevant.get(user, set())
        if not rel:
            continue
        top_k = rec_list[:k]
        n_hits = len(set(top_k) & rel)
        precisions.append(n_hits / k)
        recalls.append(n_hits / len(rel))

    if not precisions:
        return 0.0, 0.0
    return float(np.mean(precisions)), float(np.mean(recalls))
