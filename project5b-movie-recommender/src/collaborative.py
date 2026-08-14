"""
Collaborative filtering (item-based CF via cosine similarity on the
sparse ratings matrix) + a matrix-factorization alternative (TruncatedSVD)
+ popularity fallback for cold-start users.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


def item_item_similarity(ratings_matrix: csr_matrix) -> np.ndarray:
    """Cosine similarity between items (columns), computed on the sparse
    matrix directly - never densify the full user-item matrix for this,
    only the resulting (n_items x n_items) similarity matrix, which is
    much smaller (1682x1682 vs 943x1682)."""
    return cosine_similarity(ratings_matrix.T)


def recommend_item_based(user_row: int, ratings_matrix: csr_matrix,
                          item_sim: np.ndarray, col_to_item_id: dict,
                          k: int = 10) -> list[tuple[int, float]]:
    """Item-based CF: score each unrated item by the similarity-weighted
    sum of the user's ratings on similar items. Returns [(item_id, score), ...]
    sorted descending, excluding already-rated items."""
    user_ratings = ratings_matrix[user_row].toarray().ravel()
    rated_mask = user_ratings > 0

    if not rated_mask.any():
        return []  # cold start - caller should fall back to popularity

    # score[j] = sum_i sim(i, j) * rating[i]  for all rated items i
    scores = item_sim[:, rated_mask] @ user_ratings[rated_mask]
    # normalize by sum of similarities to keep scale comparable across items
    sim_sums = item_sim[:, rated_mask].sum(axis=1)
    sim_sums[sim_sums == 0] = 1e-9
    scores = scores / sim_sums

    scores[rated_mask] = -np.inf  # never recommend already-rated items
    top_cols = np.argsort(-scores)[:k]

    return [(col_to_item_id[c], float(scores[c])) for c in top_cols
            if scores[c] > -np.inf]


def fit_svd(ratings_matrix: csr_matrix, n_components: int = 20, random_state: int = 42):
    """Matrix factorization alternative to item-based CF. Returns the
    fitted TruncatedSVD model and the user-factor matrix."""
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    user_factors = svd.fit_transform(ratings_matrix)
    return svd, user_factors


def recommend_svd(user_row: int, ratings_matrix: csr_matrix, svd: TruncatedSVD,
                   user_factors: np.ndarray, col_to_item_id: dict,
                   k: int = 10) -> list[tuple[int, float]]:
    """Reconstruct predicted ratings for one user from the SVD factors,
    exclude already-rated items, return top-k."""
    predicted = user_factors[user_row] @ svd.components_
    rated_mask = ratings_matrix[user_row].toarray().ravel() > 0
    predicted[rated_mask] = -np.inf
    top_cols = np.argsort(-predicted)[:k]
    return [(col_to_item_id[c], float(predicted[c])) for c in top_cols]


def recommend_popular(ratings: pd.DataFrame, items: pd.DataFrame, k: int = 10,
                       min_ratings: int = 20) -> pd.DataFrame:
    """Cold-start fallback: most popular movies by rating count among
    movies with at least `min_ratings` ratings, ranked by mean rating.
    This is what a brand-new user with zero ratings gets - the system
    must never silently fail or return nothing for this case."""
    agg = ratings.groupby("item_id")["rating"].agg(["mean", "count"])
    agg = agg[agg["count"] >= min_ratings].sort_values("mean", ascending=False)
    top_ids = agg.head(k).index
    result = items[items["movie_id"].isin(top_ids)][["movie_id", "movie_title"]].copy()
    result = result.merge(agg[["mean", "count"]], left_on="movie_id", right_index=True)
    return result.sort_values("mean", ascending=False).reset_index(drop=True)
