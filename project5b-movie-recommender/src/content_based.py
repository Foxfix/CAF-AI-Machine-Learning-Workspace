"""
Content-based filtering: recommend movies similar (by genre) to ones a
user rated highly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.data_loader import GENRE_COLUMNS


def build_genre_similarity(items: pd.DataFrame) -> np.ndarray:
    """Cosine similarity matrix (n_items x n_items) over the 19-dim
    genre one-hot vectors."""
    genre_matrix = items[GENRE_COLUMNS].values.astype(float)
    return cosine_similarity(genre_matrix)


def recommend_similar_movies(item_id: int, items: pd.DataFrame,
                              genre_sim: np.ndarray, k: int = 10) -> pd.DataFrame:
    """Return the top-k movies most similar in genre to the given item_id
    (excluding the movie itself)."""
    # items is 0-indexed by row but movie_id may have gaps, so map explicitly.
    id_to_row = {mid: i for i, mid in enumerate(items["movie_id"].values)}
    if item_id not in id_to_row:
        raise KeyError(f"movie_id {item_id} not found in items table.")
    row = id_to_row[item_id]

    sims = genre_sim[row].copy()
    sims[row] = -1  # exclude itself
    top_idx = np.argsort(-sims)[:k]

    result = items.iloc[top_idx][["movie_id", "movie_title"]].copy()
    result["similarity"] = sims[top_idx]
    return result.reset_index(drop=True)


def recommend_for_user_content_based(user_id: int, ratings: pd.DataFrame,
                                      items: pd.DataFrame, genre_sim: np.ndarray,
                                      k: int = 10, like_threshold: int = 4) -> pd.DataFrame:
    """Aggregate content-based recommendations across every movie the
    user rated >= like_threshold, weighted by their rating, then return
    the top-k movies the user hasn't already rated.

    This is also the cold-start-safe path: if the user has zero ratings
    (e.g. a brand-new user), liked_movies is empty and this function
    falls back to `recommend_popular` - see collaborative.py.
    """
    id_to_row = {mid: i for i, mid in enumerate(items["movie_id"].values)}
    user_ratings = ratings[ratings["user_id"] == user_id]
    liked = user_ratings[user_ratings["rating"] >= like_threshold]

    if liked.empty:
        # Explicit cold-start signal - caller should fall back to
        # popularity-based recommendations. Returning an empty frame,
        # not raising, so calling code can detect this state and act.
        return pd.DataFrame(columns=["movie_id", "movie_title", "score"])

    already_rated = set(user_ratings["item_id"])
    score = np.zeros(len(items))
    for _, r in liked.iterrows():
        row = id_to_row.get(r["item_id"])
        if row is None:
            continue
        score += genre_sim[row] * (r["rating"] / 5.0)

    order = np.argsort(-score)
    recs = []
    for idx in order:
        movie_id = items.iloc[idx]["movie_id"]
        if movie_id in already_rated:
            continue
        recs.append((movie_id, items.iloc[idx]["movie_title"], score[idx]))
        if len(recs) >= k:
            break

    return pd.DataFrame(recs, columns=["movie_id", "movie_title", "score"])
