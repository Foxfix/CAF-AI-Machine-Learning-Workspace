"""
Loaders for the MovieLens 100K raw files (u.data, u.item).

Format verified against GroupLens' own README for ml-100k
(https://files.grouplens.org/datasets/movielens/ml-100k.zip):
- u.data: tab-separated, no header: user_id, item_id, rating, timestamp
- u.item: pipe-separated, no header, 24 columns:
    movie_id | movie_title | release_date | video_release_date | IMDb_URL |
    then 19 genre one-hot flags (0/1) in this fixed order:
    unknown, Action, Adventure, Animation, Children's, Comedy, Crime,
    Documentary, Drama, Fantasy, Film-Noir, Horror, Musical, Mystery,
    Romance, Sci-Fi, Thriller, War, Western
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "ml-100k"

GENRE_COLUMNS = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]

ITEM_COLUMNS = ["movie_id", "movie_title", "release_date", "video_release_date",
                 "imdb_url"] + GENRE_COLUMNS


def _require_file(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download ml-100k.zip per README.md and "
            "unzip it so that this exact path exists."
        )


def load_ratings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load u.data -> DataFrame[user_id, item_id, rating, timestamp]."""
    path = data_dir / "u.data"
    _require_file(path)
    return pd.read_csv(
        path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )


def load_items(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load u.item -> DataFrame with movie_id, movie_title, and 19 genre
    one-hot columns. Uses latin-1 encoding, matching the original files
    (they predate UTF-8 becoming the de facto standard)."""
    path = data_dir / "u.item"
    _require_file(path)
    return pd.read_csv(
        path, sep="|", names=ITEM_COLUMNS, encoding="latin-1", engine="python",
    )


def build_sparse_ratings_matrix(ratings: pd.DataFrame):
    """Build a sparse (n_users x n_items) ratings matrix, 0 where unrated.

    Returns (matrix, user_id_to_row, item_id_to_col, row_to_user_id,
    col_to_item_id) so callers can map between MovieLens IDs (1-indexed,
    with gaps possible) and matrix row/col indices.
    """
    user_ids = sorted(ratings["user_id"].unique())
    item_ids = sorted(ratings["item_id"].unique())

    user_id_to_row = {uid: i for i, uid in enumerate(user_ids)}
    item_id_to_col = {iid: i for i, iid in enumerate(item_ids)}

    rows = ratings["user_id"].map(user_id_to_row).values
    cols = ratings["item_id"].map(item_id_to_col).values
    vals = ratings["rating"].values.astype(np.float32)

    matrix = csr_matrix((vals, (rows, cols)), shape=(len(user_ids), len(item_ids)))

    row_to_user_id = {v: k for k, v in user_id_to_row.items()}
    col_to_item_id = {v: k for k, v in item_id_to_col.items()}

    return matrix, user_id_to_row, item_id_to_col, row_to_user_id, col_to_item_id
