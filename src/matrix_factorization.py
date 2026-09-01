"""
Matrix Factorization (SVD) recommender.

Idea: decompose the user-item rating matrix R ≈ U @ Sigma @ V^T
- U: user latent factors (taste dimensions)
- V: movie latent factors (attribute dimensions)
- Sigma: importance of each latent dimension

Better than memory-based CF for sparse data — learns compact representations.
"""

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from src.features import build_user_item_matrix, filter_active_users_and_movies
from src.recommender import Recommendation


class SVDRecommender:
    """Truncated SVD collaborative filtering recommender."""

    def __init__(
        self,
        n_factors: int = 50,
        min_user_ratings: int = 20,
        min_movie_ratings: int = 20,
    ):
        self.n_factors = n_factors
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings
        self.movies: pd.DataFrame = None
        self.user_item_df: pd.DataFrame = None
        self.user_ids: np.ndarray = None
        self.movie_ids: np.ndarray = None
        self.user_factors: np.ndarray = None
        self.movie_factors: np.ndarray = None
        self.global_mean: float = 0.0

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> "SVDRecommender":
        self.movies = movies.set_index("movieId")
        filtered = filter_active_users_and_movies(
            ratings, self.min_user_ratings, self.min_movie_ratings
        )
        self.user_item_df, matrix = build_user_item_matrix(filtered)
        self.user_ids = self.user_item_df.index.values
        self.movie_ids = self.user_item_df.columns.values

        # SVD works best on mean-centered data
        self.global_mean = matrix[matrix > 0].mean()
        centered = matrix.copy()
        centered[matrix > 0] = matrix[matrix > 0] - self.global_mean

        k = min(self.n_factors, min(centered.shape) - 1)
        u, sigma, vt = svds(centered.astype(np.float64), k=k)

        # Sort by singular value (svds returns ascending order)
        order = np.argsort(sigma)[::-1]
        self.user_factors = u[:, order]
        sigma = sigma[order]
        self.movie_factors = (np.diag(sigma) @ vt[order]).T

        return self

    def predict(self, user_id: int, movie_id: int) -> float | None:
        """Predict rating for a user-movie pair."""
        if user_id not in self.user_ids or movie_id not in self.movie_ids:
            return None
        u_idx = np.where(self.user_ids == user_id)[0][0]
        m_idx = np.where(self.movie_ids == movie_id)[0][0]
        return float(
            self.global_mean
            + self.user_factors[u_idx] @ self.movie_factors[m_idx]
        )

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> list[Recommendation]:
        """Recommend top-K movies with highest predicted ratings."""
        if user_id not in self.user_ids:
            return []

        u_idx = np.where(self.user_ids == user_id)[0][0]
        seen = set(
            self.user_item_df.loc[user_id][self.user_item_df.loc[user_id] > 0].index
        )

        scores = self.global_mean + self.user_factors[u_idx] @ self.movie_factors.T
        recommendations = []

        for m_idx, score in enumerate(scores):
            movie_id = self.movie_ids[m_idx]
            if movie_id in seen:
                continue
            row = self.movies.loc[movie_id]
            recommendations.append(
                Recommendation(
                    movie_id=int(movie_id),
                    title=row["title"],
                    score=float(score),
                    genres=row["genres"],
                )
            )

        recommendations.sort(key=lambda r: r.score, reverse=True)
        return recommendations[:top_k]
