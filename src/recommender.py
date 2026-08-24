"""
Step 6: Movie recommendation engines — putting it all together.

Three recommenders (classic textbook approaches):

1. ItemBasedCF  — similar movies to ones you liked (item-item collaborative filtering)
2. UserBasedCF  — similar users liked these (user-user collaborative filtering)
3. ContentBased — similar genre/tag profile (no collaborative signal needed)
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features import (
    build_movie_genre_features,
    build_movie_tag_features,
    build_user_item_matrix,
    filter_active_users_and_movies,
    normalize_rows,
    user_mean_center,
)
from src.neighbors import fit_nearest_neighbors, find_neighbors, top_k_from_similarity
from src.similarity import cosine_similarity_matrix


@dataclass
class Recommendation:
    movie_id: int
    title: str
    score: float
    genres: str


class ItemBasedCF:
    """
    Item-based collaborative filtering.

    Idea: If you liked movie A, recommend movies similar to A.
    Similarity is computed from co-rating patterns across all users.
    """

    def __init__(self, min_user_ratings: int = 20, min_movie_ratings: int = 20):
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings
        self.movies: pd.DataFrame = None
        self.ratings_filtered: pd.DataFrame = None
        self.user_item_df: pd.DataFrame = None
        self.movie_ids: np.ndarray = None
        self.item_similarity: np.ndarray = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> "ItemBasedCF":
        self.movies = movies.set_index("movieId")
        self.ratings_filtered = filter_active_users_and_movies(
            ratings, self.min_user_ratings, self.min_movie_ratings
        )
        self.user_item_df, user_item_np = build_user_item_matrix(self.ratings_filtered)

        # Items as columns → transpose so each row is a movie vector across users
        item_matrix = self.user_item_df.T.values
        self.movie_ids = self.user_item_df.columns.values

        # Mean-center each movie's ratings (remove popularity bias)
        item_df = self.user_item_df.T
        for movie_id in item_df.index:
            row = item_df.loc[movie_id]
            rated = row[row > 0]
            if len(rated) > 0:
                item_df.loc[movie_id, row > 0] = row[row > 0] - rated.mean()

        self.item_similarity = cosine_similarity_matrix(item_df.values)
        return self

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        min_rating: float = 3.5,
    ) -> list[Recommendation]:
        """Recommend movies user hasn't seen, weighted by similarity to liked movies."""
        if user_id not in self.user_item_df.index:
            return []

        user_ratings = self.user_item_df.loc[user_id]
        liked_movie_ids = user_ratings[user_ratings >= min_rating].index.values
        seen_movie_ids = set(user_ratings[user_ratings > 0].index)

        scores = np.zeros(len(self.movie_ids))
        for movie_id in liked_movie_ids:
            if movie_id not in self.movie_ids:
                continue
            movie_idx = np.where(self.movie_ids == movie_id)[0][0]
            rating = user_ratings[movie_id]
            scores += rating * self.item_similarity[movie_idx]

        recommendations = []
        for idx, score in enumerate(scores):
            movie_id = self.movie_ids[idx]
            if movie_id in seen_movie_ids:
                continue
            if score <= 0:
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


class UserBasedCF:
    """
    User-based collaborative filtering.

    Idea: Find users with similar taste, recommend what they liked.
    """

    def __init__(self, min_user_ratings: int = 20, min_movie_ratings: int = 20):
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings
        self.movies: pd.DataFrame = None
        self.user_item_df: pd.DataFrame = None
        self.user_ids: np.ndarray = None
        self.user_similarity: np.ndarray = None

    def fit(self, movies: pd.DataFrame, ratings: pd.DataFrame) -> "UserBasedCF":
        self.movies = movies.set_index("movieId")
        ratings_filtered = filter_active_users_and_movies(
            ratings, self.min_user_ratings, self.min_movie_ratings
        )
        self.user_item_df, _ = build_user_item_matrix(ratings_filtered)
        centered = user_mean_center(self.user_item_df)
        self.user_ids = self.user_item_df.index.values
        self.user_similarity = cosine_similarity_matrix(centered.values)
        return self

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        n_neighbors: int = 20,
    ) -> list[Recommendation]:
        if user_id not in self.user_item_df.index:
            return []

        user_idx = np.where(self.user_ids == user_id)[0][0]
        neighbor_indices, neighbor_scores = top_k_from_similarity(
            self.user_similarity, user_idx, top_k=n_neighbors
        )

        user_ratings = self.user_item_df.values
        target_seen = set(self.user_item_df.loc[user_id][self.user_item_df.loc[user_id] > 0].index)

        predicted = np.zeros(self.user_item_df.shape[1])
        weight_sum = 0.0
        for idx, sim in zip(neighbor_indices, neighbor_scores):
            if sim <= 0:
                continue
            predicted += sim * user_ratings[idx]
            weight_sum += sim

        if weight_sum > 0:
            predicted /= weight_sum

        movie_ids = self.user_item_df.columns
        recommendations = []
        for i, score in enumerate(predicted):
            movie_id = movie_ids[i]
            if movie_id in target_seen:
                continue
            if score <= 0:
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


class ContentBasedRecommender:
    """
    Content-based filtering using movie genres (and optionally tags).

    Idea: Recommend movies with similar genre/tag profiles to ones you liked.
    No need for other users' data at inference time.
    """

    def __init__(self, use_tags: bool = False):
        self.use_tags = use_tags
        self.movies: pd.DataFrame = None
        self.movie_ids: np.ndarray = None
        self.feature_matrix: np.ndarray = None
        self.similarity: np.ndarray = None
        self.nn_model = None

    def fit(
        self,
        movies: pd.DataFrame,
        tags: pd.DataFrame | None = None,
    ) -> "ContentBasedRecommender":
        self.movies = movies.set_index("movieId")
        genre_df, genre_np = build_movie_genre_features(movies)

        if self.use_tags and tags is not None and len(tags) > 0:
            tag_df, tag_np = build_movie_tag_features(tags, movies)
            # Align and concatenate genre + tag features
            common_ids = genre_df.index.intersection(tag_df.index)
            genre_df = genre_df.loc[common_ids]
            tag_df = tag_df.loc[common_ids]
            combined = np.hstack([genre_df.values, tag_df.values])
            self.movie_ids = common_ids.values
        else:
            combined = genre_np
            self.movie_ids = genre_df.index.values

        self.feature_matrix = normalize_rows(combined)
        self.similarity = cosine_similarity_matrix(self.feature_matrix)
        self.nn_model = fit_nearest_neighbors(self.feature_matrix, n_neighbors=21)
        return self

    def similar_movies(self, movie_id: int, top_k: int = 10) -> list[Recommendation]:
        """Find movies most similar to a given movie (content-based)."""
        if movie_id not in self.movie_ids:
            return []

        idx = np.where(self.movie_ids == movie_id)[0][0]
        neighbor_indices, neighbor_scores = top_k_from_similarity(
            self.similarity, idx, top_k=top_k + 1
        )

        results = []
        for ni, score in zip(neighbor_indices, neighbor_scores):
            mid = self.movie_ids[ni]
            if mid == movie_id:
                continue
            row = self.movies.loc[mid]
            results.append(
                Recommendation(
                    movie_id=int(mid),
                    title=row["title"],
                    score=float(score),
                    genres=row["genres"],
                )
            )
        return results[:top_k]

    def recommend_for_user(
        self,
        ratings: pd.DataFrame,
        user_id: int,
        top_k: int = 10,
        min_rating: float = 3.5,
    ) -> list[Recommendation]:
        """Aggregate content similarity to movies the user liked."""
        user_ratings = ratings[ratings["userId"] == user_id]
        liked = user_ratings[user_ratings["rating"] >= min_rating]["movieId"].values
        seen = set(user_ratings["movieId"].values)

        if len(liked) == 0:
            return []

        scores = np.zeros(len(self.movie_ids))
        for movie_id in liked:
            if movie_id not in self.movie_ids:
                continue
            idx = np.where(self.movie_ids == movie_id)[0][0]
            scores += self.similarity[idx]

        recommendations = []
        for i, score in enumerate(scores):
            mid = self.movie_ids[i]
            if mid in seen:
                continue
            row = self.movies.loc[mid]
            recommendations.append(
                Recommendation(
                    movie_id=int(mid),
                    title=row["title"],
                    score=float(score),
                    genres=row["genres"],
                )
            )

        recommendations.sort(key=lambda r: r.score, reverse=True)
        return recommendations[:top_k]


def format_recommendations(recs: list[Recommendation]) -> str:
    lines = []
    for i, r in enumerate(recs, 1):
        lines.append(f"  {i}. [{r.movie_id}] {r.title} (score={r.score:.3f}) — {r.genres}")
    return "\n".join(lines) if lines else "  (no recommendations)"
