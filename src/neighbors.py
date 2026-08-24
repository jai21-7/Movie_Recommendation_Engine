"""
Step 5: Nearest neighbor retrieval — find the K closest items/users.

Nearest Neighbors is the workhorse of memory-based collaborative filtering:
- Item-based CF: "users who liked X also liked Y" → find movies similar to ones you rated
- User-based CF: find users with similar taste → recommend what they liked

We show both sklearn's NearestNeighbors and manual top-K via similarity matrix.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def fit_nearest_neighbors(
    feature_matrix: np.ndarray,
    metric: str = "cosine",
    n_neighbors: int = 20,
) -> NearestNeighbors:
    """
    Fit sklearn NearestNeighbors on a feature matrix.

    metric='cosine' → uses cosine distance (1 - cosine_similarity).
    Each row is one item/user vector.
    """
    # algorithm='brute' is explicit and easy to understand for learning
    nn = NearestNeighbors(metric=metric, algorithm="brute", n_neighbors=n_neighbors)
    nn.fit(feature_matrix)
    return nn


def find_neighbors(
    model: NearestNeighbors,
    query_vector: np.ndarray,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Query the fitted model for nearest neighbors of query_vector.

    Returns:
        distances: cosine distances (lower = more similar)
        indices: row indices into the original feature matrix
    """
    query = query_vector.reshape(1, -1)
    distances, indices = model.kneighbors(query, n_neighbors=top_k)
    return distances[0], indices[0]


def top_k_from_similarity(
    similarity_matrix: np.ndarray,
    query_index: int,
    top_k: int = 10,
    exclude_indices: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Manual nearest-neighbor retrieval from a precomputed similarity matrix.

    similarity_matrix[i, j] = similarity between item i and item j.

    Returns:
        indices: top-K neighbor indices
        scores: their similarity scores
    """
    scores = similarity_matrix[query_index].copy()
    if exclude_indices:
        for idx in exclude_indices:
            scores[idx] = -np.inf
    # Don't recommend self
    scores[query_index] = -np.inf

    top_indices = np.argsort(scores)[::-1][:top_k]
    return top_indices, scores[top_indices]


def aggregate_neighbor_ratings(
    neighbor_indices: np.ndarray,
    neighbor_similarities: np.ndarray,
    rating_matrix: np.ndarray,
    target_user_index: int,
) -> np.ndarray:
    """
    Weighted average of neighbors' ratings to predict scores for all movies.

    Used in user-based CF:
    - neighbor_indices: users similar to target user
    - rating_matrix: user-item matrix (users × movies)
    """
    # Similarity-weighted sum of ratings
    weights = neighbor_similarities
    weight_sum = np.sum(weights)
    if weight_sum == 0:
        return np.zeros(rating_matrix.shape[1])

    weighted_ratings = np.zeros(rating_matrix.shape[1])
    for i, user_idx in enumerate(neighbor_indices):
        weighted_ratings += weights[i] * rating_matrix[user_idx]

    return weighted_ratings / weight_sum

