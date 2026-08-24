"""
Step 4: Cosine similarity — measure how alike two vectors are.

Cosine similarity = cos(θ) between two vectors
  = (A · B) / (||A|| × ||B||)

Range: [-1, 1] for general vectors, [0, 1] for non-negative features.

Why cosine (not Euclidean distance)?
- Magnitude-independent: a user who rates 1-5 vs 3-7 still compares fairly
- Works well for sparse high-dimensional data (classic in recommender systems)
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


def cosine_similarity_manual(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1D vectors by hand.

    This is the formula you should internalize:
        sim(a, b) = dot(a, b) / (norm(a) * norm(b))
    """
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def cosine_similarity_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Pairwise cosine similarity for all rows in a matrix.

    matrix shape: (n_items, n_features)
    returns shape: (n_items, n_items)

    sklearn uses the same formula but optimized with linear algebra.
    """
    return sklearn_cosine_similarity(matrix)


def explain_similarity(
    name_a: str,
    name_b: str,
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> dict:
    """
    Return a breakdown useful for learning/debugging.
    """
    manual = cosine_similarity_manual(vec_a, vec_b)
    sklearn = sklearn_cosine_similarity([vec_a], [vec_b])[0, 0]
    return {
        "item_a": name_a,
        "item_b": name_b,
        "cosine_similarity_manual": round(manual, 4),
        "cosine_similarity_sklearn": round(sklearn, 4),
        "dot_product": round(float(np.dot(vec_a, vec_b)), 4),
        "norm_a": round(float(np.linalg.norm(vec_a)), 4),
        "norm_b": round(float(np.linalg.norm(vec_b)), 4),
    }


def top_similar_indices(
    similarity_row: np.ndarray,
    top_k: int = 5,
    exclude_self: bool = True,
) -> np.ndarray:
    """
    Get indices of top-K most similar items from one row of a similarity matrix.

    Uses argsort (O(n log n)) — fine for learning-scale data.
    For production at scale, use approximate nearest neighbors (ANN).
    """
    scores = similarity_row.copy()
    if exclude_self:
        # We'll set self-index later when we know it; caller can mask index
        pass
    # Sort descending
    ranked = np.argsort(scores)[::-1]
    return ranked[:top_k]

