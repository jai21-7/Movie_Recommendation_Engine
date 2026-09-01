"""
Evaluation metrics for recommendation systems.

Metrics:
- RMSE: how far off are predicted ratings? (lower is better)
- Precision@K: of top-K recommendations, how many are relevant? (higher is better)
"""

import numpy as np
import pandas as pd


def train_test_split_by_user(
    ratings: pd.DataFrame,
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Hold out a fraction of each user's ratings for testing.

    Why per-user split? Ensures every user appears in both train and test.
    """
    rng = np.random.default_rng(random_state)
    train_parts, test_parts = [], []

    for user_id, group in ratings.groupby("userId"):
        n_test = max(1, int(len(group) * test_ratio))
        test_idx = rng.choice(group.index, size=n_test, replace=False)
        test_parts.append(ratings.loc[test_idx])
        train_parts.append(group.drop(test_idx))

    return pd.concat(train_parts), pd.concat(test_parts)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error between actual and predicted ratings."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def precision_at_k(
    recommended_ids: list[int],
    relevant_ids: set[int],
    k: int = 10,
) -> float:
    """
    Precision@K = |recommended[:K] ∩ relevant| / K

    relevant = movies the user actually liked (e.g. rating >= 4 in test set)
    """
    if k == 0:
        return 0.0
    top_k = recommended_ids[:k]
    hits = sum(1 for mid in top_k if mid in relevant_ids)
    return hits / k


def evaluate_rating_predictions(
    test_ratings: pd.DataFrame,
    predict_fn,
) -> dict:
    """
    Compute RMSE on a test set using a predict_fn(user_id, movie_id) -> float.
    """
    preds, actuals = [], []
    for _, row in test_ratings.iterrows():
        pred = predict_fn(int(row["userId"]), int(row["movieId"]))
        if pred is not None:
            preds.append(pred)
            actuals.append(row["rating"])

    if not preds:
        return {"rmse": None, "n_predictions": 0}

    return {
        "rmse": round(rmse(np.array(actuals), np.array(preds)), 4),
        "n_predictions": len(preds),
    }


def evaluate_recommendations(
    test_ratings: pd.DataFrame,
    recommend_fn,
    k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int = 50,
) -> dict:
    """
    Compute average Precision@K across test users.

    recommend_fn(user_id, top_k) -> list of movie_ids
    """
    test_users = test_ratings["userId"].unique()[:max_users]
    scores = []

    for user_id in test_users:
        user_test = test_ratings[test_ratings["userId"] == user_id]
        relevant = set(user_test[user_test["rating"] >= relevance_threshold]["movieId"])
        if not relevant:
            continue

        recommended = recommend_fn(int(user_id), k)
        scores.append(precision_at_k(recommended, relevant, k))

    if not scores:
        return {"precision_at_k": None, "k": k, "n_users": 0}

    return {
        "precision_at_k": round(float(np.mean(scores)), 4),
        "k": k,
        "n_users": len(scores),
    }
