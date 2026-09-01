"""
STEP 6: Evaluate recommenders with RMSE and Precision@K.

Run: python steps/step06_evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_data import download_movielens
from src.data_loader import load_all
from src.evaluation import (
    evaluate_rating_predictions,
    evaluate_recommendations,
    train_test_split_by_user,
)
from src.matrix_factorization import SVDRecommender
from src.recommender import ItemBasedCF, UserBasedCF


def main():
    print("\n>>> STEP 6: Evaluation Metrics\n")

    download_movielens()
    data = load_all()
    movies, ratings = data["movies"], data["ratings"]

    train, test = train_test_split_by_user(ratings, test_ratio=0.2)
    print(f"Train: {len(train):,} ratings | Test: {len(test):,} ratings\n")

    print("Training models on train set...")
    svd = SVDRecommender(n_factors=50).fit(movies, train)
    item_cf = ItemBasedCF().fit(movies, train)
    user_cf = UserBasedCF().fit(movies, train)
    print("Done.\n")

    print("=" * 55)
    print("RMSE (lower is better) — predicts exact star ratings")
    print("=" * 55)

    for name, model in [("SVD", svd), ("Item-Based CF", item_cf)]:
        if name == "SVD":
            predict_fn = model.predict
        else:
            # Item-CF doesn't have direct predict; skip for RMSE
            continue

        result = evaluate_rating_predictions(
            test,
            predict_fn=lambda uid, mid: model.predict(uid, mid),
        )
        print(f"  {name:20s}  RMSE={result['rmse']}  (n={result['n_predictions']})")

    print()
    print("=" * 55)
    print("Precision@10 (higher is better) — relevant = rating >= 4")
    print("=" * 55)

    models = [
        ("SVD", svd),
        ("Item-Based CF", item_cf),
        ("User-Based CF", user_cf),
    ]

    for name, model in models:
        result = evaluate_recommendations(
            test,
            recommend_fn=lambda uid, k: [r.movie_id for r in model.recommend_for_user(uid, top_k=k)],
            k=10,
            max_users=30,
        )
        print(
            f"  {name:20s}  Precision@10={result['precision_at_k']}  "
            f"(users={result['n_users']})"
        )

    print()
    print("Key takeaway:")
    print("  • RMSE measures rating prediction accuracy.")
    print("  • Precision@K measures recommendation quality (hits in top-K).")
    print("  • SVD often beats memory-based CF on sparse data.\n")


if __name__ == "__main__":
    main()
