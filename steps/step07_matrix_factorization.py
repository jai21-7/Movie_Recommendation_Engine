"""
STEP 7: Matrix Factorization with SVD.

Run: python steps/step07_matrix_factorization.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_data import download_movielens
from src.data_loader import load_all
from src.matrix_factorization import SVDRecommender
from src.recommender import format_recommendations


def main():
    print("\n>>> STEP 7: Matrix Factorization (SVD)\n")

    download_movielens()
    data = load_all()
    movies, ratings = data["movies"], data["ratings"]

    user_counts = ratings.groupby("userId").size()
    demo_user = user_counts.idxmax()

    print("Training SVD with 50 latent factors...")
    svd = SVDRecommender(n_factors=50).fit(movies, ratings)
    print(f"  Users: {len(svd.user_ids)} | Movies: {len(svd.movie_ids)}")
    print(f"  Global mean rating: {svd.global_mean:.2f}\n")

    print(f"Recommendations for user {demo_user} ({user_counts[demo_user]} ratings):")
    recs = svd.recommend_for_user(demo_user, top_k=8)
    print(format_recommendations(recs))
    print()

    # Show a prediction example
    if recs:
        sample = recs[0]
        pred = svd.predict(demo_user, sample.movie_id)
        print(f"Predicted rating for '{sample.title}': {pred:.2f}")
    print()

    print("Key takeaway:")
    print("  • SVD learns hidden taste dimensions (e.g. 'likes action', 'prefers classics').")
    print("  • R ≈ U @ V^T compresses sparse ratings into dense latent vectors.")
    print("  • Used by Netflix Prize winners and modern hybrid systems.\n")


if __name__ == "__main__":
    main()
