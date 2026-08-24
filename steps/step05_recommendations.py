"""
STEP 5: Full recommendations — item-based, user-based, and content-based.

Run: python steps/step05_recommendations.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_data import download_movielens
from src.data_loader import load_all
from src.recommender import (
    ContentBasedRecommender,
    ItemBasedCF,
    UserBasedCF,
    format_recommendations,
)


def main():
    print("\n>>> STEP 5: Movie Recommendations\n")

    download_movielens()
    data = load_all()
    movies = data["movies"]
    ratings = data["ratings"]
    tags = data["tags"]

    # Pick a user with many ratings for a rich demo
    user_counts = ratings.groupby("userId").size()
    demo_user = user_counts.idxmax()
    user_ratings = ratings[ratings["userId"] == demo_user].merge(movies, on="movieId")
    top_rated = user_ratings.nlargest(5, "rating")

    print(f"Demo user: {demo_user} ({user_counts[demo_user]} ratings)")
    print("Their top-rated movies:")
    for _, row in top_rated.iterrows():
        print(f"  ★ {row['rating']} — {row['title']}")
    print()

    # --- Item-Based CF ---
    print("─" * 50)
    print("ITEM-BASED COLLABORATIVE FILTERING")
    print("Find movies similar to ones you liked (co-rating patterns)")
    print("─" * 50)
    item_cf = ItemBasedCF().fit(movies, ratings)
    item_recs = item_cf.recommend_for_user(demo_user, top_k=8)
    print(format_recommendations(item_recs))
    print()

    # --- User-Based CF ---
    print("─" * 50)
    print("USER-BASED COLLABORATIVE FILTERING")
    print("Find users with similar taste, recommend their favorites")
    print("─" * 50)
    user_cf = UserBasedCF().fit(movies, ratings)
    user_recs = user_cf.recommend_for_user(demo_user, top_k=8)
    print(format_recommendations(user_recs))
    print()

    # --- Content-Based ---
    print("─" * 50)
    print("CONTENT-BASED FILTERING")
    print("Recommend movies with similar genres/tags")
    print("─" * 50)
    content = ContentBasedRecommender(use_tags=True).fit(movies, tags)
    content_recs = content.recommend_for_user(ratings, demo_user, top_k=8)
    print(format_recommendations(content_recs))
    print()

    # --- Similar movies demo ---
    favorite = top_rated.iloc[0]
    print("─" * 50)
    print(f"SIMILAR TO: {favorite['title']}")
    print("─" * 50)
    similar = content.similar_movies(int(favorite["movieId"]), top_k=5)
    print(format_recommendations(similar))
    print()

    print("You built a complete recommendation pipeline!")
    print("Next: run python main.py for an interactive demo.\n")


if __name__ == "__main__":
    main()
