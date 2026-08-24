"""
Interactive movie recommendation demo.

Run: python main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.download_data import download_movielens
from src.data_loader import load_all
from src.recommender import (
    ContentBasedRecommender,
    ItemBasedCF,
    UserBasedCF,
    format_recommendations,
)


def main():
    print("=" * 60)
    print("  MOVIE RECOMMENDATION ENGINE")
    print("  MovieLens dataset | CF + Content-Based")
    print("=" * 60)

    download_movielens()
    data = load_all()
    movies = data["movies"]
    ratings = data["ratings"]
    tags = data["tags"]

    print("\nTraining recommenders (this takes a few seconds)...")
    item_cf = ItemBasedCF().fit(movies, ratings)
    user_cf = UserBasedCF().fit(movies, ratings)
    content = ContentBasedRecommender(use_tags=True).fit(movies, tags)
    print("Ready!\n")

    while True:
        print("─" * 40)
        print("Commands:")
        print("  user <id>     — recommend for user ID")
        print("  similar <id>  — movies similar to movie ID")
        print("  quit          — exit")
        print("─" * 40)

        try:
            cmd = input("> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if cmd in ("quit", "q", "exit"):
            print("Goodbye!")
            break

        parts = cmd.split()
        if len(parts) == 2 and parts[0] == "user":
            try:
                user_id = int(parts[1])
            except ValueError:
                print("Invalid user ID.")
                continue

            print(f"\nRecommendations for user {user_id}:\n")
            print("[Item-Based CF]")
            print(format_recommendations(item_cf.recommend_for_user(user_id, top_k=5)))
            print("\n[User-Based CF]")
            print(format_recommendations(user_cf.recommend_for_user(user_id, top_k=5)))
            print("\n[Content-Based]")
            print(format_recommendations(content.recommend_for_user(ratings, user_id, top_k=5)))

        elif len(parts) == 2 and parts[0] == "similar":
            try:
                movie_id = int(parts[1])
            except ValueError:
                print("Invalid movie ID.")
                continue

            title = movies[movies["movieId"] == movie_id]["title"].values
            if len(title) == 0:
                print(f"Movie {movie_id} not found.")
                continue

            print(f"\nMovies similar to: {title[0]}\n")
            print(format_recommendations(content.similar_movies(movie_id, top_k=8)))

        else:
            print("Unknown command. Try: user 1  or  similar 1")


if __name__ == "__main__":
    main()
