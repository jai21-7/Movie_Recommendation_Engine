"""
STEP 2: Turn raw data into feature matrices.

Run: python steps/step02_build_features.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_data import download_movielens
from src.data_loader import load_all
from src.features import (
    build_movie_genre_features,
    build_user_item_matrix,
    filter_active_users_and_movies,
    user_mean_center,
)


def main():
    print("\n>>> STEP 2: Feature Engineering\n")

    download_movielens()
    data = load_all()
    ratings = filter_active_users_and_movies(data["ratings"])

    # --- User-Item Matrix (collaborative filtering) ---
    user_item_df, user_item_np = build_user_item_matrix(ratings)
    print("User-Item Rating Matrix:")
    print(f"  Shape: {user_item_df.shape[0]} users × {user_item_df.shape[1]} movies")
    print(f"  Non-zero entries: {(user_item_np > 0).sum():,}")
    print(f"  Sample (first user, first 8 movies):")
    print(user_item_df.iloc[0, :8])
    print()

    # --- Mean centering ---
    centered = user_mean_center(user_item_df)
    print("After mean-centering (first user, rated movies only):")
    user_row = user_item_df.iloc[0]
    rated_mask = user_row > 0
    print(f"  Original mean: {user_row[rated_mask].mean():.2f}")
    print(f"  Centered values (sample): {centered.iloc[0, rated_mask].head(5).values}")
    print()

    # --- Genre features (content-based) ---
    movies_in_matrix = data["movies"][data["movies"]["movieId"].isin(user_item_df.columns)]
    genre_df, genre_np = build_movie_genre_features(movies_in_matrix)
    print("Movie Genre Features (one-hot):")
    print(f"  Shape: {genre_df.shape[0]} movies × {genre_df.shape[1]} genres")
    print(f"  Genre columns: {list(genre_df.columns[:8])} ...")
    sample_movie = movies_in_matrix.iloc[0]
    print(f"  Example: '{sample_movie['title']}' → genres: {sample_movie['genres']}")
    print(f"  Feature vector sum (should equal # genres): {genre_df.iloc[0].sum()}")
    print()

    print("Key takeaway:")
    print("  • CF uses rating patterns as features (no metadata needed).")
    print("  • Content-based uses genres/tags as features (works for new movies).\n")


if __name__ == "__main__":
    main()
