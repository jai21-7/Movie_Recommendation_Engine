"""
STEP 3: Cosine similarity — the math behind 'similar taste'.

Run: python steps/step03_cosine_similarity.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_data import download_movielens
from src.data_loader import load_all
from src.features import build_movie_genre_features, filter_active_users_and_movies
from src.similarity import cosine_similarity_matrix, explain_similarity


def main():
    print("\n>>> STEP 3: Cosine Similarity\n")

    download_movielens()
    data = load_all()
    ratings = filter_active_users_and_movies(data["ratings"])
    movies = data["movies"]

    genre_df, genre_np = build_movie_genre_features(movies)
    sim_matrix = cosine_similarity_matrix(genre_np)

    # Pick two well-known movies for intuition
    toy_story = movies[movies["title"].str.contains("Toy Story", case=False, na=False)].iloc[0]
    jumanji = movies[movies["title"].str.contains("Jumanji", case=False, na=False)].iloc[0]

    idx_toy = genre_df.index.get_loc(toy_story["movieId"])
    idx_jum = genre_df.index.get_loc(jumanji["movieId"])

    breakdown = explain_similarity(
        toy_story["title"],
        jumanji["title"],
        genre_np[idx_toy],
        genre_np[idx_jum],
    )

    print("Manual cosine similarity breakdown:")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")
    print()

    # Find movies most similar to Toy Story (by genre)
    toy_sim = sim_matrix[idx_toy]
    top_indices = toy_sim.argsort()[::-1][1:6]  # skip self

    print(f"Movies most similar to '{toy_story['title']}' (genre cosine similarity):")
    for i in top_indices:
        m = movies[movies["movieId"] == genre_df.index[i]].iloc[0]
        print(f"  {toy_sim[i]:.3f} — {m['title']} ({m['genres']})")
    print()

    print("Key takeaway:")
    print("  • Cosine similarity measures angle between vectors, not magnitude.")
    print("  • sim=1 → identical direction, sim=0 → orthogonal (no shared signal).")
    print("  • For normalized vectors, dot product = cosine similarity (fast trick).\n")


if __name__ == "__main__":
    main()
