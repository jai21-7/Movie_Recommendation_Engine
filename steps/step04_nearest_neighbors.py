"""
STEP 4: Nearest neighbor retrieval with sklearn and manual top-K.

Run: python steps/step04_nearest_neighbors.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_data import download_movielens
from src.data_loader import load_all
from src.features import (
    build_movie_genre_features,
    filter_active_users_and_movies,
    normalize_rows,
)
from src.neighbors import fit_nearest_neighbors, find_neighbors, top_k_from_similarity
from src.similarity import cosine_similarity_matrix


def main():
    print("\n>>> STEP 4: Nearest Neighbor Retrieval\n")

    download_movielens()
    data = load_all()
    movies = data["movies"]

    genre_df, genre_np = build_movie_genre_features(movies)
    normalized = normalize_rows(genre_np)
    sim_matrix = cosine_similarity_matrix(normalized)

    # Method A: sklearn NearestNeighbors
    nn = fit_nearest_neighbors(normalized, n_neighbors=6)
    query_idx = 0
    query_movie = movies[movies["movieId"] == genre_df.index[query_idx]].iloc[0]

    distances, indices = find_neighbors(nn, normalized[query_idx], top_k=6)

    print(f"Query movie: {query_movie['title']}")
    print("\nMethod A — sklearn NearestNeighbors (cosine distance):")
    for dist, idx in zip(distances, indices):
        m = movies[movies["movieId"] == genre_df.index[idx]].iloc[0]
        sim = 1 - dist  # cosine distance → similarity
        print(f"  sim={sim:.3f} — {m['title']}")

    # Method B: precomputed similarity matrix + argsort
    top_idx, top_scores = top_k_from_similarity(sim_matrix, query_idx, top_k=5)

    print("\nMethod B — top-K from similarity matrix:")
    for idx, score in zip(top_idx, top_scores):
        m = movies[movies["movieId"] == genre_df.index[idx]].iloc[0]
        print(f"  sim={score:.3f} — {m['title']}")
    print()

    print("Key takeaway:")
    print("  • NearestNeighbors finds K closest items in feature space.")
    print("  • Precomputing similarity is O(n²) — fine for thousands, not millions.")
    print("  • Production systems use ANN libraries (FAISS, Annoy) for scale.\n")


if __name__ == "__main__":
    main()
