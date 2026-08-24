"""
Step 2: Explore the dataset — understand shape, sparsity, and distributions.

Before building a recommender, always ask:
- How many users and movies?
- How sparse is the rating matrix?
- What does a typical rating distribution look like?
"""

import pandas as pd


def explore_dataset(movies: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame) -> dict:
    """
    Compute summary statistics useful for understanding the data.

    Returns a dict you can print or inspect in a notebook.
    """
    n_users = ratings["userId"].nunique()
    n_movies = ratings["movieId"].nunique()
    n_ratings = len(ratings)
    max_possible = n_users * n_movies
    sparsity = 1 - (n_ratings / max_possible)

    ratings_per_user = ratings.groupby("userId").size()
    ratings_per_movie = ratings.groupby("movieId").size()

    summary = {
        "num_users": n_users,
        "num_movies": n_movies,
        "num_ratings": n_ratings,
        "sparsity_pct": round(sparsity * 100, 2),
        "rating_min": ratings["rating"].min(),
        "rating_max": ratings["rating"].max(),
        "rating_mean": round(ratings["rating"].mean(), 2),
        "ratings_per_user_mean": round(ratings_per_user.mean(), 1),
        "ratings_per_movie_mean": round(ratings_per_movie.mean(), 1),
        "num_tags": len(tags),
        "sample_movies": movies.head(5)[["movieId", "title", "genres"]].to_dict("records"),
    }
    return summary


def print_summary(summary: dict) -> None:
    """Pretty-print exploration summary."""
    print("=" * 50)
    print("MOVIELENS DATASET SUMMARY")
    print("=" * 50)
    print(f"Users:           {summary['num_users']:,}")
    print(f"Movies:          {summary['num_movies']:,}")
    print(f"Ratings:         {summary['num_ratings']:,}")
    print(f"Matrix sparsity: {summary['sparsity_pct']}% empty")
    print(f"Rating range:    {summary['rating_min']} – {summary['rating_max']}")
    print(f"Mean rating:     {summary['rating_mean']}")
    print(f"Ratings/user:    {summary['ratings_per_user_mean']} (avg)")
    print(f"Ratings/movie:   {summary['ratings_per_movie_mean']} (avg)")
    print(f"Tags:            {summary['num_tags']:,}")
    print()
    print("Sample movies:")
    for m in summary["sample_movies"]:
        print(f"  [{m['movieId']}] {m['title']} — {m['genres']}")
    print("=" * 50)

