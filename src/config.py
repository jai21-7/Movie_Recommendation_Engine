"""Shared paths and constants for the MovieLens pipeline."""

from pathlib import Path

# Project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Raw data downloaded from GroupLens
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"

# MovieLens "latest small" dataset (extracted zip root)
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
MOVIELENS_DIR = DATA_DIR / "ml-latest-small"

MOVIES_FILE = MOVIELENS_DIR / "movies.csv"
RATINGS_FILE = MOVIELENS_DIR / "ratings.csv"
TAGS_FILE = MOVIELENS_DIR / "tags.csv"
LINKS_FILE = MOVIELENS_DIR / "links.csv"

# Minimum ratings per user/movie to keep sparse matrix manageable for learning
MIN_USER_RATINGS = 20
MIN_MOVIE_RATINGS = 20
