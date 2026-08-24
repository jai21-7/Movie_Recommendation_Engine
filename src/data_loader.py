"""
Step 1: Load real-world MovieLens data into pandas DataFrames.

Key concepts:
- ratings: implicit feedback about user preferences (1-5 stars)
- movies: item metadata (title, genres)
- tags: user-generated labels (useful for content-based features)
"""

from pathlib import Path

import pandas as pd

from src.config import LINKS_FILE, MOVIES_FILE, RATINGS_FILE, TAGS_FILE


def load_movies() -> pd.DataFrame:
    """Load movie metadata: movieId, title, genres."""
    return pd.read_csv(MOVIES_FILE)


def load_ratings() -> pd.DataFrame:
    """Load user ratings: userId, movieId, rating, timestamp."""
    return pd.read_csv(RATINGS_FILE)


def load_tags() -> pd.DataFrame:
    """Load user-applied tags: userId, movieId, tag, timestamp."""
    return pd.read_csv(TAGS_FILE)


def load_links() -> pd.DataFrame:
    """Load external IDs (IMDB, TMDB) for each movie."""
    return pd.read_csv(LINKS_FILE)


def load_all() -> dict[str, pd.DataFrame]:
    """Return all tables as a dictionary."""
    return {
        "movies": load_movies(),
        "ratings": load_ratings(),
        "tags": load_tags(),
        "links": load_links(),
    }


def ensure_data_exists() -> None:
    """Raise if raw data is missing (run scripts/download_data.py first)."""
    if not MOVIES_FILE.exists():
        raise FileNotFoundError(
            f"MovieLens data not found at {MOVIES_FILE}. "
            "Run: python scripts/download_data.py"
        )

