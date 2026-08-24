"""
Download the MovieLens latest-small dataset.

This is a real-world benchmark dataset used in recommendation research.
~100k ratings from ~600 users on ~9k movies.
"""

import zipfile
from pathlib import Path

import requests

from src.config import DATA_DIR, MOVIELENS_DIR, MOVIELENS_URL, MOVIES_FILE


def download_movielens(force: bool = False) -> Path:
    """Download and extract MovieLens if not already present."""
    if MOVIES_FILE.exists() and not force:
        print(f"Dataset already exists at {MOVIELENS_DIR}")
        return MOVIELENS_DIR

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "ml-latest-small.zip"

    print(f"Downloading from {MOVIELENS_URL} ...")
    response = requests.get(MOVIELENS_URL, stream=True, timeout=120)
    response.raise_for_status()

    with open(zip_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Extracting to {DATA_DIR} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_DIR)

    print("Done.")
    return MOVIELENS_DIR


if __name__ == "__main__":
    download_movielens()
