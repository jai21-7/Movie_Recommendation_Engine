"""
STEP 1: Download data and explore the MovieLens dataset.

Run: python steps/step01_download_and_explore.py
"""

import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_data import download_movielens
from src.data_loader import load_all
from src.explore import explore_dataset, print_summary


def main():
    print("\n>>> STEP 1: Download & Explore MovieLens\n")

    download_movielens()
    data = load_all()

    summary = explore_dataset(data["movies"], data["ratings"], data["tags"])
    print_summary(summary)

    print("\nWhat to notice:")
    print("  • Sparsity is very high — most user-movie pairs have no rating.")
    print("  • Ratings cluster around 3-4 (people tend to rate movies they like).")
    print("  • Genres and tags give us content features for cold-start items.\n")


if __name__ == "__main__":
    main()
