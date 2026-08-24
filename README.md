# Movie Recommendation Engine

A **step-by-step, learn-by-building** movie recommendation system using the real-world [MovieLens](https://grouplens.org/datasets/movielens/) dataset.

## What You'll Learn

| Step | Topic | Key Idea |
|------|-------|----------|
| 1 | Data loading & exploration | Real ratings are sparse (~98% empty) |
| 2 | Feature engineering | Turn ratings/genres into numerical vectors |
| 3 | Cosine similarity | Measure "alike-ness" independent of scale |
| 4 | Nearest neighbors | Retrieve top-K most similar items/users |
| 5 | Recommendation engines | Item-CF, User-CF, and Content-Based |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run each step (recommended learning order)
python steps/step01_download_and_explore.py
python steps/step02_build_features.py
python steps/step03_cosine_similarity.py
python steps/step04_nearest_neighbors.py
python steps/step05_recommendations.py

# Interactive demo
python main.py
```

## Project Structure

```
├── data/                    # Downloaded MovieLens data (auto-created)
├── scripts/
│   └── download_data.py     # Fetch ml-latest-small from GroupLens
├── src/
│   ├── config.py            # Paths and constants
│   ├── data_loader.py       # Load movies, ratings, tags
│   ├── explore.py           # Dataset summary statistics
│   ├── features.py          # User-item matrix, genre one-hot, TF-IDF
│   ├── similarity.py        # Cosine similarity (manual + sklearn)
│   ├── neighbors.py         # KNN retrieval
│   └── recommender.py       # Item-CF, User-CF, Content-Based
├── steps/                   # Runnable learning scripts (start here!)
│   ├── step01_download_and_explore.py
│   ├── step02_build_features.py
│   ├── step03_cosine_similarity.py
│   ├── step04_nearest_neighbors.py
│   └── step05_recommendations.py
├── notebooks/
│   └── walkthrough.ipynb    # Jupyter notebook version
└── main.py                  # Interactive CLI demo
```

## The Three Recommender Types

### 1. Item-Based Collaborative Filtering
> "Users who liked **Toy Story** also liked **Toy Story 2**"

Computes cosine similarity between **movies** based on how users rated them.

### 2. User-Based Collaborative Filtering
> "Users similar to **you** loved **The Matrix**"

Finds users with similar rating patterns, recommends what they liked.

### 3. Content-Based Filtering
> "You liked **Action|Sci-Fi** movies, here are more Action|Sci-Fi"

Uses movie **genres** and **tags** as features — works even without many ratings.

## Core Math: Cosine Similarity

```
similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- Range: 0 (nothing in common) to 1 (identical direction)
- Why not Euclidean distance? A user who rates 4-5 vs one who rates 2-3 would look "far apart" in raw space, but have the same taste pattern.

See `src/similarity.py` for a manual implementation alongside sklearn.

## Dataset

**MovieLens Latest Small** (~100K ratings, 600 users, 9K movies). Downloaded automatically on first run from GroupLens.

## Next Steps (After This Project)

- Matrix factorization (SVD, ALS) for latent factors
- Neural collaborative filtering
- Hybrid models combining CF + content
- Approximate nearest neighbors (FAISS) for scale
- Evaluation metrics: RMSE, Precision@K, NDCG
