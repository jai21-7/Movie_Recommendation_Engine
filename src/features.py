"""
Step 3: Turn raw data into numerical features.

Two main approaches we build here:

A) COLLABORATIVE FILTERING features
   - User-item rating matrix (users × movies)
   - Each cell = rating (or 0 if unseen)
   - Users/movies become vectors in "rating space"

B) CONTENT-BASED features
   - One-hot encode movie genres → genre vector per movie
   - TF-IDF on tags → text-like feature vector per movie
   - Similar movies share similar genre/tag profiles
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def filter_active_users_and_movies(
    ratings: pd.DataFrame,
    min_user_ratings: int = 20,
    min_movie_ratings: int = 20,
) -> pd.DataFrame:
    """
    Remove users/movies with too few ratings.

    Why? Very sparse rows/columns make similarity noisy.
    For learning, a denser subset is easier to interpret.
    """
    user_counts = ratings.groupby("userId")["movieId"].transform("count")
    movie_counts = ratings.groupby("movieId")["userId"].transform("count")

    filtered = ratings[(user_counts >= min_user_ratings) & (movie_counts >= min_movie_ratings)]
    return filtered.reset_index(drop=True)


def build_user_item_matrix(ratings: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
  Build the classic user-item rating matrix.

  Shape: (num_users, num_movies)
  Values: rating if seen, 0.0 if not rated

  Returns:
      matrix_df: DataFrame with userId index and movieId columns
      matrix_np: same data as numpy array for fast math
  """
    matrix_df = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating",
        fill_value=0.0,
    )
    return matrix_df, matrix_df.values


def build_movie_genre_features(movies: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """
    One-hot encode genres from strings like "Action|Adventure|Sci-Fi".

    Each movie → binary vector over all genres in the dataset.
    """
    # Split pipe-separated genres into a list per movie
    genre_lists = movies["genres"].str.split("|")

    # Collect every unique genre
    all_genres = sorted({g for genres in genre_lists for g in genres})

    # Build one-hot rows
    rows = []
    for genres in genre_lists:
        row = {g: (1 if g in genres else 0) for g in all_genres}
        rows.append(row)

    feature_df = pd.DataFrame(rows, index=movies["movieId"].values)
    feature_df.index.name = "movieId"
    return feature_df, feature_df.values


def build_movie_tag_features(
    tags: pd.DataFrame,
    movies: pd.DataFrame,
    max_features: int = 500,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    TF-IDF on aggregated tags per movie.

    Tags are short text labels users attach to movies.
    TF-IDF weights frequent tags lower and rare distinctive tags higher.
    """
    # Combine all tags for each movie into one "document"
    tag_text = (
        tags.groupby("movieId")["tag"]
        .apply(lambda x: " ".join(x.astype(str)))
        .reset_index()
    )

    # Include movies with no tags (empty string)
    movie_ids = movies["movieId"].unique()
    tag_text = tag_text.set_index("movieId").reindex(movie_ids, fill_value="").reset_index()

    vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(tag_text["tag"])

    feature_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        index=tag_text["movieId"].values,
    )
    feature_df.index.name = "movieId"
    return feature_df, vectorizer_matrix_to_numpy(tfidf_matrix)


def vectorizer_matrix_to_numpy(matrix) -> np.ndarray:
    return matrix.toarray()


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """
    L2-normalize each row to unit length.

    After normalization, dot product between rows = cosine similarity.
    This is a key trick that makes cosine similarity fast.
    """
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # avoid division by zero
    return matrix / norms


def user_mean_center(matrix_df: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract each user's mean rating from their ratings.

    Why? Some users rate everything high (4-5), others are harsh (2-3).
    Centering removes this bias so similarity reflects taste, not scale.
    """
    # Only center non-zero (rated) entries
    result = matrix_df.copy()
    for user_id in result.index:
        row = result.loc[user_id]
        rated = row[row > 0]
        if len(rated) > 0:
            mean = rated.mean()
            mask = row > 0
            result.loc[user_id, mask] = row[mask] - mean
    return result

