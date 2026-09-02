"""Shared model loading and recommendation logic for the API."""

from scripts.download_data import download_movielens
from src.data_loader import load_all
from src.matrix_factorization import SVDRecommender
from src.recommender import (
    ContentBasedRecommender,
    ItemBasedCF,
    Recommendation,
    UserBasedCF,
)

ALGORITHMS = ("svd", "item_cf", "user_cf", "content")


class RecommenderService:
    """Loads all models once and serves recommendations."""

    def __init__(self):
        self._loaded = False
        self.movies = None
        self.ratings = None
        self.models: dict = {}
        self.user_ids: list[int] = []
        self.movie_titles: dict[int, str] = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        if self._loaded:
            return

        download_movielens()
        data = load_all()
        movies, ratings, tags = data["movies"], data["ratings"], data["tags"]

        self.movies = movies
        self.ratings = ratings
        self.user_ids = sorted(ratings["userId"].unique().tolist())
        self.movie_titles = movies.set_index("movieId")["title"].to_dict()

        self.models = {
            "svd": SVDRecommender(n_factors=50).fit(movies, ratings),
            "item_cf": ItemBasedCF().fit(movies, ratings),
            "user_cf": UserBasedCF().fit(movies, ratings),
            "content": ContentBasedRecommender(use_tags=True).fit(movies, tags),
        }
        self._loaded = True

    def _to_dict(self, recs: list[Recommendation]) -> list[dict]:
        return [
            {
                "movie_id": r.movie_id,
                "title": r.title,
                "score": round(r.score, 4),
                "genres": r.genres,
            }
            for r in recs
        ]

    def recommend(self, user_id: int, algorithm: str = "svd", top_k: int = 10) -> list[dict]:
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found")

        algo = algorithm.lower()
        if algo not in self.models:
            raise ValueError(f"Unknown algorithm '{algorithm}'. Choose: {', '.join(ALGORITHMS)}")

        model = self.models[algo]
        if algo == "content":
            recs = model.recommend_for_user(self.ratings, user_id, top_k=top_k)
        else:
            recs = model.recommend_for_user(user_id, top_k=top_k)

        return self._to_dict(recs)

    def similar_movies(self, movie_id: int, top_k: int = 10) -> list[dict]:
        if movie_id not in self.movie_titles:
            raise ValueError(f"Movie {movie_id} not found")

        recs = self.models["content"].similar_movies(movie_id, top_k=top_k)
        return self._to_dict(recs)

    def predict_rating(self, user_id: int, movie_id: int) -> float:
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not found")
        if movie_id not in self.movie_titles:
            raise ValueError(f"Movie {movie_id} not found")

        pred = self.models["svd"].predict(user_id, movie_id)
        if pred is None:
            raise ValueError("SVD model cannot predict for this user/movie pair")
        return round(pred, 4)

    def search_movies(self, query: str = "", limit: int = 20) -> list[dict]:
        df = self.movies
        if query:
            mask = df["title"].str.contains(query, case=False, na=False)
            df = df[mask]
        rows = df.head(limit)
        return [
            {"movie_id": int(r.movieId), "title": r.title, "genres": r.genres}
            for r in rows.itertuples()
        ]


# Singleton used by FastAPI lifespan
service = RecommenderService()
