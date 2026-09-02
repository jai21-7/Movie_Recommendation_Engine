"""
FastAPI REST API for movie recommendations.

Run:
    uvicorn api:app --reload

Docs:
    http://localhost:8000/docs
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.api.schemas import (
    HealthResponse,
    MovieOut,
    PredictionResponse,
    RecommendationsResponse,
    SimilarMoviesResponse,
)
from src.api.service import ALGORITHMS, service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train/load all models once on startup."""
    service.load()
    yield


app = FastAPI(
    title="Movie Recommendation API",
    description=(
        "REST API for movie recommendations using MovieLens data. "
        "Algorithms: SVD, item-based CF, user-based CF, content-based."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["meta"])
def root():
    return {
        "message": "Movie Recommendation API",
        "docs": "/docs",
        "algorithms": list(ALGORITHMS),
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    return HealthResponse(
        status="ok",
        models_loaded=service.is_loaded,
        num_users=len(service.user_ids),
        num_movies=len(service.movie_titles),
    )


@app.get("/users", tags=["users"])
def list_users(limit: int = Query(50, ge=1, le=500)):
    """Return sample user IDs available in the dataset."""
    return {"users": service.user_ids[:limit], "total": len(service.user_ids)}


@app.get("/movies", response_model=list[MovieOut], tags=["movies"])
def list_movies(
    q: str = Query("", description="Search by title"),
    limit: int = Query(20, ge=1, le=100),
):
    """Search or browse movies."""
    return service.search_movies(query=q, limit=limit)


@app.get(
    "/recommend/{user_id}",
    response_model=RecommendationsResponse,
    tags=["recommendations"],
)
def recommend(
    user_id: int,
    algorithm: str = Query("svd", description=f"One of: {', '.join(ALGORITHMS)}"),
    top_k: int = Query(10, ge=1, le=50),
):
    """
    Get top-K movie recommendations for a user.

    - **svd**: matrix factorization (best general-purpose)
    - **item_cf**: item-based collaborative filtering
    - **user_cf**: user-based collaborative filtering
    - **content**: genre/tag similarity
    """
    try:
        recs = service.recommend(user_id, algorithm=algorithm, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return RecommendationsResponse(
        user_id=user_id,
        algorithm=algorithm.lower(),
        top_k=top_k,
        recommendations=recs,
    )


@app.get(
    "/similar/{movie_id}",
    response_model=SimilarMoviesResponse,
    tags=["recommendations"],
)
def similar_movies(
    movie_id: int,
    top_k: int = Query(10, ge=1, le=50),
):
    """Find movies with similar genres/tags to a given movie."""
    title = service.movie_titles.get(movie_id)
    if not title:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

    try:
        similar = service.similar_movies(movie_id, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return SimilarMoviesResponse(
        movie_id=movie_id,
        title=title,
        top_k=top_k,
        similar=similar,
    )


@app.get(
    "/predict",
    response_model=PredictionResponse,
    tags=["predictions"],
)
def predict_rating(
    user_id: int = Query(..., description="User ID"),
    movie_id: int = Query(..., description="Movie ID"),
):
    """Predict star rating (1-5) for a user-movie pair using SVD."""
    title = service.movie_titles.get(movie_id)
    if not title:
        raise HTTPException(status_code=404, detail=f"Movie {movie_id} not found")

    try:
        rating = service.predict_rating(user_id, movie_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return PredictionResponse(
        user_id=user_id,
        movie_id=movie_id,
        title=title,
        predicted_rating=rating,
    )
