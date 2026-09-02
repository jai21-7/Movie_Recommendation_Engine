"""Pydantic response models for the FastAPI service."""

from pydantic import BaseModel, Field


class RecommendationOut(BaseModel):
    movie_id: int
    title: str
    score: float
    genres: str


class RecommendationsResponse(BaseModel):
    user_id: int
    algorithm: str
    top_k: int
    recommendations: list[RecommendationOut]


class SimilarMoviesResponse(BaseModel):
    movie_id: int
    title: str
    top_k: int
    similar: list[RecommendationOut]


class PredictionResponse(BaseModel):
    user_id: int
    movie_id: int
    title: str
    predicted_rating: float


class MovieOut(BaseModel):
    movie_id: int
    title: str
    genres: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    num_users: int
    num_movies: int


class ErrorResponse(BaseModel):
    detail: str
