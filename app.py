"""
Streamlit web UI for movie recommendations.

Run: streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from scripts.download_data import download_movielens
from src.data_loader import load_all
from src.matrix_factorization import SVDRecommender
from src.recommender import (
    ContentBasedRecommender,
    ItemBasedCF,
    UserBasedCF,
    format_recommendations,
)


@st.cache_resource
def load_models():
    download_movielens()
    data = load_all()
    movies, ratings, tags = data["movies"], data["ratings"], data["tags"]

    with st.spinner("Training recommenders (first load takes ~30s)..."):
        models = {
            "SVD (Matrix Factorization)": SVDRecommender(n_factors=50).fit(movies, ratings),
            "Item-Based CF": ItemBasedCF().fit(movies, ratings),
            "User-Based CF": UserBasedCF().fit(movies, ratings),
            "Content-Based": ContentBasedRecommender(use_tags=True).fit(movies, tags),
        }
    return models, movies, ratings


def main():
    st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")
    st.title("Movie Recommendation Engine")
    st.caption("MovieLens dataset · SVD · Collaborative Filtering · Content-Based")

    models, movies, ratings = load_models()
    user_ids = sorted(ratings["userId"].unique())

    tab_recs, tab_similar = st.tabs(["Recommend for User", "Similar Movies"])

    with tab_recs:
        col1, col2, col3 = st.columns(3)
        with col1:
            user_id = st.selectbox("User ID", user_ids, index=len(user_ids) // 2)
        with col2:
            model_name = st.selectbox("Algorithm", list(models.keys()))
        with col3:
            top_k = st.slider("Number of recommendations", 5, 20, 10)

        user_history = ratings[ratings["userId"] == user_id].merge(movies, on="movieId")
        st.subheader(f"User {user_id} — recent ratings")
        st.dataframe(
            user_history.nlargest(8, "rating")[["title", "rating", "genres"]],
            hide_index=True,
            use_container_width=True,
        )

        model = models[model_name]
        if model_name == "Content-Based":
            recs = model.recommend_for_user(ratings, user_id, top_k=top_k)
        else:
            recs = model.recommend_for_user(user_id, top_k=top_k)

        st.subheader(f"Top {top_k} recommendations ({model_name})")
        if recs:
            st.dataframe(
                [{"#": i, "Title": r.title, "Score": round(r.score, 3), "Genres": r.genres}
                 for i, r in enumerate(recs, 1)],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.warning("No recommendations found for this user.")

    with tab_similar:
        movie_titles = movies.set_index("movieId")["title"].to_dict()
        movie_id = st.selectbox(
            "Select a movie",
            options=list(movie_titles.keys()),
            format_func=lambda mid: f"[{mid}] {movie_titles[mid]}",
        )
        top_k_sim = st.slider("Similar movies", 5, 15, 8, key="sim_k")

        content_model = models["Content-Based"]
        similar = content_model.similar_movies(movie_id, top_k=top_k_sim)

        st.subheader(f"Movies similar to: {movie_titles[movie_id]}")
        if similar:
            st.dataframe(
                [{"#": i, "Title": r.title, "Similarity": round(r.score, 3), "Genres": r.genres}
                 for i, r in enumerate(similar, 1)],
                hide_index=True,
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
