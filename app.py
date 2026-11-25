import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_csv("movies.csv")

movies["genres"] = movies["genres"].fillna("")

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(
    movies["genres"]
)

similarity = cosine_similarity(
    tfidf_matrix,
    tfidf_matrix
)

def recommend(movie_name):

    idx = movies[
        movies["title"] == movie_name
    ].index[0]

    scores = list(
        enumerate(similarity[idx])
    )

    scores = sorted(
        scores,
        key=lambda x: x[1],
        reverse=True
    )

    scores = scores[1:6]

    movie_indices = [
        i[0] for i in scores
    ]

    return movies["title"].iloc[
        movie_indices
    ]

st.title("🎬 AI Movie Recommender")

selected_movie = st.selectbox(
    "Select Movie",
    movies["title"]
)

if st.button(
    "Recommend"
):

    recommendations = recommend(
        selected_movie
    )

    st.subheader(
        "Recommended Movies"
    )

    for movie in recommendations:
        st.write(movie)