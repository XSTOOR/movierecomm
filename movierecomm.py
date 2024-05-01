import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Load the movie dataset
@st.cache
def load_data():
    return pd.read_csv("movies.csv")

movies_df = load_data()

# Create a TF-IDF Vectorizer object
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_df['genres'])

# Fit a k-nearest neighbors model
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(tfidf_matrix)

# Function to get movie recommendations
def get_recommendations(movie_title, k=10):
    movie_index = movies_df[movies_df['title'] == movie_title].index[0]
    distances, indices = knn_model.kneighbors(tfidf_matrix[movie_index], n_neighbors=k+1)
    recommended_movies = [movies_df.iloc[idx]['title'] for idx in indices.flatten()[1:]]
    return recommended_movies

# Streamlit UI
st.title('Movie Recommendation System')

selected_movie = st.selectbox(
    'Select a movie:',
    movies_df['title'].values
)

if st.button('Get Recommendations'):
    st.write("### Recommendations for", selected_movie)
    recommendations = get_recommendations(selected_movie)
    for i, movie in enumerate(recommendations):
        st.write(f"{i+1}. {movie}")
