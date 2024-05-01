import streamlit as st
import pandas as pd

# Load the movie dataset
@st.cache
def load_data():
    return pd.read_csv("movies.csv")

movies_df = load_data()

# Function to calculate similarity based on genres
def calculate_similarity(movie_genres_1, movie_genres_2):
    genres_1 = set(movie_genres_1.split('|'))
    genres_2 = set(movie_genres_2.split('|'))
    intersection = genres_1.intersection(genres_2)
    similarity = len(intersection) / (len(genres_1) + len(genres_2) - len(intersection))
    return similarity

# Function to get movie recommendations
def get_recommendations(movie_title, threshold=0.2):
    movie_row = movies_df[movies_df['title'] == movie_title]
    movie_genres = movie_row['genres'].values[0]
    recommendations = []
    for index, row in movies_df.iterrows():
        if row['title'] != movie_title:
            similarity = calculate_similarity(movie_genres, row['genres'])
            if similarity >= threshold:
                recommendations.append(row['title'])
    return recommendations

# Streamlit UI
st.title('Movie Recommendation System')

selected_movie = st.selectbox(
    'Select a movie:',
    movies_df['title'].values
)

if st.button('Get Recommendations'):
    st.write("### Recommendations for", selected_movie)
    recommendations = get_recommendations(selected_movie)
    if recommendations:
        for i, movie in enumerate(recommendations):
            st.write(f"{i+1}. {movie}")
    else:
        st.write("No recommendations found for this movie.")
