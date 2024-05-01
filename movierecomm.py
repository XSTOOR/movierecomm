import streamlit as st
import pandas as pd

# Load the movie dataset
@st.cache
def load_data():
    return pd.read_csv("movies.csv")

movies_df = load_data()

# Function to calculate similarity based on genres
def calculate_similarity(movie_genres_1, movie_genres_2):
    if not movie_genres_1 or not movie_genres_2:
        return 0  # If any of the genres is empty, return 0 similarity
    try:
        genres_1 = set(movie_genres_1.split('|'))
        genres_2 = set(movie_genres_2.split('|'))
    except AttributeError:
        return 0  # If genres are not in the expected format, return 0 similarity
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

# Set page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon=":movie_camera:",
    layout="centered",  # 'wide'
    initial_sidebar_state="expanded")  # 'auto', 'expanded', 'collapsed'

# Set background color
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f3f4;  /* red */
    }
    </style>
    """,
    unsafe_allow_html=True
)

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

