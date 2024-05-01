import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise
from sklearn.feature_extraction.text import CountVectorizer

# Load the movie dataset
movies_df = pd.read_csv("movies.csv")

# Create a CountVectorizer object
count_vectorizer = CountVectorizer(stop_words='english')
count_matrix = count_vectorizer.fit_transform(movies_df['genres'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(count_matrix, count_matrix)

# Function to get movie recommendations
def get_recommendations(movie_title, cosine_sim=cosine_sim):
    idx = movies_df[movies_df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    movie_indices = [i[0] for i in sim_scores]
    return movies_df['title'].iloc[movie_indices]

# Streamlit UI
st.title('Movie Recommendation System')

selected_movie = st.selectbox(
    'Select a movie:',
    movies_df['title'].values
)

if st.button('Get Recommendations'):
    recommendations = get_recommendations(selected_movie)
    st.write(recommendations)
