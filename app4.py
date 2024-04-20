import streamlit as st
import pickle

movies = pickle.load(open("series_list.pkl", 'rb'))
similarity = pickle.load(open("similarity1.pkl", 'rb'))
movies_list = movies['title'].values

# Function to recommend movies (same as before)
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    recommend_movie = [movies.iloc[i[0]].title for i in distance[1:6]]
    return recommend_movie

# Start page (same as before)
def start_page():
    st.image("the.jpg", use_column_width=True)
    st.title("Welcome to Binge Buddy")
    if st.button("Get Started"):
        st.session_state["page"] = "Recommender"
    return False

# Recommender page for collaborative filtering (same as before)
def recommender_page():
    st.header("Series Recommender System")
    select_value = st.selectbox("Select series from dropdown", movies_list)
 
    if st.button("Show Recommendations"):
        movie_names = recommend(select_value)
        st.subheader("Recommended Series:")
        for movie_name in movie_names:
            st.write(movie_name)
 
    if st.button("View Details"):
        selected_movie = st.selectbox("Select a movie to view details:", movies_list)
        movie_details = movies[movies['title'] == selected_movie]
        st.subheader("Selected Series Details:")
        st.write(movie_details)
def recommend_based_on_demographics(age, gender, country):
  # Example weights for age, gender, and country (adjust as needed)
  weights = {'age': 0.5, 'gender': 0.3, 'country': 0.2}

  # Mock-up demographic scoring (replace with your actual logic)
  demo_score = 0.0
  demo_score += (age - 25) * weights['age']  # Adjust baseline age and weight
  if gender.lower() == "male":
    demo_score += weights['gender']  # Adjust weight for gender
  # Add logic for country (consider user preferences or regional popularity)

  # Random noise to simulate score variation (replace with actual scoring)
  demo_score += np.random.rand() * 0.1

  # Combine demo score and IMDB rating for recommendations
  data['demo_score'] = demo_score
  recommendations = data.sort_values(by=['demo_score', 'IMDB Rating'], ascending=[False, False]).head(10)[['title', 'IMDB Rating']]

  return recommendations

# Content-based recommender page
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load the dataset (outside the function for performance)
data = pd.read_csv('wsmetadata.csv', encoding='ISO-8859-1')

# Create combined features for content-based filtering (outside the function)
data['combined_features'] = data[['title', 'Genre', 'Description']].fillna('').astype(str).agg(' '.join, axis=1)

# Initialize TF-IDF Vectorizer globally (avoid redundant creation)
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(data['combined_features'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


def recommend_content_based(series_title):
    """
    Recommends similar series based on content using TF-IDF and cosine similarity.
    """
    try:
        idx = data.index[data['title'].str.lower() == series_title.lower()].tolist()[0]
    except IndexError:
        st.error("Series title not found in the dataset.")
        return None

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11] 
    series_indices = [i[0] for i in sim_scores]
    return data.iloc[series_indices][['title', 'IMDB Rating']].sort_values(by='IMDB Rating', ascending=False)


def content_based_page():
    """
    Streamlit app layout and logic for content-based recommendations
    """
    st.title("Series Recommendation System - Content-Based")

    # User input for series title
    user_series = st.text_input("Enter the name of a series you like:")
    gender_options = ("Male", "Female", "Non-binary", "Prefer Not to Say")
    user_gender = st.selectbox("Select your preferred gender for recommendations:", gender_options)

    if st.button("Recommend"):
        recommendations = recommend_content_based(user_series)
        if recommendations is not None:
            st.header("Recommendations Based on Similar Content:")
            st.table(recommendations)
        else:
            st.warning("No recommendations found for the entered series title.")
def page3():
  st.title("Series Recommendation System")

  # Collect user inputs
  age = st.number_input("Enter your age: ")
  gender = st.selectbox("Select your gender:", ["Male", "Female", "Other"])
  country = st.text_input("Enter your country: ")

  # Recommend movies based on user input
  if st.button("Recommend Series"):
    recommendations = recommend_based_on_demographics(age, gender, country)
    if not recommendations.empty:
      st.subheader("Recommendations for you based on demographics:")
      st.dataframe(recommendations)
    else:
      st.warning("No recommendations found based on your demographics.")

# Main function
def main():
  """
  Main function to handle navigation and call appropriate recommendation pages
  """
  page = st.sidebar.radio("Navigation", ["Start", "Recommender", "Content-Based","Demographic-Based"])

  if page == "Start":
    if start_page():
       recommender_page()  # Exit after start page interaction

  if page == "Recommender":
    recommender_page()

  if page == "Content-Based":
    content_based_page()
  if page == "Demographic-Based":
    page3()

if __name__ == "__main__":
  main()
