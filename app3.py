import streamlit as st
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
    sim_scores = sim_scores[1:11]  # Get scores of the 10 most similar series
    series_indices = [i[0] for i in sim_scores]
    return data.iloc[series_indices][['title', 'IMDB Rating']].sort_values(by='IMDB Rating', ascending=False)


def recommend_based_on_demographics(age, gender, country):
    """
    Mock-up recommendation function based on demographics (not used in this version).
    """
    # Example weights for age, gender, and country (not used)
    weights = {'age': 0.5, 'gender': 0.3, 'country': 0.2}

    # Randomly generate a demo score to simulate demographic scoring (mock-up)
    data['demo_score'] = np.random.rand(len(data))

    # Return top 10 series sorted by the mock-up demographic score and IMDB Rating (mock-up)
    return data.sort_values(by=['demo_score', 'IMDB Rating'], ascending=[False, False]).head(10)[['Series Title', 'IMDB Rating']]


def main():
    """
    Streamlit app layout and logic
    """

    st.title("Movie Recommendation System")

    # User input for series title
    user_series = st.text_input("Enter the name of a series you like:")

    # Recommendation type selection (currently only content-based is available)
    recommendation_type = st.selectbox("Recommendation Type", ["Content-Based"])

    if st.button("Recommend"):
        if recommendation_type == "Content-Based":
            recommendations = recommend_content_based(user_series)
            if recommendations is not None:
                st.header("Recommendations Based on Similar Content:")
                st.table(recommendations)
            else:
                st.warning("No recommendations found for the entered series title.")
        else:
            st.warning("Demographic-based recommendations are not currently available.")


if __name__ == "__main__":
    main()
