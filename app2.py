import streamlit as st
import pickle

movies = pickle.load(open("series_list.pkl", 'rb'))
similarity = pickle.load(open("similarity1.pkl", 'rb'))
movies_list = movies['title'].values

# Function to recommend movies
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda vector: vector[1])
    recommend_movie = [movies.iloc[i[0]].title for i in distance[1:6]]
    return recommend_movie

# Start page
def start_page():
    st.image("the.jpg", use_column_width=True)
    st.title("Welcome to Binge Buddy")
    if st.button("Get Started"):
        st.session_state["page"] = "recommender_page" 
    return False

# Main page for recommender system
def recommender_page():
    st.header("Movie Recommender System")
    select_value = st.selectbox("Select movie from dropdown", movies_list)
    
    if st.button("Show Recommendations"):
        movie_names = recommend(select_value)
        st.subheader("Recommended Movies:")
        for movie_name in movie_names:
            st.write(movie_name)
    
    if st.button("View Details"):
        selected_movie = st.selectbox("Select a movie to view details:", movies_list)
        movie_details = movies[movies['title'] == selected_movie]
        st.subheader("Selected Movie Details:")
        st.write(movie_details)

# Main function
def main():
    page = st.sidebar.radio("Navigation", ["Start", "Recommender"])
    
    if page == "Start":
        if start_page():
            st.experimental_rerun()
    elif page == "Recommender":
        recommender_page()

if __name__ == "__main__":
    main()
