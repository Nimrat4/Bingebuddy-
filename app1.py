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
    st.image("E:\movie_recommender_system-main\1.png", use_column_width=True)
    st.title("Welcome to Movie Recommender System")
    if st.button("Get Started"):
        return True
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
