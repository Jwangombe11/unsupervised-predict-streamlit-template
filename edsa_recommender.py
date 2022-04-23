

"""
--------------------------------------------------------
The following application was developed by Team 13: 2110ACDS_T13
For the Advanced Classification Sprint at Explore Data Science Academy.

The application is intended as a text sentiment predictr fr tweet messages.

Authors: Teddy Waweru, , Hunadi Mawela, Uchenna Unigwe, Stanley Agbo 

Github Link: https://github.com/Jwangombe11/unsupervised-predict-streamlit-template 
Official Presentation Link:	https://docs.google.com/presentation/d/1-AIbZcDdUDmvVoIB4WoJcIZslbbdb6S9bujMNEgpuHw/edit?usp=sharing

The content is under the GNU icense & is free-to-use.

"""

# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

#Plotting of Graphs
import plotly.express as px

# Streamlit dependencies
import joblib,os			#Loading the model & accessing OS File System
from PIL import Image		#Importing logo Image
from io import BytesIO		#Buffering Images



#-------------------------------------------------------------------
#START
#-------------------------------------------------------------------


# Load Website's photo clip art
clip_art = Image.open('resources/imgs/EDSA_logo.png') 

#Set the Pages Initial Configuration Settings
st.set_page_config(page_title= 'JitT Inc.: Movie Recommendation System',
					page_icon= clip_art,
					layout="wide",
					menu_items = {
							'Report a Bug': 'https://www.google.com'
					})








# App declaration
def main():



    # Load Datasets
    @st.experimental_singleton
    def load_datasets():
        title_list = load_movie_titles('resources/data/movies.csv')
        return title_list


    # Load Models
    @st.experimental_singleton
    def load_models():
        pass

    # Read Markdown files
    def load_markdown(file):
        return Path(file).read_text()



    def project_overview():
        st.title('Movie Recommendation System')
        st.markdown('---')
        st.subheader('Developed by JitT Inc. - Team 7')

        col1, col2, col3 = st.columns([1,8,1])
        # st.markdown(' - Jessica Njuguna \n - Hunadi Mawela \n - Uchenna Unigwe \n - Stanley Agbo \n - Teddy Waweru \n')
        with col1:
            pass
        with col2:
            # st.markdown('---')
            st.markdown('### Development Team')
            team_members = Image.open('resources/imgs/landing_page_sample.png')
            st.image(team_members)

        with col3:
            pass
			
        st.markdown('## Introduction')
        st.markdown('---')


    def recommender_system():

        title_list = load_datasets()
        # DO NOT REMOVE the 'Recommender System' option below, however,
        # you are welcome to add more options to enrich your app.
        page_options = ["Recommender System","Solution Overview"]

        # -------------------------------------------------------------------
        # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
        # -------------------------------------------------------------------
        # page_selection = st.sidebar.selectbox("Choose Option", page_options)
        # if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                    ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                            We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                        top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                            We'll need to fix it!")


    # -------------------------------------------------------------------
    def solution_overview():
        # ------------- SAFE FOR ALTERING/EXTENSION -------------------
        # if page_selection == "Solution Overview":
            st.title("Solution Overview")
            st.write("Describe your winning approach on this page")

        # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.

    #Dict of available pages to browse on the application
    BROWSE_PAGES = {
        'Project Overview': project_overview,
        'Recommender System': recommender_system,
        'Solution Overview': solution_overview
    }

	#Page Navigation Title & Radio BUttons
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to:', list(BROWSE_PAGES.keys()))

	#Load function depending on radio selected above.
	#Used to navigate through pages
    BROWSE_PAGES[page]()







if __name__ == '__main__':
    main()
