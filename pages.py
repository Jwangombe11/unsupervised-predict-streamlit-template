import streamlit as st

# Streamlit dependencies
from PIL import Image	

# Load Website's photo clip art
clip_art = Image.open('resources/imgs/EDSA_logo.png')

def project_overview():
    st.title('CHILLFLIX')
    st.subheader('Developed by JitT Inc. - Team 7')
    st.image('resources/imgs/app_name.png')
    st.markdown(get_markdown('resources/markdowns/project_overview/team_intro.md'))
        
    st.markdown('## some data we had to work with')
    st.image('resources/imgs/data.png')

    st.markdown('## A little indepth look')
    st.image('resources/imgs/most_generes.png')
    st.image('resources/imgs/rating_distribution.png')
    st.image('resources/imgs/movies_by_year.png')

    st.markdown('## Our plans for the future')
    st.image('resources/imgs/future_development.png')

def solution_overview():
    st.title("Solution Overview")
    st.write("Describe your winning approach on this page")
    st.image('resources/imgs/models.png')
    st.image('resources/imgs/model_performance.png')

def meet_the_team():
    st.title("The team")
    st.image('resources/imgs/team.png')
    st.write("Describe your winning approach on this page")

def get_markdown(file_name):
    with open(file_name, 'r') as file:
        markdown_text = file.read()
    return markdown_text