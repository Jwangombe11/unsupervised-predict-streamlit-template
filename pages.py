import streamlit as st

# Streamlit dependencies
from PIL import Image	

# Load Website's photo clip art
clip_art = Image.open('resources/imgs/EDSA_logo.png')

def project_overview():
    st.title('CHILLFLIX')
    st.subheader('Developed by JitT Inc. - Team 7')
    st.image('resources/imgs/app_name.png')
    st.markdown(get_markdown('resources/markdowns/project_overview/app_intro.md'))
        
    st.markdown('## The Data')
    st.image('resources/imgs/data.png')
    st.markdown(get_markdown('resources/markdowns/project_overview/data.md'))
    st.markdown('## A deeper dive')
    st.write('-----------')
    st.image('resources/imgs/most_generes.png')
    st.markdown(get_markdown('resources/markdowns/project_overview/key_insight_1.md'))
    st.write('-----------')
    st.image('resources/imgs/rating_distribution.png')
    st.markdown(get_markdown('resources/markdowns/project_overview/key_insight_2.md'))
    st.write('-----------')
    st.image('resources/imgs/movies_by_year.png')
    st.markdown(get_markdown('resources/markdowns/project_overview/key_insight_3.md'))
    st.write('-----------')
    st.markdown('## Our plans for the future')
    st.image('resources/imgs/future_development.png')
    st.markdown(get_markdown('resources/markdowns/project_overview/future_plans.md'))

def solution_overview():
    st.title("Solution Overview")
    st.markdown(get_markdown('resources/markdowns/solution_overview/intro.md'))
    st.image('resources/imgs/models.png')
    st.write('-----------')
    st.markdown(get_markdown('resources/markdowns/solution_overview/model_performance.md'))
    st.image('resources/imgs/model_performance.png')


def meet_the_team():
    st.title("The team")
    st.image('resources/imgs/team.png')
    st.markdown(get_markdown('resources/markdowns/meet_the_team/team.md'))


def get_markdown(file_name):
    with open(file_name, 'r') as file:
        markdown_text = file.read()
    return markdown_text