"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity

# Importing data
movies = pd.read_csv('../resources/data/movies.csv')
movies_vec = load_npz('../resources/data/movies_vec.npz')
# ratings = pd.read_csv('resources/data/ratings.csv')
# movies.dropna(inplace=True)

def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """# Initializing the empty list of recommended movies
    recommended_movies = []
    indices = pd.Series(movies['title'])
    
    index = []

    for movie in movie_list:
        index.append(indices[indices == movie].index[0])
        
    # create a user profile using the average of the movie features
    profile = movies_vec[index[0]]

    for i in range(len(index)):
        if i == 0:
            continue
        profile = profile + movies_vec[index[i]]

    profile = profile/len(index)
    
    # get an array of similarities
    similarity = cosine_similarity(movies_vec, profile).reshape(1,-1)[0]
    # convert to a pandas series and sort the resulting series in descending order
    profile_similarity = pd.Series(similarity, index=movies.title, name='name')
    profile_similarity = profile_similarity.sort_values(ascending=False)
    # remove the movies selected by the user from the top matches
    forbidden = set(movie_list)
    similar_movies = profile_similarity.index.to_numpy()
    count = 0
    for i in range(200):
        if similar_movies[i] in forbidden:
            continue
        recommended_movies.append(similar_movies[i])
        count = count + 1

        if count == top_n:
            break
    return recommended_movies