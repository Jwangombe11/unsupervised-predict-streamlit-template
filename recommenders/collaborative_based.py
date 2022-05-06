"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import operator
import pandas as pd
import numpy as np
import pickle
import scipy as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

train_df = pd.read_csv('~/data/train.csv')
train_df.drop(['timestamp'], axis=1,inplace=True)
model = pickle.load(open('resources/models/220422_svd.pkl', 'rb'))
movies_df = pd.read_csv('resources/data/movies.csv')
movies_df.dropna(inplace= True)

def prediction_item(item_id, filter_train_df):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """

    movie_pred_df = pd.DataFrame(
    {
        'userId':list(set(filter_train_df['userId'].tolist()))
    }
    )

    try:
        movie_pred_df['prediction'] = movie_pred_df.apply(lambda x : model.predict(uid = x['userId'], iid= item_id)[3], axis=1)
    except Exception as e:
        print(e)
    return movie_pred_df

def pred_movies(movie_list_ids):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    filter_train_df = train_df.loc[(train_df['movieId'].isin(movie_list_ids)) \
                                 & (train_df['rating'] > 3) ,:]

    #Aggregate dataset for the three movies selected.
    ideal_sim_user = train_df.loc[train_df['movieId'].isin(movie_list_ids),:]\
            .groupby('userId') \
            .agg(rating_sum= ('rating', 'sum')) \
            .sort_values('rating_sum', ascending= False) \
            .iloc[0].name
    

    #If too few users are available with the above specification
    #ie. the movies are not rated highly
    if filter_train_df.shape[0] < 3000:
        filter_train_df = train_df.sample(n= 3500)

    id_store_df = pd.DataFrame()

    for i in movie_list_ids:
        #Filter Train_df based on the ratings. ie. only consider users
        # a >3 rating for the selected movies

        prediction_df = prediction_item(item_id = i,
                                        filter_train_df= filter_train_df)
                                        
        prediction_df.sort_values('prediction', ascending= False, inplace= True)

        prediction_df.reset_index(drop= True, inplace= True)

        # Take the top 10 user id's from each movie with highest rankings
        # for pred in predictions[:10]:
        #     id_store.append(pred.uid)
        id_store_df = id_store_df.append(prediction_df.loc[:30,:])
        # id_store_df = prediction_df.loc[:10,:]
    # Return a list of user id's
    return id_store_df

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n):
    """Performs Collaborative filtering based upon a list of movies supplied
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

    """
    movie_list_ids = [movies_df.loc[movies_df['title']== movie,'movieId'].item() for movie in movie_list] 


    #Ideal User based on Aggregate rating for the three movies selected.
    ideal_sim_user = train_df.loc[train_df['movieId'].isin(movie_list_ids),:]\
            .groupby('userId') \
            .agg(rating_sum= ('rating', 'sum')) \
            .sort_values('rating_sum', ascending= False) \
            .iloc[0].name

    sim_user_ids = pred_movies(movie_list_ids)['userId'].tolist() 

    if ideal_sim_user not in sim_user_ids:
        sim_user_ids.append(ideal_sim_user)
    df_init_users = pd.DataFrame()   


    df_init_users = train_df.loc[train_df['userId'].isin(sim_user_ids),:]
    
    util_matrix = pd.pivot_table(df_init_users, index= 'userId', columns= 'movieId', values= 'rating')

    util_matrix_norm = util_matrix.apply(lambda x: (x - np.mean(x))/(np.max(x) - np.min(x)), axis= 1)
    util_matrix_norm.fillna(0, inplace= True)
    util_matrix_norm = util_matrix_norm.T
    util_matrix_norm = util_matrix_norm.loc[:,(util_matrix_norm != 0).any(axis= 0)]

    util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)

    users_cosine_similarity = cosine_similarity((util_matrix_sparse.T))

    sim_users_df = pd.DataFrame(users_cosine_similarity,
                           index = util_matrix_norm.columns,
                           columns = util_matrix_norm.columns)


    fav_sim_users_movies = []
    for user in sim_users_df.sort_values(by= ideal_sim_user, ascending= False).index:
        max_score = util_matrix_norm.loc[:,user].max()

        fav_sim_users_movies.append(util_matrix_norm[util_matrix_norm.loc[:,user] == max_score].index.tolist())

    flat_fav_sim_users_movies = [fav_movie for fav_sim_user_movies in fav_sim_users_movies for fav_movie in fav_sim_user_movies]

    tally_favs = {movie:flat_fav_sim_users_movies.count(movie) for movie in set(flat_fav_sim_users_movies)}

    sort_favs = sorted(tally_favs.items(), key= operator.itemgetter(1), reverse= True)


    top_ten_recommends = [movies_df.loc[movies_df['movieId']== tallied_movie[0],'title'].item() \
                     for tallied_movie in sort_favs[:11] \
                        if movies_df.loc[movies_df['movieId']== tallied_movie[0],'title'].item() not in movie_list]

    return top_ten_recommends
