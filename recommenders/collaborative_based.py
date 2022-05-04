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
import time
import pandas as pd
import numpy as np
import pickle
import copy
import scipy as sp
from surprise import Reader, Dataset
# from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
print(f"Time:{time.asctime(time.localtime())} : Start loading Model")
model=pickle.load(open('resources/models/220422_svd.pkl', 'rb'))
print(f"Time:{time.asctime(time.localtime())} : Start loading DataFrames")
movies_df = pd.read_feather('resources/data/movies.feather')
# ratings_df = pd.read_csv('resources/data/ratings.csv')
# ratings_df.drop(['timestamp'], axis=1,inplace=True)
print(f"Time:{time.asctime(time.localtime())} : Start loading Train DataFrames")
train_df = pd.read_feather('resources/data/train.feather')
train_df.drop(['timestamp'], axis=1,inplace=True)

print(f"Time:{time.asctime(time.localtime())} : Finish loading DataFrames")


# reader = Reader(rating_scale=(0, 5))
# LOAD_DF = Dataset.load_from_df(train_df,reader)
# A_TRAIN = LOAD_DF.build_full_trainset()



# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
# model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

def prediction_item(item_id):
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
    # Data preprosessing
    print(f"Time:{time.asctime(time.localtime())} : Start Surprise Reading Data")

    # reader = Reader(rating_scale=(0, 5))
    # load_df = Dataset.load_from_df(train_df,reader)
    # a_train = load_df.build_full_trainset()

    print(f"Time:{time.asctime(time.localtime())} : Start Predictions")

    movie_pred_df = pd.DataFrame(
    {
        'userId':list(set(train_df['userId'].tolist()))
    }
    )


    # user_ids = set(np.tolist(train_df['userId']))
    try:
        # predictions = model.predict(uid=set(np.tolist(train_df['userId'])), iid=np.full((item_id)), verbose= False)
        movie_pred_df['prediction'] = movie_pred_df.apply(lambda x : model.predict(uid = x['userId'], iid= item_id)[3], axis=1)
    except Exception as e:
        print(e)
    # for ui in A_TRAIN.all_users():
    #     predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return movie_pred_df

def pred_movies(movie_list):
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
    # Store the id of users
    # id_store=[]
    id_store_df = pd.DataFrame()

    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        print(f"Time:{time.asctime(time.localtime())} : Start Predictions for {i}")

        prediction_df = prediction_item(item_id = i)
        # predictions = prediction_df['prediction']
        # predictions = prediction_item(item_id = i)
        prediction_df.sort_values('prediction', ascending= False, inplace= True)
        # predictions.sort(key=lambda x: x.est, reverse=True)
        print(f"Time:{time.asctime(time.localtime())} : Complete Predictions for {i}")

        # Take the top 10 user id's from each movie with highest rankings
        # for pred in predictions[:10]:
        #     id_store.append(pred.uid)
        id_store_df = id_store_df.append(prediction_df.loc[:10,:])
        # id_store_df = prediction_df.loc[:10,:]
    # Return a list of user id's
    return id_store_df

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
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

    indices = pd.Series(movies_df['title']) #indices- pandas series of the Movie titles
    sim_user_ids = pred_movies(movie_list)['userId']     #sim_user_ids- holds a list of userIDs who rated the selected movies highly
    # df_init_users = ratings_df[ratings_df['userId']==sim_user_ids[0]]      #initializes the df_init_users DF
    df_init_users = pd.DataFrame()      #initializes the df_init_users DF
    print(f"Time:{time.asctime(time.localtime())} : Start Init Iteration")

    for i in sim_user_ids:                                                #for each userID in movies_id,
        df_init_users=df_init_users.append(train_df[train_df['userId']==i]) #Append their ratings to the df_init_users DF
    util_matrix = pd.pivot_table(df_init_users, index= 'userId', columns= 'movieId', values= 'rating')
    print(f"Time:{time.asctime(time.localtime())} : Start Init Iteration")


    #------------------------ 220430

    #Normalize each row
    util_matrix_norm = util_matrix.apply(lambda x: (x - np.mean(x))/(np.max(x) - np.min(x)), axis= 1)
    util_matrix_norm.fillna(0, inplace= True)
    util_matrix_norm = util_matrix_norm.T
    util_matrix_norm = util_matrix_norm.loc[:,(util_matrix_norm != 0).any(axis= 0)]

    util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)

    user_similarity = cosine_similarity((util_matrix_sparse.T))


    fav_user_movies = []
    for user_id in sim_user_ids:
        max_score = util_matrix_norm.loc[:,user_id].max()

        fav_user_movies.append(util_matrix_norm[util_matrix_norm.loc[:,user_id] == max_score]).index.tolist()

    tally_favs = {movie:fav_user_movies.count(movie) for movie in fav_user_movies}

    sort_favs = sorted(tally_favs.items(), key= operator.itemgetter(1), reverse= True)




    #Get the Cosine Similarity Matrix






    #------------------------------










    # Getting the cosine similarity matrix
    cosine_sim = cosine_similarity(np.array(df_init_users), np.array(df_init_users))
    # cosine_sim = cosine_similarity(piv_table,piv_table)
    #Not sure what happens above. Might be that the variables are wrong.
    idx_1 = indices[indices == movie_list[0]].index[0]              #Get indexes of the three selected movies.
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1] #Get the 'rank' of the movies in the cosine_similarity matrix. This is where the code fails
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]
    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
     # Appending the names of movies
    listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
    recommended_movies = []
    # Choose top 50
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies_df['title'])[i])
    return recommended_movies
