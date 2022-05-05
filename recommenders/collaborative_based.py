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
import operator
import pandas as pd
import numpy as np
import pickle
# import pickle
import copy
import scipy as sp
# from surprise import Reader, Dataset
# from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

train_df = pd.read_csv('~/data/train.csv')
train_df.drop(['timestamp'], axis=1,inplace=True)
model = pickle.load(open('../../data/220422_svd.pkl', 'rb'))
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
    # Data preprosessing
    # print(f"Time:{time.asctime(time.localtime())} : Start Surprise Reading Data")

    # reader = Reader(rating_scale=(0, 5))
    # load_df = Dataset.load_from_df(train_df,reader)
    # a_train = load_df.build_full_trainset()

    print(f"Time:{time.asctime(time.localtime())} : Start Predictions")

    movie_pred_df = pd.DataFrame(
    {
        'userId':list(set(filter_train_df['userId'].tolist()))
    }
    )


    # user_ids = set(np.tolist(train_df['userId']))
    try:
        # predictions = model.predict(uid=set(np.tolist(train_df['userId'])), iid=np.full((item_id)), verbose= False)
        movie_pred_df['prediction'] = movie_pred_df.apply(lambda x : MODEL.predict(uid = x['userId'], iid= item_id)[3], axis=1)
    except Exception as e:
        print(e)
    # for ui in A_TRAIN.all_users():
    #     predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
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
    # Store the id of users
    # id_store=[]

    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    # movie_list_ids = [movies_df.loc[movies_df['title']== movie,'movieId'].item() for movie in movie_list] 
    filter_train_df = TRAIN_DF.loc[(TRAIN_DF['movieId'].isin(movie_list_ids)) \
                                 & (TRAIN_DF['rating'] > 3) ,:]

    #Aggregate dataset for the three movies selected.
    ideal_sim_user = TRAIN_DF.loc[TRAIN_DF['movieId'].isin(movie_list_ids),:]\
            .groupby('userId') \
            .agg(rating_sum= ('rating', 'sum')) \
            .sort_values('rating_sum', ascending= False) \
            .iloc[0].name
    

    #If too few users are available with the above specification
    #ie. the movies are not rated highly
    if filter_train_df.shape[0] < 3000:
        filter_train_df = TRAIN_DF.sample(n= 3500)

    id_store_df = pd.DataFrame()

    for i in movie_list_ids:
        print(f"Time:{time.asctime(time.localtime())} : Start Predictions for {i}")

        #Filter Train_df based on the ratings. ie. only consider users
        # a >3 rating for the selected movies

        prediction_df = prediction_item(item_id = i,
                                        filter_train_df= filter_train_df)

        # predictions = prediction_df['prediction']
        # predictions = prediction_item(item_id = i)
        prediction_df.sort_values('prediction', ascending= False, inplace= True)
        # predictions.sort(key=lambda x: x.est, reverse=True)

        prediction_df.reset_index(drop= True, inplace= True)

        print(f"Time:{time.asctime(time.localtime())} : Complete Predictions for {i}")

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

    # MOVIE_INDICES = pd.Series(movies_df['title']) #MOVIE_INDICES- pandas series of the Movie titles
    sim_user_ids = pred_movies(movie_list_ids)['userId'].tolist()   #sim_user_ids- holds a list of userIDs who rated the selected movies highly

    if ideal_sim_user not in sim_user_ids:
        sim_user_ids.append(ideal_sim_user)
    # df_init_users = ratings_df[ratings_df['userId']==sim_user_ids[0]]      #initializes the df_init_users DF
    df_init_users = pd.DataFrame()      #initializes the df_init_users DF
    print(f"Time:{time.asctime(time.localtime())} : Start Init Iteration")


    df_init_users = train_df.loc[train_df['userId'].isin(sim_user_ids),:]
    # for i in sim_user_ids:                                                #for each userID in movies_id,
    #     df_init_users=df_init_users.append(train_df[train_df['userId']==i]) #Append their ratings to the df_init_users DF
    util_matrix = pd.pivot_table(df_init_users, index= 'userId', columns= 'movieId', values= 'rating')
    print(f"Time:{time.asctime(time.localtime())} : Finished Init Iteration")


    #------------------------ 220430

    #Normalize each row
    print(f"Time:{time.asctime(time.localtime())} : Creating Utility & Sparse Matrices")

    util_matrix_norm = util_matrix.apply(lambda x: (x - np.mean(x))/(np.max(x) - np.min(x)), axis= 1)
    util_matrix_norm.fillna(0, inplace= True)
    util_matrix_norm = util_matrix_norm.T
    util_matrix_norm = util_matrix_norm.loc[:,(util_matrix_norm != 0).any(axis= 0)]

    util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)

    users_cosine_similarity = cosine_similarity((util_matrix_sparse.T))

    sim_users_df = pd.DataFrame(users_cosine_similarity,
                           index = util_matrix_norm.columns,
                           columns = util_matrix_norm.columns)

    print(f"Time:{time.asctime(time.localtime())} : Completed Creating Utility & Sparse Matrices")


    print(f"Time:{time.asctime(time.localtime())} : Selecting Similar Users")

    fav_sim_users_movies = []
    for user in sim_users_df.sort_values(by= ideal_sim_user, ascending= False).index:
        max_score = util_matrix_norm.loc[:,user].max()

        fav_sim_users_movies.append(util_matrix_norm[util_matrix_norm.loc[:,user] == max_score].index.tolist())

    flat_fav_sim_users_movies = [fav_movie for fav_sim_user_movies in fav_sim_users_movies for fav_movie in fav_sim_user_movies]

    tally_favs = {movie:flat_fav_sim_users_movies.count(movie) for movie in set(flat_fav_sim_users_movies)}

    sort_favs = sorted(tally_favs.items(), key= operator.itemgetter(1), reverse= True)

    print(f"Time:{time.asctime(time.localtime())} : Developing Recommendations")


    top_ten_recommends = [movies_df.loc[movies_df['movieId']== tallied_movie[0],'title'].item() \
                     for tallied_movie in sort_favs[:11] \
                        if movies_df.loc[movies_df['movieId']== tallied_movie[0],'title'].item() not in movie_list]

    print(f"Time:{time.asctime(time.localtime())} : Completed developing Recommendations")
        

    # End of Working Code

    return top_ten_recommends
