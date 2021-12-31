# 1. PREPROCESSING
# Import Python Libraries


import numpy as np
import scipy as sp
import pandas as pd
from IPython.display import display, HTML
from scipy.stats import pearsonr

df = pd.read_csv('MovieLens.csv')



# 99999999999999999999999999999 -------------- 99999999999999999999999999999
# this line indicates that there are some values you need to change to get other number of centroids 



# PREPROCESSING
# 1. CHECK IF THERE ARE MISSING VALUES
# 2. CONVERT THE DF (USERID, MOVIEID, RATING) INTO A MATRIX OF USER VECTORS
# 3. CONVERT THE USER VECTORS (DF_RATINGS) INTO A SPARSE MATRIX
#
#     @params df
#     @return sdf, df (in trials, the df and sdf is reduced to see if the code works without waiting so much time)

def preprocessing(df):

    # 1. CHECK IF THERE ARE MISSING VALUES
    cols = df.columns
    print('ColumnName, DataType, MissingValues')
    for i in cols:
        print(i, ',', df[i].dtype, ',', df[i].isnull().any())

    # 2. CONVERT THE DF (USERID, MOVIEID, RATING) INTO A MATRIX OF USER VECTORS
    df_ratings = df.pivot(index='userId', columns='movieId', values='rating')

    # Fill missing values with 0
    df_ratings = df_ratings.fillna(0)

    # 3. CONVERT THE USER VECTORS (DF_RATINGS) INTO A SPARSE MATRIX (SDF)
    sdf = df_ratings.astype(pd.SparseDtype("float", 0.0))
    print(sdf.shape)

    # Iu_df : df of the number of items rated by each user

    Iu_df = pd.DataFrame(df.groupby('userId').count()[['movieId']])
    Iu_df = Iu_df.rename(columns={'movieId': 'moviesRated'}, inplace=False)

    ########### !!!!!!!!!  Delete rows using iloc selector  !!!!!!!!! ##################
    #Iu_df = Iu_df.iloc[120:200,]

    ################## !!!!!!!!!                      !!!!!!!!!  ##################

    print('DF of the number of items rated by each user')
    display(HTML(Iu_df.head(5).to_html()))

    ########### !!!!!!!!!  Delete rows using iloc selector  !!!!!!!!! ##################
    #sdf = sdf.iloc[120:200,]

    ################## !!!!!!!!!                      !!!!!!!!!  ##################

    #### Getting the power user and his number of ratings ###

    # Iup = total number of ratings of the user that has rated the max number of movies
    Iup = Iu_df.moviesRated.max()
    print('Iup', Iup)

    display(HTML(sdf.head(3).to_html()))

    print('dense : {:0.2f} bytes'.format(
        df_ratings.memory_usage().sum() / 1e3))
    print('sparse: {:0.2f} bytes'.format(sdf.memory_usage().sum() / 1e3))

    return sdf, Iup, Iu_df


sdf2, Iup, Iu_df = preprocessing(df)

# sdf2 is not necessary right now
# sdf2.to_csv(r'sdf.csv')

# IMPORTING THE ALREADY EXISTING SPARSE DF - RESULT OF THE PREPROCESSING PART
sdf = pd.read_csv('sdf.csv')
sdf.index = sdf.userId
sdf = sdf.drop(columns={'userId'}).astype(pd.SparseDtype("float", 0.0))
print('imported sdf')
display(HTML(sdf.head(7).to_html()))

# 99999999999999999999999999999 -------------- 99999999999999999999999999999
# IMPORTING THE ALREADY EXISTING CENTROID SET
c_set = pd.read_csv('95_centroids.csv')
c_set.index = c_set.userId
c_set = c_set.drop(columns={'userId'}).astype(pd.SparseDtype("float", 0.0))
print('imported c_set')
display(HTML(c_set.head(7).to_html()))


# IF THERE IS NOT ANY CENTROID SET ALREADY CREATED
# SELECTION OF THE FIRST CENTROID
# 1. IU_DF COLUMNS: USERID AND NUMBER OF ITEMS RATED
# 2. GETTING THE UP - USER WHO RATED THE HIGHEST AMOUNT OF ITEMS
# 3. CREATION OF THE SET OF CENTROIDS - DF THAT WILL CONTAIN C1 AND THE REST OF CENTROIDS
#      @params df - initial dataframe / users - sdf vector of each user as rows
#      @return c_set - set of centroids containing c1
def select_c1(df, users, Iu_df, Iup):

    # c1 = userId of the Power User
    c1_id = Iu_df.moviesRated.idxmax()
    print('C1_id', c1_id)
    display(HTML(users.head(3).to_html()))

    c1 = users.loc[c1_id]
    c_set = pd.DataFrame(c1)
    c_set = c_set.transpose().astype(pd.SparseDtype("float", 0.0))
    print('DF of the centroid 1 - power user')
    display(HTML(c_set.head(2).to_html()))
    return c_set

#c_set= select_c1(df,sdf, Iu_df, Iup)


# AUXILIAR PROCEDURE TO FIND MAX PROBABILITY OF USERS - NEXT CENTROID
# Procedure to find the max probability
# @params users = dataframe of the vector users
# k = clusters
# c = c_set - dataframe with the centroids
# iu = Iu_df, iup = Iup

def max_probability(users, c, iu, iup):
    print('im in max_probability')
    max_sim_ui_c = 0
    sum_dist_users = 0
    sum_powers = 0
    power_users = np.empty(0, dtype=float)
    distance_users = np.empty(0, dtype=float)
    sim_users = np.empty(0, dtype=float)

    ##################### 2 parts #####################
    ##################### 1 #####################
    # 1. Calculation of the sumatories of the prob
    for i in users.index:
        max_sim_ui_c = 0
        for j in c.index:
            pcc, p = pearsonr(users.loc[i], c.loc[j])
            similarity1 = 1 + pcc

            if similarity1 >= max_sim_ui_c:
                max_sim_ui_c = similarity1

        if max_sim_ui_c == 0:
            distance_ui_c = 1
        else:
            distance_ui_c = 1/max_sim_ui_c

        # array that contains all the distances bw the user and the closest centroid
        distance_users = np.append(distance_users, distance_ui_c)
        # sum of all the distances^2 bw the users and their closest centroid
        sum_dist_users += distance_ui_c**2

        # user power
        Iu_i = iu.loc[i]
        power_ui = abs(Iu_i) / abs(iup)

        # array that contains the power of each user
        power_users = np.append(power_users, power_ui)
        # sum of all the powers^2 of the users
        sum_powers += power_ui**2

    ##################### 2 #####################
    # 2. Calculation of the probability for all users
    prob_array = np.empty(0, dtype=float)
    prob_user = 0
    f_1 = 0
    f_2 = 0

    for i in range(len(users)):
        f_1 = distance_users[i]**2/sum_dist_users
        f_2 = power_users[i]**2/sum_powers
        prob_user = f_1+f_2
        prob_array = np.append(prob_array, prob_user)

        f_1 = 0
        f_2 = 0

    # The index will be used in the future
    movies_rated = iu['moviesRated']

    # &&&&&&&   &&&&&&& DELETE THE OTHER DPP_DF
    dpp_df = pd.DataFrame({'Movies rated': movies_rated,
                          'User probability': prob_array})

    # for all users - c_set, keep the prob
    users_aux = users.copy(deep=True)
    for i in c.index:
        users_aux = users_aux.drop(index=i)
        dpp_df = dpp_df.drop(index=i)

    print('# rows of the array (users - centroids): ', len(dpp_df))

    print('Dataframe with probability of each user')
    display(HTML(dpp_df.head(7).to_html()))

    # id of the centroid with max_probability
    new_centroid = dpp_df[['User probability']].idxmax()
    # max_probability
    max_prob = dpp_df[['User probability']].max()
    print('id new centroid: ', new_centroid)
    print('prob of the new centroid: ', max_prob)
    return new_centroid


def kmeansplusprobpower(df, users, k, c_set, Iup, Iu_df):
    # choose initial centroid c1 = up
    #c_set, Iup, Iu_df = select_c1(df, users)
    print('c_set before kmeansplusprobpower: ')
    display(HTML(c_set.head(len(c_set)).to_html()))
    # auxiliar c_set, contains the initial set of centroids
    c_aux = c_set.copy()

    # choose next centroid ci selecting ci = u'€ U with max prob
    # loop to calculate k centroids
    # mind that some of them may be already calculated
    for j in range(len(c_aux), k):
        print('im in the loop')
       # calculation of the user id with max prob
        ci_id = max_probability(users, c_set, Iu_df, Iup)

        c2 = users.loc[ci_id]
        # here c2 is kept as a row
        new_c_df = pd.DataFrame(c2)
        new_c_df = new_c_df.transpose()
        
        # here new_c_df as a column
        print(new_c_df)
        # c_set now transposed
        c_set = c_set.transpose()
        # joining the new_c_df column into the c_set centroids are the set of columns
        c_set = c_set.join(new_c_df)


        c_set = c_set.transpose().astype(pd.SparseDtype("float", 0.0))
        display(HTML(c_set.tail(3).to_html()))
    print('c_set after kmeansplusprobpower: ')
    display(HTML(c_set.head(k).to_html()))
    return c_set


# 99999999999999999999999999999 -------------- 99999999999999999999999999999
# change the 105 numbers you find below here to update the results
centroids_set_105 = kmeansplusprobpower(df, sdf, 105, c_set, Iup, Iu_df)

centroids_set_105.to_csv(r'105_centroids.csv')

