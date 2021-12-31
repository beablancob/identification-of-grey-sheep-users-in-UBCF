# 1. PREPROCESSING
# Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd

from IPython.display import display, HTML

from joblib import Parallel, delayed
from scipy.stats import pearsonr

import timeit


def process(n, k, user_row, centroids_df):
    print('  STARTED user number: ', n)
    #start2 = timeit.default_timer()
    # aux similarity between every user and each centroid
    similarity1 = 0
    # max_sim_ui_c is the max similarity between a user and all the centroids
    max_sim_ui_c = 0
    # centroid j is the most similar centroid for the user
    cluster_j = 0
    # distance_ui_c is the min distance between a user and all the centroids
    distance_ui_c = 0
    # loop for every centroid
    for j in range(k):
        #start3 = timeit.default_timer()
        #print('    cluster number', j)
        # pearson similarity between every user and every centroid
        pcc, p = pearsonr(user_row, centroids_df.iloc[j])
        similarity1 = 1 + pcc
        # when aux similarity is greater than max similarity, keep that value and the corresponding cluster
        if similarity1 >= max_sim_ui_c:
            max_sim_ui_c = similarity1
            cluster_j = j
        #print('    time3',timeit.default_timer()-start3)
    if max_sim_ui_c == 0:
        distance_ui_c = 1
    else:
        distance_ui_c = 1/max_sim_ui_c

    # array that contains all the distances bw the user and the closest centroid
    #distance_users = np.append(distance_users, distance_ui_c)

    # array that contains all the similarities bw the user and the closest centroid, needed for the gsu selection
    #sim_users = np.append(sim_users,max_sim_ui_c)
    # array that contains the cluster where belongs each user
    #cluster = np.append(cluster, cluster_j)
    print('  ENDED user number: ', n)
    return (n, cluster_j, distance_ui_c)
    #print('  time2',timeit.default_timer()-start2)


if __name__ == "__main__":
    # importing the required csv files
    df = pd.read_csv('MovieLens.csv')
    all_ssd = pd.read_csv('all_ssd.csv')
    all_ssd = all_ssd.drop(columns={'Unnamed: 0'})

    # PREPROCESSING
    # 1. CHECK IF THERE ARE MISSING VALUES
    # 2. CONVERT THE DF (USERID, MOVIEID, RATING) INTO A MATRIX OF USER VECTORS
    # 3. CONVERT THE USER VECTORS (DF_RATINGS) INTO A SPARSE MATRIX
    #
    #     @params df
    #     @return sdf, df (in trials, the df and sdf is reduced to see if the code works without waiting so much time)

    def preprocessing(df):

        # 1. CHECK IF THERE ARE MISSING VALUES
        # cols=df.columns
        #print('ColumnName, DataType, MissingValues')
        # for i in cols:
        #    print(i, ',', df[i].dtype,',',df[i].isnull().any())

        # 2. CONVERT THE DF (USERID, MOVIEID, RATING) INTO A MATRIX OF USER VECTORS
        df_ratings = df.pivot(
            index='userId', columns='movieId', values='rating')

        # Fill missing values with 0
        df_ratings = df_ratings.fillna(0)

        # 3. CONVERT THE USER VECTORS (DF_RATINGS) INTO A SPARSE MATRIX (SDF)
        sdf = df_ratings.astype(pd.SparseDtype("float", 0.0))
        print(sdf.shape)

        display(HTML(sdf.head(3).to_html()))

        print('dense : {:0.2f} bytes'.format(
            df_ratings.memory_usage().sum() / 1e3))
        print('sparse: {:0.2f} bytes'.format(sdf.memory_usage().sum() / 1e3))

        # return sdf, Iup, Iu_df
        return sdf
    #sdf2, Iup, Iu_df  = preprocessing(df)
    #sdf, Iup, Iu_df  = preprocessing(df)
    sdf = preprocessing(df)

    # sdf2.to_csv(r'sdf.csv')

    # 2. IMPORTING CENTROIDS
    # IMPORTING THE ALREADY EXISTING CENTROID SET
    centroids_105 = pd.read_csv('105_centroids.csv')
    #c_set = c_set.rename(columns = {'Unnamed: 0': 'userId'}, inplace = False)
    centroids_105.index = centroids_105.userId
    centroids_105 = centroids_105.drop(
        columns={'userId'}).astype(pd.SparseDtype("float", 0.0))

    
    
    # !!!!!!!!!!!!!!!!!!!!!!!! CHANGE THIS VALUE !!!!!!!!!!!!!!!!!!!!!!!!
    m = 100
    centroids_2 = centroids_105.iloc[:m, ]
    print('Centroids set:')
    display(HTML(centroids_2.head(m).to_html()))

    # 4. SELECTION OF THE BEST K -- ELBOW METHOD
    # SSE_k method
    # a) selection of the first set of centroids
    # b) perform the iterations (same as GSUdetector)
    # c) after the iterations, return the sum of squared errors of this last distances between users and their corresponding centroids
    def sse_k(df, users, centroids_df, k, itr):
        # def sse_k(df, users, centroids_df , k, itr, Iup, Iu_df):
        # def sse_k(df, users, k, itrw, Iup, c, Iu_df):
        previous_cluster = np.empty(0, dtype=float)

        # first centroid set selection
        #centroids_df = kmeansplusprobpower(df, users, k, c, Iup, Iu_df)
        # copy of the first centroid set
        first_centroids = centroids_df.copy(deep=True)
        print('len centroids_df: ', len(centroids_df))

        # iterations
        for i in range(itr):

            print('iteration number', i)
            #start1 = timeit.default_timer()
            # each iteration has empty arrays of clusters and distances!
            cluster = np.empty(0, dtype=float)
            #sim_users = np.empty(0, dtype=float)
            distance_users = np.empty(0, dtype=float)

            # empty aux dfs for every iteration
            df_cluster_i = pd.DataFrame()
            new_centers = pd.DataFrame()
            sdf2 = pd.DataFrame()
            cluster_df = pd.DataFrame()
            display(HTML(centroids_df.head(k).to_html()))

            # for every iteration, new centroids
            # loop for all users
            # for n in users.index:
            # process(n,k,users,centroids_dfdistance_users,cluster)
            results = Parallel(n_jobs=-1)(delayed(process)(n, k,
                                                           users.loc[n], centroids_df) for n in users.index)
            print(results)
            print('iteration number', i)
            res_df = pd.DataFrame(
                results, columns=['userId', 'cluster', 'distance'])
            res_df.index = res_df.userId
            res_df = res_df.drop(columns={'userId'})

            # get the distances into an array
            distance_column = res_df.loc[:, 'distance']
            distance_users = distance_column.values
            res_df = res_df.drop(columns={'distance'})
            # get the clusters into an array
            cluster_column = res_df.loc[:, 'cluster']
            cluster = cluster_column.values
            results = tuple()

            sdf2 = users.join(res_df).astype(pd.SparseDtype("float", 0.0))

            print('÷÷÷÷÷÷÷÷÷÷÷÷÷  cluster_df before and after deleting it ÷÷÷÷÷÷÷÷÷÷÷÷')

            # display(HTML(cluster_df.head(5).to_html()))
            cluster_df = pd.DataFrame()
            # display(HTML(cluster_df.head(5).to_html()))

            print('$$$$$$$$$$ sdf2 $$$$$$$$$$ ')

            # display(HTML(sdf2.head(5).to_html()))

            # Setting the new centers using the aux df of users
            for var in range(k):

                # df_cluster_i df of all users belonging to the cluster i
                df_cluster_i = pd.DataFrame()
                # df_cluster_i contains the dataframe of all users belonging to the cluster 'var'
                df_cluster_i = sdf2.loc[sdf2['cluster'] == var].astype(
                    pd.SparseDtype("float", 0.0))
                print('users from cluster ', var)
                           

                # deleting redundant columns: cluster
                df_cluster_i = df_cluster_i.drop(columns={'cluster'})
                #df_cluster_i = df_cluster_i.drop(columns={'movies rated','cluster', 'distance'})
                

                print('***** iteration finished, number:', i)

                # new_centers: mean of each cluster
                new_centers = new_centers.append(
                    df_cluster_i.mean(), ignore_index=True).astype(pd.SparseDtype("float", 0.0))
            # empty the centroids_df to add the new ones
            #centroids_df = centroids_df.iloc[0:0]
            # set the new centroids_df
            centroids_df = new_centers

            print('####### iteration finished, number:', i)
            # TO SEE IF IT IS NECESSARY TO KEEP GOING ON WITH THE ITERATIONS
            flag = 1
            # for the first iteration, assignation of the previous_cluster aux variable
            if (i == 0):
                previous_cluster = cluster
                print('previous_cluster', previous_cluster)

            # for the resting iterations, comparison between actual cluster and previous one

            else:
                flag = 0
                for u in range(len(cluster)):
                    if cluster[u] != previous_cluster[u]:
                        flag = 1
                        print(' previous cluster != cluster -> CONTINUE ITERATING')
            # when they are the same, no more iterations
            if (flag == 0):
                print('flag = 0!!!!! iteration finished, number:', i)
                break

            previous_cluster = cluster
            cluster = 0
            print('iteration finished, number:', i)
        # for each number of k, i calculates the summatory of (distance users)^2
        ssd = np.sum(distance_users ** 2)
        dist_df = pd.DataFrame({'distance': distance_users})
        dist_df.index = sdf2.index
        sdf2 = sdf2.join(dist_df).astype(pd.SparseDtype("float", 0.0))

        return ssd, centroids_df, sdf2

    ssd, centroids_df, sdf2 = sse_k(df, sdf, centroids_2, m, 10)
    centroids_df.to_csv("100_centroids_10itr.csv")
    sdf2.to_csv("u_clusters_100centroids_10itr.csv")
    print(ssd)
    sum_sq = pd.DataFrame()
    sum_sq['ssd'] = [ssd]
    sum_sq['centroids'] = [m]
    all = pd.concat([all_ssd, sum_sq], ignore_index=True)

    all.to_csv("all_ssd.csv")

