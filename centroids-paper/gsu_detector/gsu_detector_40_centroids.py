
# 1. PREPROCESSING
# Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
from statsmodels.stats.weightstats import ttest_ind
from IPython.display import display, HTML
from joblib import Parallel, delayed
from scipy.stats import pearsonr

import timeit


def process(n, k, user_row, centroids_df):
    print('  STARTED user number: ', n)
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

    print('  ENDED user number: ', n)
    return (n, cluster_j, distance_ui_c, max_sim_ui_c)
    #print('  time2',timeit.default_timer()-start2)


if __name__ == "__main__":
    # importing the required csv files
    df = pd.read_csv('MovieLens.csv')

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

        return sdf

    sdf = preprocessing(df)

    # 2. IMPORTING 10 CENTROIDS
    # IMPORTING THE ALREADY EXISTING CENTROID SET
    centroids_100 = pd.read_csv('40_centroids_10_iterations.csv')
    centroids_100.index = centroids_100['Unnamed: 0']
    centroids_100 = centroids_100.drop(columns={'Unnamed: 0'})
    display(HTML(centroids_100.head(3).to_html()))

    def similarity_measure(users, centroids_df, k):
        results = Parallel(n_jobs=-1)(delayed(process)(n, k,
                                                       users.loc[n], centroids_df) for n in users.index)
        print(results)
        #print('iteration number', i)
        res_df = pd.DataFrame(
            results, columns=['userId', 'cluster', 'distance', 'similarity'])
        # res_df.sort_values(by=['userId'])
        res_df.index = res_df.userId
        res_df = res_df.drop(columns={'userId'})

        # get the distances into an array
        distance_column = res_df.loc[:, 'distance']
        distance_users = distance_column.values
        res_df = res_df.drop(columns={'distance'})

       # get the clusters into an array
        cluster_column = res_df.loc[:, 'cluster']
        cluster = cluster_column.values

        # introducing SIMILARITY and CLUSTER columns into the sdf
        sdf2 = users.join(res_df).astype(pd.SparseDtype("float", 0.0))
        return sdf2

    sdf_cluster_sim = similarity_measure(sdf, centroids_100, 40)

    sdf_cluster_sim.to_csv("sdf_cluster_sim_100centroids.csv")

    print('RESULTING SDF WITH CLUSTERS AND SIMILARITY:')
    display(HTML(sdf_cluster_sim.head(3).to_html()))
    # print('Mean',sdf_cluster_sim.similarity.Mean())
    # print('max',sdf_cluster_sim.similarity.max())
    # print('min',sdf_cluster_sim.similarity.min())

    df = pd.read_csv('final_errors.csv')
    df.index = df.userId
    display(HTML(df.head(3).to_html()))
    arr_w = np.empty(0, dtype=int)
    arr_mae_gsu = np.empty(0, dtype=int)
    arr_mae_non_gsu = np.empty(0, dtype=int)
    arr_r_sk = np.empty(0, dtype=int)
    arr_p_value_gsu_ngsu = np.empty(0, dtype=int)
    arr_p_value_gsu_rsk = np.empty(0, dtype=int)
    arr_p_value_ngsu_rsk = np.empty(0, dtype=int)
    arr_n_gsu = np.empty(0, dtype=int)

    def gsu_detector(df, sdf_matrix, w, arr_p_value_gsu_ngsu, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_w):

        non_gsu_df = df.copy(deep=True)
        pot_gsu_df = pd.DataFrame()

        for i in sdf_matrix.index:
            if (sdf_matrix['similarity'].loc[i] < w):
                id_i = df['userId'] == i
                non_gsu_df = non_gsu_df[non_gsu_df.userId != i]
                # display(HTML(non_gsu_df.head(3).to_html()))
                df_id_i = df[id_i]
                pot_gsu_df = pd.concat([pot_gsu_df, df_id_i], axis=0)
                # display(HTML(non_gsu_df.head(3).to_html()))
        pot_gsu_df.index = pot_gsu_df.userId
        index_gsu = pot_gsu_df.index
        list_id_gsu = index_gsu.tolist()
        print('id gsu list: ', list_id_gsu)
        print('len gsu', len(list_id_gsu))

        # getting the MAE from all_users, non-gsu, gsu
        non_gsu_mae = non_gsu_df.error.mean()
        gsu_mae = pot_gsu_df.error.mean()

        tt = ttest_ind(non_gsu_df.error, pot_gsu_df.error)
        arr_p_value_gsu_ngsu = np.append(arr_p_value_gsu_ngsu, tt[1])
        arr_mae_gsu = np.append(arr_mae_gsu, gsu_mae)
        arr_mae_non_gsu = np.append(arr_mae_non_gsu, non_gsu_mae)
        arr_w = np.append(arr_w, w)
        arr_n_gsu = np.append(arr_n_gsu, len(list_id_gsu))

        return arr_p_value_gsu_ngsu, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_w

    for w in np.arange(1.2, 1.99, 0.1):
        arr_p_value_gsu_ngsu, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_w = gsu_detector(
            df, sdf_cluster_sim, w, arr_p_value_gsu_ngsu, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_w)

    # list_id_gsu.to_csv("paper2_list_id_gsu.csv")

    results = pd.DataFrame()

    results["W"] = arr_w
    results["MAE-GSU"] = arr_mae_gsu
    results["MAE-NON-GSU"] = arr_mae_non_gsu
    results["P-Value GSU-NGSU"] = arr_p_value_gsu_ngsu
    results["# GSU:"] = arr_n_gsu

    results.index = results.W
    results = results.drop(columns={'W'})
    display(HTML(results.head(40).to_html()))
    results.to_csv("40_centroids_maes_results.csv")
