# Import Python Libraries
from statsmodels.stats.weightstats import ttest_ind
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
import numpy as np

import pandas as pd

from IPython.display import display, HTML
# # PAPER 8. STEP 2.

# importing the already existing distribution statistics, if already existing
sim_distr = pd.read_csv('distr_stats_final.csv')
sim_distr = sim_distr.drop(columns={'Unnamed: 0'})

############ NORMALIZATION OF THE DESCRIPTIVE STATISTICS ############


sim_norm = sim_distr.copy(deep=True)
sim_user_id = sim_norm['User ID']
sim_norm = sim_norm.drop(columns={'User ID'})

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
# get column names
cols_numeric = sim_norm.select_dtypes(include=numerics).columns.tolist()


# Normalization method 1
scaler = MinMaxScaler()
sim_norm[cols_numeric] = scaler.fit_transform(sim_norm[cols_numeric])
display(HTML(sim_norm.head(10).to_html()))

############# PAPER 8. STEP 2. #############
# FILTERING PART

# q1 mean > q1 -> user deleted
# q3 skewness < q3 -> user deleted
n_users_dropped = 0
sim_distr_filtered = sim_norm.copy(deep=True)

for i in sim_norm.index:
    if (sim_norm.Mean.quantile(q=0.25) > sim_norm.loc[i].Mean) or (sim_norm.loc[i].Skewness > sim_norm.Skewness.quantile(q=0.75)):
        n_users_dropped += 1
        sim_distr_filtered = sim_distr_filtered.drop(index=i)


print('number of users dropped:', n_users_dropped)


print('Final number of users', sim_distr_filtered.shape)
display(HTML(sim_distr_filtered.head(3).to_html()))


############# PAPER 8. STEP 3. #############

# OUTLIER DETECTION
clf = LocalOutlierFactor(n_neighbors=140)
clf.fit(sim_distr_filtered)
array = clf.negative_outlier_factor_

df_nof = pd.DataFrame(array, columns=['NOF'])
sim_distr_filtered = sim_distr_filtered.join(df_nof)

# LET'S SEE FIRST THE DISTRIBUTION OF THE LOF
distr = sim_distr_filtered['NOF']
distr.plot(kind="hist")


sim_distr_filtered['userId'] = sim_user_id
sim_distr_filtered.index = sim_distr_filtered['userId']

print("Descriptive statistics normalized:")
display(HTML(sim_distr_filtered.head(3).to_html()))


# NOW, WE ARE GOING TO TRY DIFFERENT VALUES
df = pd.read_csv('final_errors.csv')
df.index = df.userId


###### RESULTS INTO A CSV ######
results = pd.DataFrame()
# 7 columns: LOF, MAE GSU, MAE NON-GSU,MAE RS, P-VALUE GSU-NGSU, P-VALUE GSU-RSK, P-VALUE NGSU-RSK, # GSU
arr_lof = np.empty(0, dtype=int)
arr_mae_gsu = np.empty(0, dtype=int)
arr_mae_non_gsu = np.empty(0, dtype=int)
arr_r_sk = np.empty(0, dtype=int)
arr_p_value_gsu_ngsu = np.empty(0, dtype=int)
arr_p_value_gsu_rsk = np.empty(0, dtype=int)
arr_p_value_ngsu_rsk = np.empty(0, dtype=int)
arr_n_gsu = np.empty(0, dtype=int)


def lof_users(lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu):

    list_id_gsu = np.empty(0, dtype=int)
    # filtering users
    for i in sim_distr_filtered.index:
        if (sim_distr_filtered.NOF[i] < lof_threshold):
            id_gsu = sim_distr_filtered['userId'][i]
            list_id_gsu = np.append(list_id_gsu, id_gsu)

    non_gsu_df = df.copy(deep=True)
    filtered_sim_df = pd.DataFrame()
    filtered_df = pd.DataFrame()

    for i in list_id_gsu:

        id_i = df['userId'] == i
        non_gsu_df = non_gsu_df[non_gsu_df.userId != i]
        df_id_i = df[id_i]

        filtered_df = pd.concat([filtered_df, df_id_i], axis=0)
        id_i = sim_distr_filtered['userId'] == i
        sim_id_i = sim_distr_filtered[id_i]
        filtered_sim_df = pd.concat([filtered_sim_df, sim_id_i], axis=0)

    filtered_df.index = filtered_df.userId
    filtered_df = filtered_df.drop(columns={'userId'})
    print('Filtered matrix')
    print(filtered_df.head(5))
    print('filtered matrix, mean error: ', filtered_df.error.mean())

    non_gsu_df.index = non_gsu_df.userId
    non_gsu_df = non_gsu_df.drop(columns={'userId'})
    non_gsu_mae = non_gsu_df.error.mean()
    gsu_mae = filtered_df.error.mean()

    tt = ttest_ind(non_gsu_df.error, filtered_df.error)

    arr_p_value_gsu_ngsu = np.append(arr_p_value_gsu_ngsu, tt[1])
    arr_n_gsu = np.append(arr_n_gsu, len(list_id_gsu))
    arr_lof = np.append(arr_lof, lof_threshold)
    arr_mae_gsu = np.append(arr_mae_gsu, gsu_mae)
    arr_mae_non_gsu = np.append(arr_mae_non_gsu, non_gsu_mae)

    return arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu


lof_threshold = -1
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)

lof_threshold = -1.05
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)

lof_threshold = -1.1
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)

lof_threshold = -1.15
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)


lof_threshold = -1.2
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)


lof_threshold = -1.3
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)

lof_threshold = -1.4
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)


lof_threshold = -1.5
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)


'''
lof_threshold = -1.6
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)

lof_threshold = -1.7
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)


lof_threshold = -1.8
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)

lof_threshold = -1.9
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)

lof_threshold = -2
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)


lof_threshold = -2.05
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)


lof_threshold = -2.15
arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu = lof_users(
    lof_threshold, sim_distr_filtered, df, arr_p_value_gsu_ngsu, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu)


lof_threshold = -2.1
list_id_gsu, filtered_df, filtered_sim_df = lof_users(
    lof_threshold, sim_norm, df)
arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk = filtering_users(
    df, filtered_df,  arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk)

lof_threshold = -2.2
list_id_gsu, filtered_df, filtered_sim_df = lof_users(
    lof_threshold, sim_norm, df)
arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk = filtering_users(
    df, filtered_df,  arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk)

lof_threshold = -2.3
list_id_gsu, filtered_df, filtered_sim_df = lof_users(
    lof_threshold, sim_norm, df)
arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk = filtering_users(
    df, filtered_df,  arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk)

lof_threshold = -2.4
list_id_gsu, filtered_df, filtered_sim_df = lof_users(
    lof_threshold, sim_norm, df)
arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk = filtering_users(
    df, filtered_df,  arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk)


lof_threshold = -2.5
list_id_gsu, filtered_df, filtered_sim_df = lof_users(
    lof_threshold, sim_norm, df)
arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk = filtering_users(
    df, filtered_df,  arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk)

lof_threshold = -2.6
list_id_gsu, filtered_df, filtered_sim_df = lof_users(
    lof_threshold, sim_norm, df)
arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk = filtering_users(
    df, filtered_df,  arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk)

lof_threshold = -2.7
list_id_gsu, filtered_df, filtered_sim_df = lof_users(
    lof_threshold, sim_norm, df)
arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk = filtering_users(
    df, filtered_df,  arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk)

lof_threshold = -2.8
list_id_gsu, filtered_df, filtered_sim_df = lof_users(
    lof_threshold, sim_norm, df)
arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk = filtering_users(
    df, filtered_df,  arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk)



lof_threshold = -2.9
list_id_gsu, filtered_df, filtered_sim_df = lof_users(
    lof_threshold, sim_norm, df)
arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk = filtering_users(
    df, filtered_df,  arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk)
    
lof_threshold = -3
list_id_gsu, filtered_df, filtered_sim_df = lof_users(
    lof_threshold, sim_norm, df)
arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk = filtering_users(
    df, filtered_df,  arr_p_value_gsu_ngsu, arr_p_value_ngsu_rsk, arr_p_value_gsu_rsk, arr_lof, arr_mae_gsu, arr_mae_non_gsu, arr_n_gsu, arr_r_sk)



'''

############## EXPORTING THE RESULTS ##############
results["LOF"] = arr_lof
results["MAE-GSU"] = arr_mae_gsu
results["MAE-NON-GSU"] = arr_mae_non_gsu

results["P-Value GSU-NGSU"] = arr_p_value_gsu_ngsu

results["# GSU:"] = arr_n_gsu


results.index = results.LOF
results = results.drop(columns={'LOF'})
display(HTML(results.head(40).to_html()))
results.to_csv("paper8_q1_mean_q3skew_k140_.csv")
