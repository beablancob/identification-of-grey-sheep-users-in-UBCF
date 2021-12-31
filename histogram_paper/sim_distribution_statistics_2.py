# Import Python Libraries
import numpy as np
import pandas as pd
from IPython.display import display, HTML


# IMPORTING THE ALREADY EXISTING SIMILARITY MATRIX
sim_matrix = pd.read_csv('histogram_intersection_sim_matrix.csv')
sim_matrix.index = sim_matrix.userId
sim_matrix = sim_matrix.drop(columns={'userId'})
print('imported sim_matrix')
display(HTML(sim_matrix.head(7).to_html()))


# PAPER 8. STEP 1.

# REPRESENTATION OF EACH USER BY THE DISTRIBUTION OF USER-USER SIMILARITY
u_mean = np.empty(0, dtype=float)
user_ids = np.empty(0, dtype=int)
q1 = np.empty(0, dtype=float)
q2 = np.empty(0, dtype=float)
q3 = np.empty(0, dtype=float)
std = np.empty(0, dtype=float)
skewness = np.empty(0, dtype=float)

sim_matrix2 = sim_matrix.copy(deep=True)
# through the rows, keeping the important values of each user
for i in range(0, len(sim_matrix2)):

    user_ids = np.append(user_ids, sim_matrix2.index[i])
    q1 = np.append(q1, sim_matrix2.iloc[i].quantile(q=0.25))
    q2 = np.append(q2, sim_matrix2.iloc[i].quantile(q=0.5))
    q3 = np.append(q3, sim_matrix2.iloc[i].quantile(q=0.75))
    std = np.append(std, sim_matrix2.iloc[i].std(axis=0))
    skewness = np.append(skewness, sim_matrix2.iloc[i].skew())
    u_mean = np.append(u_mean, sim_matrix2.iloc[i].mean())


sim_distr = pd.DataFrame({'User ID': user_ids, 'q1': q1, 'q2': q2,
                         'q3': q3, 'Mean': u_mean, 'STD': std, 'Skewness': skewness})

display(HTML(sim_distr.head(5).to_html()))

print('dense : {:0.2f} bytes'.format(sim_distr.memory_usage().sum() / 1e3))
sim_distr.to_csv('distr_stats_final.csv')
