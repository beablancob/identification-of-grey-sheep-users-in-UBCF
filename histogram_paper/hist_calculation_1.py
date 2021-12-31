# Import Python Libraries
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib as mpl
import seaborn as sns
from IPython.display import display, HTML

# IMPORTING THE ALREADY EXISTING SIMILARITY MATRIX
sim_matrix = pd.read_csv('sim_matrix_wholematrix.csv')
sim_matrix.index = sim_matrix.userId
sim_matrix = sim_matrix.drop(columns={'userId'})

# SIMILARITY DISTRIBUTION BY HISTOGRAM INTERSECTION

# UPDATING THE SIMILARITY MATRIX


def return_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection


def histograms(user_vector1, user_vector2):
    hist_1 = 0
    hist_2 = 0
    hist_1, _ = np.histogram(user_vector1, bins=100, range=[
        user_vector1.min(), user_vector1.max()])
    hist_2, _ = np.histogram(user_vector2, bins=100, range=[
        user_vector2.min(), user_vector2.max()])
    return hist_1, hist_2


sim_matrix2 = sim_matrix.copy(deep=True)
visited = {}
tuple_ids = tuple()
for i in range(len(sim_matrix2)):
    for j in range(len(sim_matrix2)):
        arr_ids = [sim_matrix.iloc[i].name, sim_matrix.iloc[j].name]
        arr_ids.sort()
        
        tuple_ids = tuple(arr_ids)
        if tuple_ids in visited:
            sim_matrix2.iloc[[i], [j]] = visited[tuple_ids]
            continue

        h1, h2 = histograms(sim_matrix.iloc[i], sim_matrix.iloc[j])
        sim = return_intersection(h1, h2)
        sim_matrix2.iloc[[i], [j]] = sim
        visited[tuple_ids] = sim
print(sim_matrix2.head(10))

sim_matrix2.to_csv('histogram_intersection_sim_matrix.csv')
