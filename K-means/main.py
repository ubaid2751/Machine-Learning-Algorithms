import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns 
from Kmeans import *

df = pd.read_csv(r"E:\Implementation Of Algos\data\dataset.csv")

# print(df.head(5))

# print(df.columns)
# print(df.info())

X_train = df.iloc[:, [3, 4]].values
# print(X_train)

X_train = normalize(X_train)

model = K_means()
K = 5
initial_centroids = model.kmeans_init_centroids(X_train, k=K)
centroids, idx  = model.run_kmeans(X_train, K, centroids=initial_centroids, max_iters=10, plot_progress=True)
print(centroids)