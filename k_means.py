#!/usr/bin/env python3

# k_means.py

import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
import mglearn

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# See: https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_iris.html#sphx-glr-auto-examples-cluster-plot-cluster-iris-py
# ``Though the following import is not directly being used, it is required for 3D projection to work with matplotlib < 3.2''
import mpl_toolkits.mplot3d

df = pd.read_csv("k_means_data.csv", sep=",")
df.head(2)

df2 = df.loc[:,['gp', 'wins', 'losses', 'fgm', 'stl', 'tov', 'blk', 'pts']]
df2.head(2)

# initialize object
scaler = MinMaxScaler()

# fit the scaler. In this case, I think we treat all data == X == df2
scaler.fit(df2)

print(" Min from data: ", scaler.data_min_)
print(" Max from data: ", scaler.data_max_)

# Call transform method, to actually perform the scaling, store it in a variable
trans_data = scaler.transform(df2)

# note there is no K argument to the Kmeans function. We assume K == n_clusters
# this will give us only 2 clusters, 0 and 1. 
# n_init is the "Number of times the k-means algorithm is run with different centroid seeds"
km = KMeans(n_clusters=2, n_init=10).fit(trans_data)

# the data points are already assigned to a cluster from the call to KMeans().fit above
print(km.labels_)

# add these to the dataframe, df2
df2['cluster'] = km.labels_

df2.head(2)

# initialize figure object
fig = plt.figure()

# make fig 3d
ax = fig.add_subplot(projection = '3d')

#label axes
ax.set_xlabel("wins") 
ax.set_ylabel("losses")
ax.set_zlabel("pts")

# call the scatter method
ax.scatter(df2['wins'], df2['losses'], df2['pts'], c=km.labels_, cmap='plasma')


print("\n")
print("Theory questions & answers")
print("\n")
print("1. Explain the K-means clustering algorithm in your own words. ")
print("\n")
print("The K-mean alogrithm tries to group data points into clusters while minimizing a measure called inertia. The algorithm initializes a set of centroids then assigns each datapoint to a centroid. Then the alogorithm moves the centroid interatively by getting the average distance of each assigned sample and trying to minimize the variance. basically, it tries to place the centroid so all assigned samples are equidistant from it. ")
print("\n")
print("What are some of the limitations of K-means clustering?")
print("\n")
print("While K-means will always converge, it may get stuck in local minima. It is also highly sensitive to the inital placement of the centroids. It does not perform well on irregularly shaped data. It also assumes all data are 'convex and isotropic' which basically means the algorithm assumes rounded and uniform surfaces is the data-space. I person feel the idea of moving the centroid is sort of 'meh...'' but moving some n centroid points is definitely more efficient than moving some x datapoints to the centeroids. ")
print("\n")
print("What is the role of random initialization in K-means, and how can it affect the results? ")
print("\n")
print("The role of random initialization of the centroids is to simply distribute them amongst the datapoints to be clustered. The Scikit-learn documentation states 'Given enough time, K-means will always converge, however this may be to a local minimum. This is highly dependent on the initialization of the centroids.  As a result, the computation is often done several times, with different initializations of the centroids.'")

print("\n")
print("Why is it necessary to standardize the data before applying K-means clustering?")
print("\n")
print("Standardization/normalization of the data is required by the fundamental mechanism of the alogorithm: it calculates averages while trying to minimize variance. These calculations would be meaningless if the data were in a non-uniform scale.")



# References
# [1] https://scikit-learn.org/stable/modules/clustering.html#k-means

# [2] https://www.markdownguide.org/basic-syntax/#blockquotes-1

# [3] https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html



