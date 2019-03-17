# -*- coding: utf-8 -*-
"""
KMeans method   and Hierarchical Clustering

@author: faramarz.faghihi
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
iris = load_iris()

iris.data
kmn = KMeans(n_clusters=3)
kmn.fit(iris.data)
labels = kmn.predict(iris.data)
labels

xs = iris.data[:,0]
ys = iris.data[:,2]
centroids = kmn.cluster_centers_
fig = plt.figure() 
plt.scatter(xs, ys, c=labels)
plt.scatter(centroids[:,0],centroids[:,2],marker='x',s=150,alpha=0.5)
plt.show()

print(kmn.inertia_)

inertia_list = []
for k in np.arange(1, 6):
 kmn = KMeans(n_clusters=k)
 kmn.fit(iris.data)
 inertia_list.append(kmn.inertia_)
 inertia_list
 
fig = plt.figure() 
plt.plot(np.arange(1,6),inertia_list,'ro-')
plt.xlabel('number of clusters')
plt.ylabel('Inertia')
plt.show()


###  Hierarchical Clustering ########


from scipy.cluster.hierarchy import linkage,dendrogram,fcluster
import matplotlib.pyplot as plt
hir = linkage(iris.data,method='complete')
labels = fcluster(hir, 6, criterion='distance')
print(labels)
dendrogram(hir,leaf_rotation=90)
plt.show()
fig = plt.figure() 
plt.scatter(iris.data[:,0], iris.data[:,2], c=labels)
plt.show()

from sklearn.cluster import MeanShift

x = iris.data
ms = MeanShift()
ms.fit(x)
labels = ms.labels_
cluster_center = ms.cluster_centers_
n_cluster = len(np.unique(labels))
print('Number of estimated cluster:' ,n_cluster)
fig = plt.figure()
plt.scatter(x[:,0], x[:,1], c=labels)
plt.scatter(cluster_center[:,0], cluster_center[:,1], marker='x', s=150, linewidth=5, zorder=10)
plt.show()

from sklearn.cluster import DBSCAN
dbscan = DBSCAN()
dbscan.fit(iris.data)
labels = dbscan.labels_
xx = iris.data[:,0]
yy = iris.data[:,1]
fig = plt.figure()
plt.scatter(xx,yy, c=labels)
plt.show()
