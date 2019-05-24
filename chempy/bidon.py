# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 14:50:09 2019

@author: DOMI
"""
import scipy
import scipy.cluster.hierarchy as sch
import matplotlib.pylab as plt

n=10
k=3
X = scipy.randn(n,2)
d = sch.distance.pdist(X)
Z= sch.linkage(d,method='complete')
T = sch.fcluster(Z, k, 'maxclust')

# calculate labels
labels=list('' for i in range(n))
for i in range(n):
    labels[i]=str(i)+ ',' + str(T[i])

# calculate color threshold
ct=Z[-(k-1),2]  

#plot
P =sch.dendrogram(Z,labels=labels,color_threshold=ct)
plt.show()
