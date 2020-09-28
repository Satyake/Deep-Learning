# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 17:39:39 2020

@author: satya
"""

import pandas as pd

import scipy.cluster.hierarchy as sch
from sklearn.cluster import DBSCAN
data=pd.read_csv('cars_clus.csv')

featureset = data[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]
featureset=featureset.dropna()
featureset=featureset.replace('$null$',0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
featureset=sc.fit_transform(featureset)


from sklearn.cluster import AgglomerativeClustering
dendogram=sch.dendrogram(sch.linkage(featureset,method='ward'))
plt.show()
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y=hc.fit_predict(featureset)

df=DBSCAN(eps=0.3,min_samples=2)
y=df.fit(featureset)

y=y.labels_

sample_cores=np.zeros_like(y)
sample_cores[df.core_sample_indices_]=True

np.unique(y)