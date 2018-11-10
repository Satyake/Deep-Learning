import numpy as np 
import pandas as pd
dataset=pd.read_csv('Dataset_Spine.csv')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1 ].values
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
X=scaler.fit_transform(X)
import minisom
from minisom import MiniSom
SOM=MiniSom(x=10,y=10, input_len=12,decay_function=None,sigma=1, learning_rate=0.6)
SOM.random_weights_init(X)
SOM.train_random(X, 100)
from pylab import bone, colorbar,pcolor, show,plot
bone()
pcolor(SOM.distance_map().T)
colorbar()
color=['g','b']
marker=['s','o']
for i,x in enumerate(X):
    w=SOM.winner(x)
    plot(w[0]+0.5,w[1]+0.5,marker[Y[i]], markeredgecolor=color[Y[i]], markerfacecolor=None )
show()
mapper=SOM.win_map(X)
possible_outliers=mapper[(3,1)]
possible_outliers=scaler.inverse_transform(possible_outliers)