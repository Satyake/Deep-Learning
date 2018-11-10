import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
dataset=pd.read_csv('Dataset_spine.csv')
#x=dataset.iloc[:,0:12].values
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
#from sklearn.preprocessing import StandardScaler
#sc=StandardScaler()
#from sklearn.model_selection import train_test_split
#x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=0)
#x_train=sc.fit_transform(x_train)
#x_test=sc.fit_transform(x_test)
#import keras
#from keras.layers import Dense
#from keras.models import Sequential
#from keras.layers import Dropout
#from sklearn.model_selection import cross_val_score
#def ANN():
 #  classifier=Sequential()
  # classifier.add(Dense(8,kernel_initializer='uniform',activation='relu',input_dim=12))
   #classifier.add(Dense(8,kernel_initializer='uniform',activation='relu'))
   #classifier.add(Dropout(0.2))
   #classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
   #classifier.compile(optimizer='adam', metrics=['accuracy'],loss='binary_crossentropy')
   #return classifier
#from keras.wrappers.scikit_learn import KerasClassifier
#from sklearn.model_selection import GridSearchCV
#Classifier_1=KerasClassifier(build_fn=ANN,nb_epoch=20, batch_size=10)
#params={'optimizer_1':['adam','rmsprop'],'batch_size':[2,3,4],'nb_epoch':[10,20,100]}
#gs=GridSearchCV(estimator=Classifier_1, param_grid=params,cv=10, scoring='accuracy')
#gs.fit(x_train,y_train)
#cvc=cross_val_score(estimator=Classifier_1,X=x_train,y=y_train, cv=10, scoring='accuracy')
#cvc.mean()
#cvc.std()
#result=classifier.predict(x_test)
#unknown data
#result_UK=classifier.predict(sc.transform(np.array([[67,22,37,67,100,-0.9,0.75,12,15,12,20,-20]])))

import minisom
from sklearn.preprocessing import MinMaxScaler
sc_norm=MinMaxScaler()
x=sc_norm.fit_transform(x)
from minisom import MiniSom
SOM=MiniSom(x=10,y=10, input_len=12, decay_function=None, sigma=1, learning_rate=0.5)
SOM.random_weights_init(x)
SOM.train_random(x,num_iteration=10)
from pylab import pcolor, bone,show,colorbar, plot
bone()
pcolor(SOM.distance_map().T)
colorbar()

color=['g','r']
marker=['o','s']
for i , x in enumerate(x):
    W=SOM.winner(x)
    plot(W[0]+0.5,
         W[1]+0.5,
         marker[y[i]],
         markeredgecolor= color[y[i]],
         markerfacecolor='None')
show()
mappings=SOM.win_map(x)
