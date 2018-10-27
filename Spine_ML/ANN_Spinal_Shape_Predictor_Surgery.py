import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
dataset=pd.read_csv('Dataset_spine.csv')
x=dataset.iloc[:,0:12].values
y=dataset.iloc[:,12:13].values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2, random_state=0)
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Dropout
classifier=Sequential()
classifier.add(Dense(8,kernel_initializer='uniform',activation='relu',input_dim=12))
classifier.add(Dense(8,kernel_initializer='uniform',activation='relu'))
classifier.add(Dropout(0.2))
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))
classifier.compile(optimizer='rmsprop', metrics=['accuracy'],loss='binary_crossentropy')
classifier.fit(x_train,y_train,nb_epoch=200, batch_size=13)

result=classifier.predict(x_test)
#unknown data
result_UK=classifier.predict(sc.transform(np.array([[67,22,37,67,100,-0.9,0.75,12,15,12,20,-20]])))
