# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:41:52 2018

@author: user
"""

import pandas as pd 
import matplotlib.pyplot as plt 
dataset=pd.read_excel("AirQualityUCI.xlsx")
dataset_predictor=dataset.iloc[:,5:6].values
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
dataset_predictor_scaled=scaler.fit_transform(dataset_predictor)
#use 10 samples as training set at nth sec 
x_train=[]
y_train=[]
for i in range(10,9335):
    x_train.append(dataset_predictor_scaled[i-10:i,0])
    y_train.append(dataset_predictor_scaled[i,0])
import numpy as np 
x_train=np.array(x_train)
y_train=np.array(y_train)
x_train=np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))
import keras
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import Sequential

regressor=Sequential()

regressor.add(LSTM(units=90,return_sequences=True, input_shape=(x_train.shape[1],1)))
regressor.add(Dropout(0.1))

regressor.add(LSTM(units=90,return_sequences=False ))
regressor.add(Dropout(0.1))

regressor.add(Dense(1))
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.fit(x_train,y_train, batch_size=20, nb_epoch=30)

dataset_test=pd.read_excel('AirQualityUCI_test.xlsx')
dataset_test_predictor_real=dataset_test.iloc[:,0:1].values
#concatenating the datasets 
total_dataset=pd.concat((dataset['predictor'],dataset_test['predictor']),axis=0)
test_set_input=total_dataset[len(total_dataset)-len(dataset_test)-10:].values
test_set_input=test_set_input.reshape(-1,1)
test_set_input=scaler.transform(test_set_input)

