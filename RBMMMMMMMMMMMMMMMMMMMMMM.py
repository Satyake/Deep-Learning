#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 19:54:41 2018

@author: satyake
"""

import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
train=pd.read_csv('/home/satyake/Downloads/ml-100k/u3.base',sep='\t')
train=np.array(train)
test=pd.read_csv('/home/satyake/Downloads/ml-100k/u3.test',sep='\t')
test=np.array(test)

nb_users=int(max(max(train[:,[0]]),max(test[:,[0]])))
nb_movies=int(max(max(train[:,[1]]),max(test[:,[1]])))

def convert(dataset):
    new_matrix=[]
    for i in range(0,nb_users):
     movies=dataset[:,[1]][dataset[:,[0]]==i]
     ratings_id=dataset[:,[2]][dataset[:,[0]]==i]
     ratings=np.zeros(nb_movies)
     ratings[movies-1]=ratings_id
     new_matrix.append(ratings)
    return new_matrix
train=convert(train)
test=convert(test)
train=torch.FloatTensor(train)
test=torch.FloatTensor(test)
train[train==0]=-1
train[train==1]=0
train[train>=2]=1
test[test==0]=-1
test[test==1]=0
test[test>=2]=1

class RBM():
    def __init__(self,nv,nh):
        self.W=torch.randn(nv,nh)
        self.a=torch.randn(1,nv)
        self.b=torch.randn(1,nh)
    def sampler_h(self,x):
        wx=torch.mm(self.W,x)
        activation=wx+self.b.expand_as(wx)
        p_h_given_v=torch.sigmoid(activation)
        return p_h_given_v,torch.bernoulli(p_h_given_v)
    def sampler_v(self,y):
        wy=torch.mm(self.W.t(),y)
        activation=wx+self.a.expand_as(wy)
        p_v_given_h=torch.sigmoid(activation)
        return p_v_given_h,torch.bernoulli(p_v_given_h)
    def gibbs(v0,vk,ph0,phk):
        self.W=self.W+(torch.mm(v0.t(),ph0)-torch.mm(vk.t(),phk)).t()
        self.a=self.a+torch.sum(v0-vk)
        self.b=self.b+torch.sum(ph0-phk)
    
nv=len(train[0])
nh=200
batch_size=100
epochs=10
rbm=RBM(nv,nh)

for e in range(0,epochs):
  c=0
  TR_L=0
  for j in range(nb_users,nb_users-batch_size,batch_size):
     v0=train[j:j+batch_size]
     vk=train[j:j+batch_size]
     ph0,_=rbm.sampler_h(v0)
     for k in range(0,10):
         _,hk=rbm.sampler_h(vk)
         _,vk=rbm.sampler_v(hk)
         vk[v0<0]=v0[v0<0]
     phk,_=rbm.sampler_h(vk)
     rbm.gibbs(v0,vk,ph0,phk)
     TR_L=torch.mean(torch.abs(v0-vk))
     c=c+1
  print('epoch is {}'.format(str(e)))
  print('loss {}'.format(str(TR_L)))
    