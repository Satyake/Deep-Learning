import numpy as np 
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
train=pd.read_csv('/home/satyake/Downloads/ml-100k/u2.base',sep='\t')
test=pd.read_csv('/home/satyake/Downloads/ml-100k/u2.test',sep='\t')
train=np.array(train)
test=np.array(test)

nb_users=int(max(max(train[:,0]),max(test[:,0])))
nb_movies=int(max(max(train[:,1]),max(test[:,1])))

def convert(data):
  new_list=[]
  for id_user in range(0,nb_users):
    id_movies=data[:,1][data[:,0]==id_user]
    id_ratings=data[:,2][data[:,0]==id_user]
    ratings=np.zeros(nb_movies)
    ratings[id_movies-1]=id_ratings
    new_list.append(ratings)
  return new_list
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
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self, v0, vk, ph0, phk):
        self.W = self.W+(torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)).t()
        self.b =self.b+torch.sum((v0 - vk))
        self.a =self.a+torch.sum((ph0 - phk))
        
nv=len(train[0])
nh=400
batch_size=120
nb_epochs =5
rbm=RBM(nv,nh)

for epoch in range(0,nb_epochs):
    s=0
    train_loss=0
    for j in range(0,nb_users-batch_size,batch_size):
        v0=train[j:j+batch_size]
        vk=train[j:j+batch_size]
        ph0,_=rbm.sample_h(v0)
        
        for k in range(0,10):
            _,hk=rbm.sample_h(vk)
            _,vk=rbm.sample_v(hk)
            vk[v0<0]=v0[v0<0]
        phk,_=rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        train_loss=torch.mean(torch.abs(vk-v0))
        s=s+1
    print('epoch is {}'.format(epoch))
    print('train_loss is {}'.format(train_loss/s))
    
#test
    s=0
    test_loss=0
    for j in range(0,nb_users):
        vt=test[j:j+batch_size]
        v=train[j:j+batch_size]
        
        
        for i in range(1):
            _,h=rbm.sample_h(v)
            _,v=rbm.sample_v(h)
        
    
        test_loss=torch.mean(torch.abs(v-vt))
        s=s+1
    
    print('testloss {}'.format(test_loss/s))
    
    