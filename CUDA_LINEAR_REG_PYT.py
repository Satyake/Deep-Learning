import torch
from torch.autograd import Variable
import numpy as np 
x=[0,2,4,5,6,7,8]
y=[2,4,5,6,7,8,9]
x=np.array(x,dtype=np.float32)
y=np.array(x,dtype=np.float32)
x=x.reshape(-1,1)
y=y.reshape(-1,1)
import torch.nn as nn
class LR(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LR,self).__init__()
        self.linear=nn.Linear(input_dim,output_dim)
    def forward(self,x):
        output=self.linear(x)
        return output
input_dim=1
output_dim=1
lr=0.01

model=LR(input_dim,output_dim)
model.cuda()

optimizer=torch.optim.SGD(model.parameters(),lr)
criterion=nn.MSELoss()

for e in range(100):
    e=e+1
    inputs=Variable(torch.from_numpy(x).cuda())
    labels=Variable(torch.from_numpy(y).cuda())
    optimizer.zero_grad()
    outputs=model(inputs)
    loss=criterion(labels,outputs)
    loss.backward()
    optimizer.step()
    print('epoch is {} and loss is {}'.format(e,loss.item()))