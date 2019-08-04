import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms

train=dsets.MNIST(root='G:/MNIST',train=True,transform=transforms.ToTensor(),download=True)
test=dsets.MNIST(root='G:/MNIST',train=False,transform=transforms.ToTensor(),download=True)

#making it iterable 
train_loader=torch.utils.data.DataLoader(dataset=train,batch_size=2,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=test,batch_size=2,shuffle=False)

class RNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,layer_dim,output_dim):
      super(RNN,self).__init__()
      self.hidden_dim=hidden_dim
      self.layer_dim=layer_dim
      self.RNN=nn.RNN(input_dim,hidden_dim,layer_dim,batch_first=True,nonlinearity='relu')
      self.fc1=nn.Linear(hidden_dim,output_dim)
    
    def forward(self,x):
      h0=Variable(torch.zeros(self.layer_dim,x.size(0),self.hidden_dim))
      out,hn=self.RNN(x,h0.detach())
      out=out[:,-1,:]
      out=self.fc1(out)
      return out
input_dim=28
hidden_dim=200
layer_dim=1
output_dim=10  #pass 28 each time step
model=RNN(input_dim,hidden_dim,layer_dim,output_dim)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)
iter=0
correct=0
total=0
sequence_dim=28
for e in range(100):
    e=e+1
    for i,(images,labels) in enumerate(train_loader):
        images=Variable(images.view(-1,sequence_dim,input_dim))
        labels=Variable(labels)
        optimizer.zero_grad()
        output=model(images)
        loss=criterion(output,labels)
        loss.backward()
        optimizer.step
        iter=iter+1
        if iter%300==0:
          for (images,labels) in test_loader:
              images=Variable(images.view(-1,sequence_dim,input_dim))
              labels=Variable(labels)
              outputs=model(images)
              _,pred=torch.max(outputs.data,1)
              total=total+len(labels)
              correct=correct+(pred==labels).sum()
          accuracy=(correct/total)*100
          print('epochs are {}, accuracy {}, loss{}'.format(e,loss.item(),accuracy))
          