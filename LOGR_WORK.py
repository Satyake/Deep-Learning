import torch 
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dsets


train_dataset=dsets.MNIST(root='G:/MNIST',train=True,transform=transforms.ToTensor(),download=True)
test_dataset=dsets.MNIST(root='G:/MNIST',train=False,transform=transforms.ToTensor())
iterations=3000
batch=100
epochs=iterations/(len(train_dataset)/batch)
epochs=int(epochs)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=batch,shuffle=True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=batch,shuffle=False)

class LOGISTIC(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(LOGISTIC,self).__init__()
        self.linear=nn.Linear(input_dim,output_dim)
        
    def forward(self,x):
        output=self.linear(x)
        return output
    
input_dim=28*28
output_dim=10
model=LOGISTIC(input_dim,output_dim)
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.SGD(model.parameters(),0.01)
correct=0
total=0
iter=0
for e in range(epochs):
    for i ,(images,labels) in enumerate(train_loader):
        images=Variable(images.view(-1,28*28))
        labels=Variable(labels)
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        iter=iter+1
        if iter%200==0:
            for images,labels in test_loader:
                images=Variable(images.view(-1,28*28))
                outputs=model(images)
                _,predicted=torch.max(outputs.data,1)
                total+=labels.size(0)
                correct+=(predicted==labels).sum()
            accuracy=100*(correct/total)
            print('epoch is {} and accuracy is {} and loss is {}'.format(e,accuracy,loss.item()))
        
    