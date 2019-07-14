import pandas as pd 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torch 
transform=transforms.Compose([transforms.Resize(28),transforms.ToTensor()])
data_Train=dsets.ImageFolder(root='G:/root',transform=transform)
data_Test=dsets.ImageFolder(root='G:/test',transform=transform)

train_loader=torch.utils.data.DataLoader(dataset=data_Train,batch_size=5,shuffle=True)
test_loader=torch.utils.data.DataLoader(dataset=data_Test,batch_size=5,shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=16,kernel_size=5,stride=1,padding=2)
        self.relu1=nn.ReLU()
        self.MP1=nn.MaxPool2d(kernel_size=2)
        
        self.conv2=nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5,stride=1,padding=2)
        self.relu2=nn.ReLU()
        self.MP2=nn.MaxPool2d(kernel_size=2)
        
        self.linear=nn.Linear(32*7*7,2)
        
    def forward(self,x):
        out=self.conv1(x)
        out=self.relu1(out)
        out=self.MP1(out)
        
        out=self.conv2(out)
        out=self.relu2(out)
        out=self.MP2(out)
        out=out.view(out.size(0),-1)
        out=self.linear(out)
        return out
model=CNN()
criterion=nn.CrossEntropyLoss()
lr=0.01
optim=torch.optim.SGD(model.parameters(),lr)

iter=0
total=0
correct=0
for e in range(10):
    for i,(images,labels) in enumerate(train_loader):
        #images=item['image']
        #labels=item['label']
        e=e+1
        images=Variable(images)
        labels=Variable(labels)
        optim.zero_grad()
        output=model(images)
        loss=criterion(output,labels)
        loss.backward()
        optim.step
        iter=iter+1
        if iter%500==0:
            for images,labels in test_loader:
                images=Variable(images)
                labels=Variable(labels)
                #images=item['image']
                #labels=item['label']
                output=model(images)
                _,pred=torch.max(output.data,1)
                total+=labels.size(0)
                correct+=(pred==labels).sum()
            accuracy=(correct/total)*100
            print('epochs {} loss {} accuracy{}'.format(e,loss.item(),accuracy))
            
from PIL import Image
Image1=Image.open('G:/airplane.JPEG')
Image1=transform(Image1)
Image1=Image1.unsqueeze(0)
Image1=Variable(Image1)
model.eval()
target=model(Image1)       
_,pred=torch.max(target.data,1) 
print('classes',data_Test.classes[pred[0]]) 
print('prediction',pred[0])       

    