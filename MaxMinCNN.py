"""
######################## INTRODUCTION ##############################
Just begun learning pytorch so trying my hands on it to implement a 
paper on Max-Min Convolutional Neural Networks for Image 
Classification 
Will try it on CIFAR10 Dataset
Cheers!!
####################################################################
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

"""
About The Dataset CIFAR10
CIFAR10 has 32*32 pixel size (R, G, B) Images 
"""

"""
################## Architecture Explained ####################
## All Parameters and Hyperparameters set according to 
## original CIFAR10 Paper

  Input Size = (32, 32, 3)
After First Convolution and concatenation...
  Size = (32, 32, 32)
After Pooling...
  Size = (15, 15, 32) # Overlapping takes place in pooling

After Second Convolution and concatenation...
  Size = (15, 15, 32)
After Pooling...
  Size = (7, 7, 32)

After Third Convolution and concatenation...
  Size = (7, 7, 64)
After Pooling...
  Size = (3, 3, 64)
  
After Flattening the Layer
  Parameters = 3*3*64 = 576
  
Second Fully Connected Layer
  Parameters = 64
  
output = 10 (10 classes)

"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(32, 16, 5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(576, 64)
        self.fc2 = nn.Linear(64, 10)
                
    def forward(self, x):
        x = torch.cat((self.conv1(x), -1*(self.conv1(x))), 1)
        x = self.pool1(F.relu(x))
        x = torch.cat((self.conv2(x), -1*(self.conv2(x))), 1)
        x = self.pool2(F.relu(x))
        x = torch.cat((self.conv3(x), -1*(self.conv3(x))), 1)
        x = self.pool3(F.relu(x))
        x = x.view(-1, 576)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

###### Training The Dataset
for epoch in range(25):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')


### Testing the Dataset ###
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

### Output ###
outputs = net(images)
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))
    
"""
Results:-
Well GPU required definitely as training time significantly 
increased but good good results after just 2 epochs of about 1.5
loss as compared to 1.8 originally
DONE :)
"""