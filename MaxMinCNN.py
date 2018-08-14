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


"""
################## Model Explained ####################
1)-> 3 input channels (R, G, B), 32 filters, 5 kernel_size
2)-> pooling reduces size ()
1)-> 3 input channels (R, G, B), 64 filters, 5 kernel_size
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.pool3 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear()
        self.fc2 = nn.Linear()
        self.fc3 = nn.Linear()
                
    def forward(self, x):
        x = np.concatenate((self.conv1(x), -1*(self.conv1(x))), axis=1)
        x = self.pool1(F.relu(x))
        x = np.concatenate((self.conv2(x), -1*(self.conv2(x))), axis=1)
        x = self.pool2(F.relu(x))
        x = np.concatenate((self.conv3(x), -1*(self.conv3(x))), axis=1)
        x = self.pool3(F.relu(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
net = Net()
















        
        