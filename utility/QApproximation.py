import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class NetWork(nn.Module):
    def __init__(self,
                 input_height : int =84,
                 input_width : int = 84, 
                 in_channels : int = 1,
                 action_num : int = 5, 
                 ):
        super(NetWork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        # self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        # self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # self.bn3 = nn.BatchNorm2d(64)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.head = nn.Linear(512, action_num)
    
    def CurvOutputSizeCount(self,height, width, kernel_size, stride):
        output_height = (height - kernel_size) / stride + 1
        output_width = (width - kernel_size) / stride + 1
        return output_height, output_width        
        
    def forward(self, x):
        x = x.float() / 255
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.head(x)