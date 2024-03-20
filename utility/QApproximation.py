import torch
import numpy as np
import torch.nn as nn

class NetWork(nn.Module):
    def __init__(self,
                 input_shape : tuple = (84, 84), 
                 in_channels : int = 1,
                 action_num : int = 5, 
                 ):
        super(NetWork, self).__init__()
        self.curv1 = nn.Conv2d(in_channels=in_channels, out_channels=16,stride=4, kernel_size=(8,8))
        self.curv2 = nn.Conv2d(in_channels= 16,out_channels=32,stride=2, kernel_size=(4,4))
        self.curv3 = nn.Conv2d(in_channels= 32,out_channels=64,stride=1, kernel_size=(3,3))
        # 计算全连接层的输入大小
        outheight1, outwidth1 = self.CurvOutputSizeCount(input_shape[0], input_shape[1], 8, 4)
        outheight2, outwidth2 = self.CurvOutputSizeCount(outheight1, outwidth1, 4, 2)
        outheight3, outwidth3 = self.CurvOutputSizeCount(outheight2, outwidth2, 3, 1)
        self.liner_input_size = outheight3*outwidth3*32
        self.fc = nn.Linear(self.liner_input_size, action_num)

    def CurvOutputSizeCount(input_height, input_width, kernel_size, stride):
        output_height = (input_height - kernel_size) / stride + 1
        output_width = (input_width - kernel_size) / stride + 1
        return output_height, output_width        
        
    def forward(self, x):
        x = self.curv1(x)
        x = nn.functional.relu(x)
        x = self.curv2(x)
        x = nn.functional.relu(x)
        x = self.curv3(x)
        x = x.view(-1, self.liner_input_size)  # 展平卷积层的输出
        x = nn.functional.relu(x)
        x = self.fc(x)
        return x