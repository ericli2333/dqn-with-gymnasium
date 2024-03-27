import torch.nn as nn
import torch.nn.functional as F
import torch


class NetWork(nn.Module):
    """
    Deep Q-Network (DQN) class.

    Args:
        in_channels (int): Number of input channels.
        num_actions (int): Number of possible actions.

    Attributes:
        conv1 (nn.Conv2d): First convolutional layer.
        conv2 (nn.Conv2d): Second convolutional layer.
        conv3 (nn.Conv2d): Third convolutional layer.
        fc4 (nn.Linear): Fourth fully connected layer.
        fc5 (nn.Linear): Fifth fully connected layer.
    """

    def __init__(self, in_channels, num_actions):
        super(NetWork, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc5 = nn.Linear(in_features=512, out_features=num_actions)

    def forward(self, x):
        """
        Forward pass of the DQN.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)
