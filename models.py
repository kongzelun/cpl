import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import Dataset
import dense_net


class Config:
    # dataset_path = 'data/fashion-mnist_train.csv'
    # pkl_path = "pkl/fashion-mnist.pkl"
    # tensor_view = (-1, 28, 28)
    # in_channels = 1
    dataset_path = 'data/cifar10_train.csv'
    pkl_path = "pkl/cifar10.pkl"
    tensor_view = (-1, 32, 32)
    in_channels = 3

    threshold = 10.0

    # gamma * threshold < 10
    gamma = 1.0


class DataSet(Dataset):
    def __init__(self, dataset, pairwise=False):
        self.data = []
        self.pairwise = pairwise

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(*Config.tensor_view)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))

    def __getitem__(self, index):
        if self.pairwise:

            return self.data[index], self.data[(index + 1) if not index < len(self) else -1]
        else:
            return self.data[index]

    def __len__(self):
        return len(self.data)


class CNNNet(nn.Module):
    def __init__(self, device):
        super(CNNNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=Config.in_channels, out_channels=10, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, device, number_layers, growth_rate, reduction=2, bottleneck=True, drop_rate=0.0):
        super(DenseNet, self).__init__()

        channels = 2 * growth_rate

        if bottleneck:
            block = dense_net.BottleneckBlock
        else:
            block = dense_net.BasicBlock

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(in_channels=Config.in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)

        # 1st block
        self.block1 = dense_net.DenseBlock(number_layers, channels, block, growth_rate, drop_rate)
        channels = channels + number_layers * growth_rate
        self.trans1 = dense_net.TransitionBlock(channels, channels // reduction, drop_rate)
        channels = channels // reduction

        # 2nd block
        self.block2 = dense_net.DenseBlock(number_layers, channels, block, growth_rate, drop_rate)
        channels = channels + number_layers * growth_rate
        self.trans2 = dense_net.TransitionBlock(channels, channels // reduction, drop_rate)
        channels = channels // reduction

        # 3rd block
        self.block3 = dense_net.DenseBlock(number_layers, channels, block, growth_rate, drop_rate)
        channels = channels + number_layers * growth_rate

        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.AvgPool2d(kernel_size=2)

        # self.fc1 = nn.Linear(channels * 4 * 4, 100)
        # self.fc2 = nn.Linear(100, 10)

        self.channels = channels

        self.device = device
        self.to(device)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.pooling(out)
        # out = self.fc1(out.view(1, -1))
        # out = self.fc2(out)
        return out
