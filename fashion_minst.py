import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import Dataset

DATASET_PATH = 'data/fashion-mnist_train.csv'
PKL_PATH = "pkl/fashion-mnist.pkl"
THRESHOLD = 1.5


class CPLDataset(Dataset):
    def __init__(self, dataset):
        self.data = []

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(-1, 28, 28)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CPLNet(nn.Module):
    def __init__(self, device):
        super(CPLNet, self).__init__()

        self.conv1 = nn.Sequential(
            # [batch_size, 1, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=50, kernel_size=7, padding=3),
            nn.ReLU(),
            # [batch_size, 50, 28, 28]
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            # [batch_size, 50, 14, 14]
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=5, padding=2),
            nn.ReLU(),
            # [batch_size, 100, 14, 14]
            nn.MaxPool2d(2, 2)
        )

        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, droprate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=droprate, inplace=True)

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))

        return


class DenseNet(nn.Module):
    def __init__(self, device):
        super(DenseNet, self).__init__()

    def forward(self, x):
        pass
