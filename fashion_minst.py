import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import Dataset
import dense_net


class Config:
    dataset_path = 'data/fashion-mnist_train.csv'
    pkl_path = "pkl/fashion-mnist.pkl"
    threshold = 1.5


class FashionMnist(Dataset):
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


class CNNNet(nn.Module):
    def __init__(self, device):
        super(CNNNet, self).__init__()

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


class DenseNet(nn.Module):
    def __init__(self, device, depth, learning_rate, growth_rate, reduction, bottleneck=True, drop_rate=0.0):
        super(DenseNet, self).__init__()

        in_channels =

    def forward(self, x):
        pass
