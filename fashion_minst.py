import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor
from torch.utils.data import Dataset

DATASET_PATH = 'data/fashion-mnist_train.csv'
PKL_PATH = "pkl/fashion-mnist.pkl"
THRESHOLD = 1.0


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
            nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, padding=2),
            nn.ReLU(),
            # [batch_size, 10, 28, 28]
            nn.MaxPool2d(2, 2)
        )
        self.conv2 = nn.Sequential(
            # [batch_size, 10, 14, 14]
            nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5, padding=2),
            nn.ReLU(),
            # [batch_size, 20, 14, 14]
            nn.MaxPool2d(2, 2)
        )

        self.cnn_output_dim = 20 * 7 * 7

        self.device = device
        self.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
