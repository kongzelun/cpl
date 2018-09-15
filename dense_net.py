import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))
        return torch.cat([x, out], dim=1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(BottleneckBlock, self).__init__()

        inter_channels = out_channels * 4

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=1, padding=0, bias=False)

        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)

    def forward(self, x):
        out = self.dropout(self.conv1(self.relu(self.bn1(x))))
        out = self.dropout(self.conv2(self.relu(self.bn2(out))))
        return torch.cat([x, out], dim=1)


class TransitionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, drop_rate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.dropout = nn.Dropout(p=drop_rate, inplace=False)
        self.pooling = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        return self.pooling(self.dropout(self.conv1(self.relu(self.bn1(x)))))


class DenseBlock(nn.Module):
    def __init__(self, number_layers, in_channels, block, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()

        layers = []

        for i in range(number_layers):
            layers.append(block(in_channels=in_channels + i * growth_rate, out_channels=growth_rate, drop_rate=drop_rate))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
