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
    dataset_path = "data/cifar10_train.csv"
    pkl_path = "pkl/cifar10 - home.pkl"
    log_path = "log - home.txt"
    tensor_view = (-1, 32, 32)
    in_channels = 3

    threshold = 10.0
    tao = 10.0
    lambda_ = 0.1

    # gamma * threshold ~ 1
    gamma = 0.1


class DataSet(Dataset):
    def __init__(self, dataset):
        self.data = []

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(*Config.tensor_view)
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
    def __init__(self, device, number_layers=6, growth_rate=12, reduction=2, bottleneck=True, drop_rate=0.0):
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


class Prototypes(object):
    def __init__(self, label):
        self.label = label
        self.sample_count = []
        self.features = []

    def append(self, feature):
        self.features.append(feature)
        self.sample_count.append(1)

    def update(self, index, feature):
        self[index] = (self.features[index] * self.sample_count[index] + feature) / (self.sample_count[index] + 1)
        self.sample_count[index] += 1

    def __getitem__(self, item):
        return self.features[item]

    def __setitem__(self, key, value):
        self.features[key] = value

    def __len__(self):
        return len(self.features)


class GCPLLoss(nn.Module):
    def __init__(self, threshold, gamma=0.1, tao=10.0, b=1.0, beta=1.0, lambda_=0.1):
        super(GCPLLoss, self).__init__()

        self.threshold = threshold
        self.lambda_ = lambda_
        self.gamma = gamma
        self.b = b
        self.tao = tao
        self.beta = beta

    def forward(self, feature, label, all_prototypes):
        closest_prototype, min_distance = self.assign_prototype(feature.data, label, all_prototypes)

        probability = compute_probability(feature, label, all_prototypes, gamma=self.gamma)
        dce_loss = -probability.log()
        # p_loss = compute_distance(feature, closest_prototype).pow(2)

        # pairwise loss
        pw_loss = self._g(self.b - (self.tao - compute_distance(feature, closest_prototype).pow(2)))

        for l in all_prototypes:
            if label != l:
                prototypes = torch.cat(all_prototypes[l].features)
                distances = compute_multi_distance(feature, prototypes)
                d = distances.min().pow(2)
                pw_loss += self._g(self.b + (self.tao - d))

        return dce_loss + self.lambda_ * pw_loss, min_distance

    def assign_prototype(self, feature, label, all_prototypes):
        closest_prototype = feature
        min_distance = 0.0

        if label not in all_prototypes:
            all_prototypes[label] = Prototypes(label)
            all_prototypes[label].append(feature)
        else:
            # find closest prototype from prototypes in corresponding class
            prototypes = torch.cat(all_prototypes[label].features)
            distances = compute_multi_distance(feature, prototypes)
            min_distance, closest_prototype_index = distances.min(dim=0)
            min_distance = min_distance.item()

            if min_distance < self.threshold:
                all_prototypes[label].update(closest_prototype_index, feature)
                closest_prototype = all_prototypes[label][closest_prototype_index]
            else:
                all_prototypes[label].append(feature)

        return closest_prototype, min_distance

    def _g(self, z):
        if z > 10:
            return z
        else:
            return (1 + (self.beta * z).exp()).log() / self.beta


compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)


def compute_probability(feature, label, all_prototypes, gamma=Config.gamma):
    one = 0.0
    probability = None

    for l in all_prototypes:
        prototypes = torch.cat(all_prototypes[l].features)
        distances = compute_multi_distance(feature, prototypes)
        one += (-gamma * distances.pow(2)).exp().sum()

        if label == l:
            probability = (-gamma * distances.pow(2)).exp().sum()

    if one.item() > 0.0:
        probability /= one

    return probability


def find_closest_prototype(feature, all_prototypes):
    # find closest prototype from all prototypes
    min_distance = None
    label = None

    for l in all_prototypes:
        prototypes = torch.cat(all_prototypes[l].features)
        distances = compute_multi_distance(feature, prototypes)
        d = distances.min().item()
        if min_distance is None or d < min_distance:
            min_distance = d
            label = l

    return label, min_distance


def predict(feature, all_prototypes):
    predicted_label, min_distance = find_closest_prototype(feature, all_prototypes)
    probability = compute_probability(feature, predicted_label, all_prototypes)

    return predicted_label, probability.item(), min_distance
