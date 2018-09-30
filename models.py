import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import Dataset
import dense_net


class Config:
    pass


class DataSet(Dataset):
    def __init__(self, dataset, tensor_view):
        self.data = []
        self.label_set = set()

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(tensor_view)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))
            self.label_set.add(s[-1])

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CNNNet(nn.Module):
    def __init__(self, device, in_channels):
        super(CNNNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=10, kernel_size=5, padding=2),
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
    def __init__(self, device, in_channels, number_layers=6, growth_rate=12, reduction=2, bottleneck=True, drop_rate=0.0):
        super(DenseNet, self).__init__()

        channels = 2 * growth_rate

        if bottleneck:
            block = dense_net.BottleneckBlock
        else:
            block = dense_net.BasicBlock

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False)

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
        return out


class LinearNet(nn.Module):
    def __init__(self, device, in_features):
        super(LinearNet, self).__init__()

        self.fc1 = nn.Linear(in_features, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.fc3 = nn.Linear(100, 10)

        self.relu = nn.ReLU(inplace=True)

        self.device = device
        self.to(device)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Prototypes(object):
    def __init__(self):
        self.features = []
        self.dict = {}

    def append(self, feature, label):
        setattr(feature, 'label', label)
        setattr(feature, 'sample_count', 1)
        self.features.append(feature)

        if label not in self.dict:
            self.dict[label] = []

        self.dict[label].append(feature)

    @staticmethod
    def update(prototype, feature):
        count = prototype.sample_count
        prototype = (prototype * count + feature) / (count + 1)
        prototype.sample_count = count + 1

    def clear(self):
        self.features.clear()
        self.dict.clear()

    def get(self, label):
        return self.dict[label]

    def __getitem__(self, item):
        return self.features[item]

    def __setitem__(self, key, value):
        self.features[key] = value

    def __len__(self):
        return len(self.features)


class DCELoss(nn.Module):
    def __init__(self, threshold=10.0, gamma=0.1, lambda_=0.1):
        super(DCELoss, self).__init__()
        self.threshold = threshold
        self.lambda_ = lambda_
        self.gamma = gamma
        self.compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
        self.compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)
        self.prototypes = Prototypes()

    def forward(self, feature, label):
        raise NotImplementedError

    def compute_dec_loss(self, feature, label):
        closest_prototype, min_distance = self.assign_prototype(feature.data, label)

        probability = self.compute_probability(feature, label)
        dce_loss = -probability.log()

        return dce_loss, closest_prototype, min_distance

    def assign_prototype(self, feature, label):
        closest_prototype = feature
        min_distance = 0.0

        if label not in self.prototypes.dict:
            self.prototypes.append(feature, label)
        else:
            # find closest prototype from prototypes in corresponding class
            prototypes = self.prototypes.get(label)
            distances = self.compute_multi_distance(feature, torch.cat(prototypes))
            min_distance, closest_prototype_index = distances.min(dim=0)
            min_distance = min_distance.item()

            if min_distance < self.threshold:
                Prototypes.update(prototypes[closest_prototype_index], feature)
                closest_prototype = prototypes[closest_prototype_index]
            else:
                self.prototypes.append(feature, label)

        return closest_prototype, min_distance

    def compute_probability(self, feature, label):
        distances = self.compute_multi_distance(feature, torch.cat(self.prototypes.features))
        one = (-self.gamma * distances.pow(2)).exp().sum()

        distances = self.compute_multi_distance(feature, torch.cat(self.prototypes.get(label)))
        probability = (-self.gamma * distances.pow(2)).exp().sum()

        if one.item() > 0.0:
            probability /= one

        return probability

    def predict(self, feature):
        # find closest prototype from all prototypes
        distances = self.compute_multi_distance(feature, torch.cat(self.prototypes.features))
        min_distance, index = distances.min(dim=0)

        predicted_label = self.prototypes[index].label
        probability = self.compute_probability(feature, predicted_label)

        return predicted_label, probability.item(), min_distance.item()

    def clear(self):
        self.prototypes.clear()


class PairwiseDCELoss(DCELoss):
    def __init__(self, threshold=10.0, gamma=0.1, tao=10.0, b=1.0, beta=1.0, lambda_=0.1):
        super(PairwiseDCELoss, self).__init__(threshold, gamma, lambda_)

        self.threshold = threshold
        self.gamma = gamma
        self.lambda_ = lambda_

        self.b = b
        self.tao = tao
        self.beta = beta

    def forward(self, feature, label):
        dce_loss, closest_prototype, min_distance = self.compute_dec_loss(feature, label)

        # pairwise loss
        distance = self.compute_distance(feature, closest_prototype)
        pw_loss = self._g(self.b - (self.tao - distance.pow(2)))

        for l in self.prototypes.dict:
            if l != label:
                prototypes = torch.cat(self.prototypes.get(l))
                distance = self.compute_multi_distance(feature, prototypes).min()
                pw_loss += self._g(self.b + (self.tao - distance.pow(2)))

        return dce_loss + self.lambda_ * pw_loss, min_distance

    def _g(self, z):
        return (1 + (self.beta * z).exp()).log() / self.beta


class GCPLLoss(DCELoss):
    def __init__(self, threshold, gamma=0.1, lambda_=0.01):
        super(GCPLLoss, self).__init__()

        self.threshold = threshold
        self.lambda_ = lambda_
        self.gamma = gamma
        self.compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
        self.compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)
        self.log_softmax = nn.LogSoftmax()
        self.prototypes = Prototypes()

    def forward(self, feature, label):
        dce_loss, closest_prototype, min_distance = self.compute_dec_loss(feature, label)
        p_loss = self.compute_distance(feature, closest_prototype).pow(2)

        return dce_loss + self.lambda_ * p_loss, min_distance
