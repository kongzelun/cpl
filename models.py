import torch
import torch.nn as nn
from torch import tensor
from torch.utils.data import Dataset
import dense_net


class Config(object):
    # dataset_path = 'data/fashion-mnist_train.csv'
    # pkl_path = "pkl/fashion-mnist.pkl"
    # tensor_view = (-1, 28, 28)
    # in_channels = 1
    # dataset_path = "data/cifar10_train.csv"
    dataset_path = None
    pkl_path = None
    log_path = None
    tensor_view = None
    in_channels = None

    learning_rate = None
    threshold = None
    gamma = None
    tao = None
    b = None
    lambda_ = None

    epoch_number = 1
    test_frequency = 1
    train_test_split = 5000

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.values = kwargs

    def __repr__(self):
        return "{}".format(self.values)


class DataSet(Dataset):
    def __init__(self, dataset, tensor_view):
        self.data = []

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float) / 255).view(tensor_view)
            y = tensor(s[-1], dtype=torch.long)
            self.data.append((x, y))

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

        # self.fc1 = nn.Linear(channels * 4 * 4, 1000)
        # self.fc2 = nn.Linear(1000, 300)

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


class GCPLLoss(nn.Module):
    def __init__(self, threshold, gamma=0.1, tao=10.0, b=1.0, beta=1.0, lambda_=0.1):
        super(GCPLLoss, self).__init__()

        self.threshold = threshold
        self.lambda_ = lambda_
        self.gamma = gamma
        self.b = b
        self.tao = tao
        self.beta = beta
        self.compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
        self.compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, feature, label, all_prototypes):
        min_distance = self.assign_prototype(feature.data, label, all_prototypes)

        probability = self.compute_probability(feature, label, all_prototypes)
        dce_loss = -probability.log()
        # p_loss = compute_distance(feature, closest_prototype).pow(2)

        # pairwise loss
        distances = self.compute_multi_distance(feature, torch.cat(all_prototypes.dict[label]))
        pw_loss = self._g(self.b - (self.tao - distances)).sum()

        for l in all_prototypes.dict:
            distances = self.compute_multi_distance(feature, torch.cat(all_prototypes.dict[l]))
            pw_loss += self._g(self.b + (self.tao - distances)).sum()

        return dce_loss + self.lambda_ * pw_loss, min_distance

    def assign_prototype(self, feature, label, all_prototypes: Prototypes):
        # closest_prototype = feature
        min_distance = 0.0

        if label not in all_prototypes.dict:
            all_prototypes.append(feature, label)
        else:
            # find closest prototype from prototypes in corresponding class
            prototypes = all_prototypes.get(label)
            distances = self.compute_multi_distance(feature, torch.cat(prototypes))
            min_distance, closest_prototype_index = distances.min(dim=0)
            min_distance = min_distance.item()

            if min_distance < self.threshold:
                Prototypes.update(prototypes[closest_prototype_index], feature)
                # closest_prototype = prototypes[closest_prototype_index]
            else:
                all_prototypes.append(feature, label)

        return min_distance

    def _g(self, z):
        return (1 + (self.beta * z).exp()).log() / self.beta

    def compute_probability(self, feature, label, all_prototypes):
        distances = self.compute_multi_distance(feature, torch.cat(all_prototypes.features))
        one = (-self.gamma * distances.pow(2)).exp().sum()

        distances = self.compute_multi_distance(feature, torch.cat(all_prototypes.get(label)))
        probability = (-self.gamma * distances.pow(2)).exp().sum() / one

        return probability

    def predict(self, feature, all_prototypes):
        # find closest prototype from all prototypes
        distances = self.compute_multi_distance(feature, torch.cat(all_prototypes.features))
        min_distance, index = distances.min(dim=0)

        predicted_label = all_prototypes[index].label
        probability = self.compute_probability(feature, predicted_label, all_prototypes)

        return predicted_label, probability.item(), min_distance.item()
