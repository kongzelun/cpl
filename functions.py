import torch
import torch.nn as nn


class DCE(nn.Module):
    def __init__(self, gamma=1.0):
        super(DCE, self).__init__()

        self.gamma = gamma
        self.distance = nn.PairwiseDistance(p=2)

    def forward(self, feature, label, all_prototypes):
        one = 0.0

        for c in all_prototypes:
            for p in all_prototypes[c]:
                one += torch.tensor(-self.gamma * self.distance(feature, p.feature)).exp()

        probability = 0.0

        for p in all_prototypes[label]:
            probability += -(torch.tensor(-self.gamma * self.distance(feature, p.feature)).exp() / one).log()

        return probability


class PL(nn.Module):
    def __init__(self, lambda_=1):
        super(DCE, self).__init__()

        self.lambda_ = lambda_

    def forward(self, features, closest_prototype):
        pass


class Prototype:
    def __init__(self, label):
        self.label = label
        self.feature = None
        self.samples = []

    def update(self, feature):
        self.samples.append(feature)
        self.feature = sum(self.samples) / len(self.samples)
