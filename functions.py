import torch
import torch.nn as nn


class DCE(nn.Module):
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def forward(self, feature, label, all_prototypes):
        one = 0.0

        for c in all_prototypes:
            for p in c:
                one += torch.tensor(-self.gamma * DCE.distance(feature, p)).exp()

        probability = 0.0

        for p in all_prototypes[label]:
            probability += -(torch.tensor(-self.gamma * DCE.distance(feature, p)).exp() / one).log()

        return probability



    @staticmethod
    def distance(features, prototype):
        return (features - prototype).data.norm() ** 2


class PL(nn.Module):
    def __init__(self, lambda_=1):
        self.lambda_ = lambda_

    def forward(self, features, closest_prototype):
        pass
