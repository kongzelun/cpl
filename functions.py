import torch
import torch.nn as nn

distance = nn.PairwiseDistance(p=2)


def assign_prototype(feature, label, all_prototypes, threshold):
    if label not in all_prototypes:
        all_prototypes[label] = []
        prototype = Prototype(label)
        prototype.update(feature)
        all_prototypes[label].append(prototype)
        closest_prototype = prototype
    else:
        closest_prototype, minimum_distance = find_closest_prototype(feature, all_prototypes, label)

        if minimum_distance < threshold:
            closest_prototype.update(feature)
        else:
            prototype = Prototype(label)
            prototype.update(feature)
            all_prototypes[label].append(prototype)
            closest_prototype = prototype

    return closest_prototype


def find_closest_prototype(feature, all_prototypes, label=None):
    minimum_distance = None
    closest_prototype = None

    if label is None:
        # find closest prototype from all prototypes
        for c in all_prototypes:
            for p in all_prototypes[c]:
                d = distance(feature, p.feature)

                if minimum_distance is None:
                    closest_prototype = p
                    minimum_distance = d
                elif d < minimum_distance:
                    closest_prototype = p
                    minimum_distance = d
    else:
        # find closest prototype from prototypes in corresponding class
        for p in all_prototypes[label]:
            d = distance(feature, p.feature)

            if minimum_distance is None:
                closest_prototype = p
                minimum_distance = d
            elif d < minimum_distance:
                closest_prototype = p
                minimum_distance = d

    return closest_prototype, minimum_distance


def compute_probability(feature, label, all_prototypes, gamma=1.0):
    one = 0.0

    for c in all_prototypes:
        for p in all_prototypes[c]:
            d = distance(feature, p.feature)
            one += torch.tensor(-gamma * d ** 2).exp()

    probability = 0.0

    for p in all_prototypes[label]:
        d = distance(feature, p.feature)
        probability += torch.tensor(-gamma * d ** 2).exp()

    probability /= one

    return probability


class DCELoss(nn.Module):
    def __init__(self, gamma=1.0):
        super(DCELoss, self).__init__()

        self.gamma = gamma

    def forward(self, feature, label, all_prototypes):
        probability = compute_probability(feature, label, all_prototypes, gamma=self.gamma)

        return -probability.log()


class PLoss(nn.Module):
    def __init__(self):
        super(PLoss, self).__init__()

    def forward(self, feature, closest_prototype):
        d = distance(feature, closest_prototype.feature)

        return d ** 2


class GCPLLoss(nn.Module):
    def __init__(self, gamma=1.0, lambda_=0.1):
        super(GCPLLoss, self).__init__()

        self.lambda_ = lambda_
        self.dce = DCELoss(gamma)
        self.pl = PLoss()

    def forward(self, feature, label, all_prototypes, closest_prototype):
        return self.dce(feature, label, all_prototypes) + self.lambda_ * self.pl(feature, closest_prototype)


class Prototype:
    def __init__(self, label):
        self.label = label
        self.feature = None
        self.samples = []

    def update(self, feature):
        self.samples.append(feature)
        self.feature = sum(self.samples) / len(self.samples)
