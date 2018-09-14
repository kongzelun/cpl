import torch
import torch.nn as nn

compute_distance = nn.PairwiseDistance(p=2)
compute_multi_distance = nn.PairwiseDistance(p=2, keepdim=True)


# def assign_prototype(feature, label, all_prototypes, threshold):
#     if label not in all_prototypes:
#         all_prototypes[label] = []
#         prototype = Prototype(label)
#         prototype.update(feature)
#         all_prototypes[label].append(prototype)
#         closest_prototype = prototype
#     else:
#         closest_prototype, minimum_distance = find_closest_prototype(feature, all_prototypes, label)
#
#         if minimum_distance < threshold:
#             closest_prototype.update(feature)
#         else:
#             prototype = Prototype(label)
#             prototype.update(feature)
#             all_prototypes[label].append(prototype)
#             closest_prototype = prototype
#
#     return closest_prototype


def assign_prototype(feature, label, all_prototypes, threshold):
    if label not in all_prototypes:
        all_prototypes[label] = Prototypes(label)
        all_prototypes[label].append(feature)
        closest_prototype_index = len(all_prototypes[label]) - 1
        min_distance = 0.0
    else:
        closest_prototype_index, min_distance = find_closest_prototype(feature, all_prototypes, label)
        if min_distance < threshold:
            all_prototypes[label].update(closest_prototype_index, feature)
        else:
            all_prototypes[label].append(feature)
            min_distance = 0.0

    return closest_prototype_index, min_distance


# def find_closest_prototype(feature, all_prototypes, label=None):
#     minimum_distance = None
#     closest_prototype = None
#
#     if label is None:
#         # find closest prototype from all prototypes
#         for c in all_prototypes:
#             for p in all_prototypes[c]:
#                 d = compute_distance(feature, p.feature)
#
#                 if minimum_distance is None:
#                     closest_prototype = p
#                     minimum_distance = d
#                 elif d < minimum_distance:
#                     closest_prototype = p
#                     minimum_distance = d
#     else:
#         # find closest prototype from prototypes in corresponding class
#         for p in all_prototypes[label]:
#             d = compute_distance(feature, p.feature)
#
#             if minimum_distance is None:
#                 closest_prototype = p
#                 minimum_distance = d
#             elif d < minimum_distance:
#                 closest_prototype = p
#                 minimum_distance = d
#
#     return closest_prototype, minimum_distance

def find_closest_prototype(feature, all_prototypes, label=None):
    if label is None:
        # find closest prototype from all prototypes
        min_distance = None
        closest_prototype_index = None

        for label in all_prototypes:
            prototypes = torch.cat(all_prototypes[label].prototypes)
            distances = compute_multi_distance(feature, prototypes)
            d, index = distances.min(dim=0)
            if min_distance is None or d < min_distance:
                min_distance, closest_prototype_index = d, index
    else:
        # find closest prototype from prototypes in corresponding class
        prototypes = torch.cat(all_prototypes[label].prototypes)
        distances = compute_multi_distance(feature, prototypes)
        min_distance, closest_prototype_index = distances.min(dim=0)

    return closest_prototype_index, min_distance


# def compute_probability(feature, label, all_prototypes, gamma):
#     one = 0.0
#
#     for c in all_prototypes:
#         for p in all_prototypes[c]:
#             d = compute_distance(feature, p.feature)
#             one += torch.tensor(-gamma * d ** 2).exp()
#
#     probability = 0.0
#
#     for p in all_prototypes[label]:
#         d = compute_distance(feature, p.feature)
#         probability += torch.tensor(-gamma * d ** 2).exp()
#
#     probability /= one
#
#     return probability

def compute_probability(feature, label, all_prototypes, gamma):
    one = 0.0

    for l in all_prototypes:
        prototypes = torch.cat(all_prototypes[l].prototypes)
        distances = compute_multi_distance(feature, prototypes)
        distances = (-gamma * distances.pow(2)).exp()
        one += distances.sum()

    prototypes = torch.cat(all_prototypes[label].prototypes)
    distances = compute_multi_distance(feature, prototypes)
    distances = (-gamma * distances.pow(2)).exp()
    probability = distances.sum() / one

    return probability


# class DCELoss(nn.Module):
#     def __init__(self, gamma=1.0):
#         super(DCELoss, self).__init__()
#         self.gamma = gamma
#
#     def forward(self, feature, label, all_prototypes):
#         probability = compute_probability(feature, label, all_prototypes, gamma=self.gamma)
#
#         return -probability.log()
#
#
# class PLoss(nn.Module):
#     def __init__(self):
#         super(PLoss, self).__init__()
#
#     def forward(self, feature, closest_prototype):
#         d = compute_distance(feature, closest_prototype)
#
#         return d ** 2
#
#
# class GCPLLoss(nn.Module):
#     def __init__(self, gamma=1.0, lambda_=0.1):
#         super(GCPLLoss, self).__init__()
#
#         self.lambda_ = lambda_
#         self.dce = DCELoss(gamma)
#         self.pl = PLoss()
#
#     def forward(self, feature, label, all_prototypes, closest_prototype):
#         return self.dce(feature, label, all_prototypes) + \
#                self.lambda_ * self.pl(feature, closest_prototype)
#
#
# class Prototype:
#     def __init__(self, label):
#         self.label = label
#         self.feature = None
#         self.sample_count = 0
#
#     def update(self, feature):
#         if self.feature is None:
#             self.feature = feature
#         else:
#             self.feature = (self.feature * self.sample_count + feature) / (self.sample_count + 1)
#
#         self.sample_count += 1

class GCPLLoss(nn.Module):
    def __init__(self, gamma=1.0, lambda_=0.1):
        super(GCPLLoss, self).__init__()

        self.lambda_ = lambda_
        self.gamma = gamma

    def forward(self, feature, label, all_prototypes, closest_prototype):
        probability = compute_probability(feature, label, all_prototypes, gamma=self.gamma)
        dce_loss = -probability.log()
        p_loss = compute_distance(feature, closest_prototype).pow(2)

        return dce_loss + self.lambda_ * p_loss


class Prototypes(object):
    def __init__(self, label):
        self.label = label
        self.sample_counts = []
        self.prototypes = []

    def append(self, feature):
        self.prototypes.append(feature)
        self.sample_counts.append(1)

    def update(self, index, feature):
        self[index] = (self.prototypes[index] * self.sample_counts[index] + feature) / (self.sample_counts[index] + 1)
        self.sample_counts[index] += 1

    def __getitem__(self, item):
        return self.prototypes[item]

    def __setitem__(self, key, value):
        self.prototypes[key] = value

    def __len__(self):
        return len(self.prototypes)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.prototypes)
