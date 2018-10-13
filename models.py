import torch
import torch.nn as nn
import numpy as np
import dense_net

compute_distance = nn.PairwiseDistance(p=2, eps=1e-6)
compute_multi_distance = nn.PairwiseDistance(p=2, eps=1e-6, keepdim=True)


# class DataSet(Dataset):
#     def __init__(self, dataset, tensor_view):
#         self.data = []
#         self.label_set = set()
#
#         for s in dataset:
#             x = (tensor(s[:-1], dtype=torch.float)).view(tensor_view)
#             y = tensor(s[-1], dtype=torch.long)
#             self.data.append((x, y))
#             self.label_set.add(int(s[-1]))
#
#     def __getitem__(self, index):
#         return self.data[index]
#
#     def __len__(self):
#         return len(self.data)

class DenseNet(nn.Module):
    def __init__(self, device, in_channels, number_layers=6, growth_rate=12, reduction=2, bottleneck=True, drop_rate=0.0):
        super(DenseNet, self).__init__()

        channels = 2 * growth_rate

        if bottleneck:
            block = dense_net.BottleneckBlock
        else:
            block = dense_net.BasicBlock

        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=True)

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
        self.pooling = nn.MaxPool2d(kernel_size=2)

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
        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU(inplace=True)

        self.device = device
        self.to(device)

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.softmax(self.fc3(out))
        return out


class Prototype(object):
    def __init__(self, feature, label):
        self.feature = feature
        self.label = label
        self.sample_count = 1

    def update(self, feature):
        prototype = (self.feature * self.sample_count + feature) / (self.sample_count + 1)
        prototype.sample_count = self.sample_count + 1


class Prototypes(object):
    def __init__(self, threshold):
        super(Prototypes, self).__init__()

        self._list = []
        self._dict = {}
        self.threshold = threshold

    def assign(self, feature, label):
        """
        Assign the sample to a prototype.
        :param feature: feature of the sample.
        :param label: label of the sample.
        :return:
        """

        min_distance = 0.0
        closest_prototype = Prototype(feature, label)

        with torch.no_grad():
            if label not in self._dict:
                self._append(feature, label)
            else:
                # find closest prototype from prototypes in corresponding class
                prototypes = self.get(label)
                distances = compute_multi_distance(feature, torch.cat(prototypes))
                distance, closest_prototype_index = distances.min(dim=0)
                min_distance = distance.item()

                if min_distance < self.threshold:
                    closest_prototype = self._dict[label][closest_prototype_index]
                    closest_prototype.update(feature)
                else:
                    self._append(feature, label)

        return min_distance, closest_prototype

    def _append(self, feature, label):
        # setattr(feature, 'label', label)
        # setattr(feature, 'sample_count', 1)
        prototype = Prototype(feature, label)
        self._list.append(prototype)

        if label not in self._dict:
            self._dict[label] = []

        self._dict[label].append(prototype)

    def clear(self):
        self._list.clear()
        self._dict.clear()

    def upgrade(self):
        for p in self._list:
            p.sample_count = 1

    def get(self, label=None):
        collection = self._list if label is None else self._dict[label]
        return list(map(lambda p: p.feature, collection))

    def save(self, pkl_path):
        torch.save(self, pkl_path)

    @staticmethod
    def load(pkl_path):
        prototypes = torch.load(pkl_path)

        if not isinstance(prototypes, Prototypes):
            raise RuntimeError("Prototypes pickle file format error!")

        return prototypes

    def __getitem__(self, item):
        return self._list[item]

    def __setitem__(self, key, value):
        self._list[key] = value

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self):
            self.index += 1
            return self[self.index - 1]
        else:
            raise StopIteration

    def __len__(self):
        return len(self._list)


class DCELoss(nn.Module):
    def __init__(self, threshold, gamma=0.1, lambda_=0.1):
        super(DCELoss, self).__init__()
        self.gamma = gamma
        self.lambda_ = lambda_
        self.prototypes = Prototypes(threshold)
        # self.softmax = nn.Softmax(dim=0)

    def forward(self, feature, label):
        raise NotImplementedError

    def dec_loss(self, feature, label):
        min_distance, closest_prototype = self.prototypes.assign(feature.data, label)

        probability = self.probability(feature, label)
        dce_loss = -probability.log()

        return dce_loss, min_distance, closest_prototype

    def probability(self, feature, label):
        distances = compute_multi_distance(feature, torch.cat(self.prototypes.get()))
        one = (-self.gamma * distances).exp().sum()
        # one = (-self.gamma * distances.pow(2)).exp().sum()

        distances = compute_multi_distance(feature, torch.cat(self.prototypes.get(label)))
        prob = (-self.gamma * distances).exp().sum()
        # prob = (-self.gamma * distances.pow(2)).exp().sum()

        if one.item() > 0.0:
            prob /= one
        else:
            prob += 1e-6

        return prob

    def predict(self, feature):
        # find closest prototype from all prototypes
        distances = compute_multi_distance(feature, torch.cat(self.prototypes.get()))
        min_distance, index = distances.min(dim=0)

        predicted_label = self.prototypes[index].label
        probability = self.probability(feature, predicted_label)

        return predicted_label, probability.item(), min_distance.item()

    def save_prototypes(self, path):
        self.prototypes.save(path)

    def load_prototypes(self, path):
        self.prototypes = Prototypes.load(path)

    def clear_prototypes(self):
        self.prototypes.clear()

    def upgrade_prototypes(self):
        self.prototypes.upgrade()

    def set_threshold(self, value):
        self.prototypes.threshold = value

    def set_gamma(self, value):
        self.gamma = value


class PairwiseDCELoss(DCELoss):
    def __init__(self, threshold, gamma=0.1, tao=10.0, b=1.0, lambda_=0.1):
        super(PairwiseDCELoss, self).__init__(threshold, gamma, lambda_)

        self.b = b
        self.tao = tao

    def forward(self, feature, label):
        dec_loss, min_distance, prototype = self.dec_loss(feature, label)

        # pairwise loss
        distance = compute_distance(feature, prototype.feature)
        pw_loss = self._g(self.b - (self.tao - distance))
        # pw_loss = self._g(self.b - (self.tao - distance.pow(2)))
        #
        # for l in self.prototypes._dict:
        #     if l != label:
        #         prototypes = torch.cat(self.prototypes.get(l))
        #         distances = compute_multi_distance(feature, prototypes).min()
        #         pw_loss += self._g(self.b + (self.tao - distances.pow(2)))

        prototypes = self.prototypes.get()
        distances = compute_multi_distance(feature, torch.cat(prototypes))
        distance, closest_prototype_index = distances.min(dim=0)
        closest_prototype = self.prototypes[closest_prototype_index]
        like = 1 if closest_prototype.label == label else -1
        pw_loss += self._g(self.b - like * (self.tao - distance))
        # pw_loss += self._g(self.b - like * (self.tao - distance.pow(2)))

        # for p in self.prototypes:
        #     like = 1 if p.label == label else -1
        #     distance = compute_distance(feature, p.feature)
        #     pw_loss = self._g(self.b - like * (self.tao - distance.pow(2)))
        #     loss += self.lambda_ * pw_loss.squeeze(0)

        # for p in self.prototypes:
        #     like = 1 if p.label == label else -1
        #     distance = compute_distance(feature, p.feature)
        #     pw_loss = self._g(self.b - like * (self.tao - distance))
        #     loss += self.lambda_ * pw_loss.squeeze(0)

        return dec_loss + self.lambda_ * pw_loss, min_distance

    def _g(self, z):
        return (1 + (self.gamma * z).exp()).log() / self.gamma

    def set_tao(self, value):
        self.tao = value

    def set_b(self, value):
        self.b = value


class GCPLLoss(DCELoss):
    def __init__(self, threshold, gamma=0.1, lambda_=0.01):
        super(GCPLLoss, self).__init__(threshold, gamma, lambda_)

    def forward(self, feature, label):
        dce_loss, min_distance, closest_prototype = self.dec_loss(feature, label)
        p_loss = compute_distance(feature, closest_prototype).pow(2)

        return dce_loss + self.lambda_ * p_loss, min_distance


class Detector(object):
    def __init__(self, intra_class_distances, std_coefficient, known_labels):
        self.distances = np.array(intra_class_distances, dtype=[('label', np.int32), ('distance', np.float32)])
        self.std_coefficient = std_coefficient
        self.known_labels = known_labels
        self.average_distances = {l: np.average(self.distances[self.distances['label'] == l]['distance']) for l in self.known_labels}
        self.std_distances = {l: self.distances[self.distances['label'] == l]['distance'].std() for l in self.known_labels}
        self.thresholds = {l: self.average_distances[l] + (self.std_coefficient * self.std_distances[l]) for l in self.known_labels}
        self.results = None

    def __call__(self, predicted_label, probability, distance):
        novelty = False
        if (distance > self.thresholds[predicted_label] and probability < 0.75) or probability < 0.25:
            novelty = True

        return novelty

    def evaluate(self, results):
        self.results = np.array(results, dtype=[
            ('true_label', np.int32),
            ('predicted_label', np.int32),
            ('probability', np.float32),
            ('distance', np.float32),
            ('real_novelty', np.bool),
            ('detected_novelty', np.bool)
        ])

        # total_novelties = self.results[~np.isin(self.results['true label'], list(self.known_labels))]
        real_novelties = self.results[self.results['real_novelty']]
        detected_novelties = self.results[self.results['detected_novelty']]
        # detected_real_novelties = detected_novelties[~np.isin(detected_novelties['true label'], list(self.known_labels))]
        detected_real_novelties = self.results[self.results['detected_novelty'] & self.results['real_novelty']]

        true_positive = len(detected_real_novelties)
        false_positive = len(detected_novelties) - len(detected_real_novelties)
        false_negative = len(real_novelties) - len(detected_real_novelties)

        # precision = true_positive / (true_positive + false_positive)
        # recall = true_positive / (true_positive + false_negative)

        return true_positive, false_positive, false_negative


class SoftmaxDetector(object):
    def __init__(self, probs, std_coefficient, known_labels):
        self.probs = np.array(probs, dtype=[('label', np.float), ('prob', np.float)])
        self.mean_prob = {l: np.average(self.probs[self.probs['label'] == l]['prob']) for l in known_labels}
        self.std_prob = {l: self.probs[self.probs['label'] == l]['prob'].std() for l in known_labels}
        self.std_coefficient = std_coefficient
        self.known_labels = known_labels
        self.thresholds = {l: self.mean_prob[l] + (self.std_coefficient * self.std_prob[l]) for l in known_labels}
        self.results = None

    def __call__(self, predicted_label, prob):
        novelty = False
        if predicted_label not in self.known_labels or prob < self.thresholds[predicted_label]:
            novelty = True

        return novelty

    def evaluate(self, results):
        self.results = np.array(results, dtype=[
            ('true_label', np.int32),
            ('predicted_label', np.int32),
            ('probability', np.float32),
            ('real_novelty', np.bool),
            ('detected_novelty', np.bool)
        ])

        real_novelties = self.results[self.results['real_novelty']]
        detected_novelties = self.results[self.results['detected_novelty']]
        detected_real_novelties = self.results[self.results['detected_novelty'] & self.results['real_novelty']]

        true_positive = len(detected_real_novelties)
        false_positive = len(detected_novelties) - len(detected_real_novelties)
        false_negative = len(real_novelties) - len(detected_real_novelties)

        return true_positive, false_positive, false_negative
