import os
import argparse
import json
import logging
import time
import random
import numpy as np
import pandas as pd
import torch
from torch import tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import models
from sklearn.metrics import accuracy_score, confusion_matrix
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


class Config(object):
    # read from json
    testonly = False

    dataset_path = None
    loss_type = None

    tensor_view = None
    in_channels = None
    number_layers = 6
    growth_rate = 12

    learning_rate = None

    threshold = None
    gamma = None
    tao = None
    b = None
    lambda_ = None
    std_coefficient = None

    epoch_number = 1
    testfreq = 1
    train_test_split = None

    # derived
    running_path = None
    log_path = None
    optim_path = None
    model_path = None
    prototypes_path = None
    intra_class_distances_path = None
    probs_path = None

    device = "cpu"

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.values = kwargs

    def __repr__(self):
        return "{}".format(self.values)


class DataSet(Dataset):
    def __init__(self, dataset, tensor_view, transform=None):
        self.data = []
        self.label_set = set()
        self.transform = transform

        for s in dataset:
            x = (tensor(s[:-1], dtype=torch.float)).view(tensor_view)
            y = tensor(s[-1], dtype=torch.long)

            if self.transform:
                x = self.transform(x)

            self.data.append((x, y))
            self.label_set.add(int(s[-1]))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def setup_logger(level=logging.DEBUG, filename=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def run(config, trainset, testset):
    logger = logging.getLogger(__name__)

    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=4)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=4)

    # net = models.CNNNet(device=device)
    net = models.DenseNet(device=torch.device(config.device),
                          in_channels=config.in_channels,
                          number_layers=config.number_layers,
                          growth_rate=config.growth_rate,
                          drop_rate=0.0)
    logger.info("DenseNet Channels: %d", net.channels)

    if config.loss_type == 'gcpl':
        criterion = models.GCPLLoss(threshold=config.threshold, gamma=config.gamma, lambda_=config.lambda_)
    elif config.loss_type == 'pdce':
        criterion = models.PairwiseDCELoss(threshold=config.threshold, tao=config.tao, b=config.b, lambda_=config.lambda_)
    else:
        raise RuntimeError('Cannot find "{}" loss type.'.format(config.loss_type))

    sgd = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)
    # adam = optim.Adam(net.parameters(), lr=config.learning_rate)

    # load saved optim
    # if os.path.exists(config.model_path):
    #     state_dict = torch.load(config.optim_path)
    #     try:
    #         net.load_state_dict(state_dict)
    #         logger.info("Load optim from file '%s'.", config.optim_path)
    #     except RuntimeError:
    #         logger.error("Loading optim from file '%s' failed.", config.optim_path)

    # load saved model state dict
    if os.path.exists(config.model_path):
        state_dict = torch.load(config.model_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load model from file '%s'.", config.model_path)
        except RuntimeError:
            logger.error("Loading model from file '%s' failed.", config.model_path)

    # load saved prototypes
    if os.path.exists(config.prototypes_path):
        try:
            criterion.load_prototypes(config.prototypes_path)
            logger.info("Load prototypes from file '%s'.", config.prototypes_path)
        except RuntimeError:
            logger.error("Loading prototypes from file '%s' failed.", config.prototypes_path)

    for epoch in range(config.epoch_number):

        intra_class_distances = []

        # train
        if not config.testonly:
            logger.info("Epoch number: %d", epoch + 1)
            logger.info("threshold: %.4f, gamma: %.4f, tao: %.4f, b: %.4f",
                        config.threshold, config.gamma, config.tao, config.b)

            running_loss = 0.0
            distance_sum = 0.0

            if len(criterion.prototypes) > len(trainset.label_set):
                criterion.clear_prototypes()
            else:
                criterion.upgrade_prototypes()

            for i, (feature, label) in enumerate(trainloader):
                feature, label = feature.to(net.device), label.item()
                sgd.zero_grad()
                feature = net(feature).view(1, -1)
                loss, distance = criterion(feature, label)
                loss.backward()
                sgd.step()

                running_loss += loss.item()

                distance_sum += distance
                intra_class_distances.append((label, distance))

                logger.debug("[%d, %d] %7.4f, %7.4f", epoch + 1, i + 1, loss.item(), distance)

            distances = np.array(intra_class_distances, dtype=[
                ('label', np.int32),
                ('distance', np.float32)
            ])
            average_distance = np.average(distances['distance']).item()
            std_distance = distances['distance'].std().item()

            config.threshold = (average_distance + 3 * std_distance)
            config.gamma = 2 / average_distance
            config.tao = average_distance + std_distance
            config.b = std_distance

            criterion.set_threshold(config.threshold)
            criterion.set_gamma(config.gamma)
            criterion.set_tao(config.tao)
            criterion.set_b(config.b)

            torch.save(net.state_dict(), config.model_path)
            criterion.save_prototypes(config.prototypes_path)
            torch.save(intra_class_distances, config.intra_class_distances_path)

            logger.info("Prototypes Count: %d", len(criterion.prototypes))
        # end train

        # test
        if config.testonly:
            # load saved intra class distances
            if os.path.exists(config.intra_class_distances_path):
                try:
                    intra_class_distances = torch.load(config.intra_class_distances_path)
                    logger.info("Load intra class distances from file '%s'.", config.intra_class_distances_path)
                except RuntimeError:
                    logger.error("Loading intra class distances from file '%s' failed.", config.intra_class_distances_path)

        detector = models.Detector(intra_class_distances, config.std_coefficient, trainset.label_set)
        logger.info("Distance Average: %s", detector.average_distances)
        logger.info("Distance Std: %s", detector.std_distances)
        logger.info("Distance Threshold: %s", detector.thresholds)

        if (epoch + 1) % config.testfreq == 0 or config.testonly:

            detection_results = []

            for i, (feature, label) in enumerate(testloader):
                feature, label = net(feature.to(net.device)).view(1, -1), label.item()
                predicted_label, probability, distance = criterion.predict(feature)
                detected_novelty = detector(predicted_label, probability, distance)
                real_novelty = label not in trainset.label_set

                detection_results.append((label, predicted_label, probability, distance, real_novelty, detected_novelty))

                logger.debug("%5d: %d, %d, %7.4f, %7.4f, %s, %s",
                             i + 1, label, predicted_label, probability, distance, real_novelty, detected_novelty)

            true_positive, false_positive, false_negative = detector.evaluate(detection_results)

            precision = true_positive / (true_positive + false_positive + 1)
            recall = true_positive / (true_positive + false_negative + 1)
            cm = confusion_matrix(detector.results['true_label'], detector.results['predicted_label'], sorted(list(testset.label_set)))
            results = detector.results[np.isin(detector.results['true_label'], list(trainset.label_set))]

            logger.info("Accuracy: %7.4f", accuracy_score(results['true_label'], results['predicted_label']))
            # logger.info("Accuracy: %7.4f", accuracy_score(detector.results['true_label'], detector.results['predicted_label']))
            logger.info("True Positive: %d", true_positive)
            logger.info("False Positive: %d", false_positive)
            logger.info("False Negative: %d", false_negative)
            logger.info("Precision: %7.4f", precision)
            logger.info("Recall: %7.4f", recall)
            logger.info("Confusion Matrix: \n%s", cm)
        # end test


def run_cel(config, trainset, testset):
    logger = logging.getLogger(__name__)

    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(config.device)

    net = models.DenseNet(device=device, in_channels=config.in_channels, number_layers=config.number_layers, growth_rate=12, drop_rate=0.0)
    logger.info("DenseNet Channels: %d", net.channels)
    fc_net = models.LinearNet(device=device, in_features=net.channels * (config.tensor_view[1] // 8) * (config.tensor_view[2] // 8))

    cel = nn.CrossEntropyLoss()
    sgd = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

    # load saved model state dict
    if os.path.exists(config.model_path):
        state_dict = torch.load(config.model_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load state from file '%s'.", config.model_path)
        except RuntimeError:
            logger.error("Loading state from file '%s' failed.", config.model_path)

    for epoch in range(config.epoch_number):
        logger.info("Epoch number: %d", epoch + 1)

        probs = []

        # train
        if not config.testonly:
            running_loss = 0.0

            for i, (feature, label) in enumerate(trainloader):
                feature, label = feature.to(net.device), label.to(net.device)
                sgd.zero_grad()
                feature = net(feature).view(1, -1)
                feature = fc_net(feature)
                loss = cel(feature, label)
                loss.backward()
                sgd.step()

                feature = feature.data.squeeze()

                running_loss += loss.item()

                probs.append((label.item(), feature[label.item()]))
                logger.debug("[%d, %d] %7.4f", epoch + 1, i + 1, loss.item())

            torch.save(net.state_dict(), config.model_path)
            torch.save(probs, config.probs_path)
        # end train

        # load saved probs
        if config.testonly:
            if os.path.exists(config.probs_path):
                try:
                    probs = torch.load(config.probs_path)
                except RuntimeError:
                    logger.error("Loading probs from file '%s' failed.", config.probs_path)
                else:
                    logger.info("Load probs from file '%s'.", config.probs_path)

        # test
        detector = models.SoftmaxDetector(probs, config.std_coefficient, trainset.label_set)

        if (epoch + 1) % config.testfreq == 0 or config.testonly:

            detection_results = []
            with torch.no_grad():
                for i, (feature, label) in enumerate(testloader):
                    feature, label = feature.to(net.device), label.item()
                    feature = net(feature).view(1, -1)
                    probability, predicted_label = fc_net(feature).max(dim=1)
                    detected_novelty = detector(predicted_label, probability)
                    real_novelty = label not in trainset.label_set

                    detection_results.append((label, predicted_label.item(), probability, real_novelty, detected_novelty))
                    logger.debug("%5d: %d, %d, %7.4f, %s, %s", i + 1, label, predicted_label, probability, real_novelty, detected_novelty)

            true_positive, false_positive, false_negative = detector.evaluate(detection_results)
            precision = true_positive / (true_positive + false_positive + 1)
            recall = true_positive / (true_positive + false_negative + 1)
            cm = confusion_matrix(detector.results['true_label'], detector.results['predicted_label'], sorted(list(testset.label_set)))
            results = detector.results[np.isin(detector.results['true_label'], list(trainset.label_set))]

            logger.info("Accuracy: %7.4f", accuracy_score(results['true_label'], results['predicted_label']))
            logger.info("True Positive: %d", true_positive)
            logger.info("False Positive: %d", false_positive)
            logger.info("False Negative: %d", false_negative)
            logger.info("Precision: %7.4f", precision)
            logger.info("Recall: %7.4f", recall)
            logger.info("Confusion Matrix: \n%s", cm)
        # end test


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dir', type=str, help="Running directory path.", required=True)
    parser.add_argument('-e', '--epoch', type=int, help="Train epoch number.", default=1)
    parser.add_argument('-f', '--testfreq', type=int, help="Test frequency.", default=1)
    parser.add_argument('-t', '--testonly', help="Test only.", action="store_true")
    parser.add_argument('-c', '--clear', help="Clear running path.", action="store_true")
    parser.add_argument('-v', '--visualize', help="Visualization.", action="store_true")

    args = parser.parse_args()

    if not os.path.isdir(args.dir):
        raise RuntimeError("Config path not found!")

    config_file = "{}/config.json".format(args.dir)

    with open(config_file) as file:
        config_dict = json.load(file)

    config = Config(**config_dict)
    config.running_path = args.dir
    config.log_path = os.path.join(config.running_path, "run.log")
    config.model_path = os.path.join(config.running_path, "model.pkl")
    config.prototypes_path = os.path.join(config.running_path, "prototypes.pkl")
    config.intra_class_distances_path = os.path.join(config.running_path, "distances.pkl")
    config.probs_path = os.path.join(config.running_path, "probs.pkl")
    config.epoch_number = args.epoch
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.testonly = args.testonly

    if args.testfreq > 0:
        config.testfreq = args.testfreq

    if args.clear:
        if input("Do you really want to clear the running directory? (y/[n]) ") == 'y':
            try:
                os.remove(config.log_path)
                os.remove(config.model_path)
                os.remove(config.prototypes_path)
                os.remove(config.intra_class_distances_path)
                os.remove(config.probs_path)
                # os.remove("{}/config_dump.json".format(args.dir))
            except FileNotFoundError:
                pass

    if config.testonly:
        config.epoch_number = 1

    logger = setup_logger(level=logging.DEBUG, filename=config.log_path)

    # dataset = np.loadtxt(config.dataset_path, delimiter=',')
    dataset = pd.read_csv(config.dataset_path, sep=',', header=None).values
    train_dataset = dataset[:config.train_test_split]
    test_dataset = dataset[config.train_test_split:]
    random.shuffle(train_dataset)
    random.shuffle(test_dataset)

    # mean = [train_dataset[:, i:-1:config.in_channels].mean() for i in range(config.in_channels)]
    # std = [train_dataset[:, i:-1:config.in_channels].std() for i in range(config.in_channels)]
    #
    # transform = torchvision.transforms.Normalize(mean=mean, std=std)

    trainset = DataSet(dataset[:config.train_test_split], config.tensor_view)
    testset = DataSet(dataset[config.train_test_split:], config.tensor_view)

    # import torchvision
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # trainset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=True, transform=transform, download=False)
    # testset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=False, transform=transform, download=False)

    logger.info("********************************************************************************")
    logger.info("%s", config)
    logger.info("Trainset size: %d", len(trainset))
    logger.info("Testset size: %d", len(testset))

    if args.visualize:
        visualization(config, trainset)
    else:
        if config.loss_type == 'cel':
            run_cel(config, trainset, testset)
        else:
            run(config, trainset, testset)

    with open(config_file, mode='w') as file:
        config_dict['learning_rate'] = config.learning_rate / 2
        config_dict['threshold'] = config.threshold
        config_dict['gamma'] = config.gamma
        config_dict['tao'] = config.tao
        config_dict['b'] = config.b
        json.dump(config_dict, file)

    logger.info("---------- %.3fs ----------", time.time() - start_time)


def visualization(config, trainset):
    logger = logging.getLogger(__name__)

    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=False, num_workers=4)

    # net = models.CNNNet(device=device)
    net = models.DenseNet(device=torch.device(config.device),
                          in_channels=config.in_channels,
                          number_layers=config.number_layers,
                          growth_rate=config.growth_rate,
                          drop_rate=0.0)
    logger.info("DenseNet Channels: %d", net.channels)

    if config.loss_type == 'gcpl':
        criterion = models.GCPLLoss(threshold=config.threshold, gamma=config.gamma, lambda_=config.lambda_)
    elif config.loss_type == 'pdce':
        criterion = models.PairwiseDCELoss(threshold=config.threshold, tao=config.tao, b=config.b, lambda_=config.lambda_)
    else:
        raise RuntimeError('Cannot find "{}" loss type.'.format(config.loss_type))

    # load saved model state dict
    if os.path.exists(config.model_path):
        state_dict = torch.load(config.model_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load model from file '%s'.", config.model_path)
        except RuntimeError:
            logger.error("Loading model from file '%s' failed.", config.model_path)

    # load saved prototypes
    if os.path.exists(config.prototypes_path):
        try:
            criterion.load_prototypes(config.prototypes_path)
            logger.info("Load prototypes from file '%s'.", config.prototypes_path)
        except RuntimeError:
            logger.error("Loading prototypes from file '%s' failed.", config.prototypes_path)

    # original_features = []
    features = []
    labels = []

    for i, (feature, label) in enumerate(trainloader):
        feature, label = net(feature.to(net.device)).view(-1), label.item()

        features.append(feature.data.cpu().numpy())
        labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    feature_tsne = TSNE(n_components=2, random_state=30)
    features = feature_tsne.fit_transform(features, labels)
    # original_features = feature_tsne.fit_transform(original_features, labels)

    plt.figure(figsize=(6, 4))
    colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'gray', 'orange', 'purple'
    for c, label in zip(colors, sorted(list(trainset.label_set))):
        # print(c, label)
        plt.scatter(features[labels == label, 0], features[labels == label, 1], c=c, label=label)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
