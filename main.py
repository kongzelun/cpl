import os
import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import models
from sklearn.metrics import accuracy_score, confusion_matrix


class Config(object):
    # dataset_path = 'data/fashion-mnist_train.csv'
    # pkl_path = "pkl/fashion-mnist.pkl"
    # tensor_view = (-1, 28, 28)
    # in_channels = 1
    # dataset_path = "data/cifar10_train.csv"
    dataset_path = None
    running_path = None
    loss_type = None

    tensor_view = None
    in_channels = None
    layers = None

    learning_rate = None

    threshold = None
    gamma = None

    tao = None
    b = None

    lambda_ = None

    std_coefficient = None

    epoch_number = 1
    test_frequency = 1
    train_test_split = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.values = kwargs

    def __repr__(self):
        return "{}".format(self.values)


def setup_logger(level=logging.DEBUG, filename=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def run_cpl(config, device, trainset, testset, criterion):
    logger = logging.getLogger(__name__)

    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=0)

    # net = models.CNNNet(device=device)
    net = models.DenseNet(device=device, in_channels=config.in_channels, number_layers=config.layers, growth_rate=12, drop_rate=0.0)
    logger.info("DenseNet Channels: %d\n", net.channels)

    sgd = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

    pkl_path = os.path.join(config.running_path, "{}.pkl".format(config.running_path))
    if os.path.exists(pkl_path):
        state_dict = torch.load(pkl_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load state from file %s.", pkl_path)
        except RuntimeError:
            logger.error("Loading state from file %s failed.", pkl_path)

    # train
    for epoch in range(config.epoch_number):
        logger.info("Epoch number: %d", epoch + 1)

        # train
        criterion.clear()

        running_loss = 0.0
        intra_class_distances = []

        for i, (feature, label) in enumerate(trainloader):
            feature, label = feature.to(net.device), label.item()
            sgd.zero_grad()
            feature = net(feature).view(1, -1)
            loss, distance = criterion(feature, label)
            loss.backward()
            sgd.step()

            running_loss += loss.item()

            intra_class_distances.append((label, distance))

            logger.debug("[%d, %d] %7.4f, %7.4f", epoch + 1, i + 1, loss.item(), distance)

        torch.save(net.state_dict(), pkl_path)

        detector = models.Detector(intra_class_distances, config.std_coefficient, trainset.label_set)

        logger.info("Prototypes Count: %d", len(criterion.prototypes))

        logger.info("Distance Average: %s", detector.average_distances)
        logger.info("Distance Std: %s", detector.std_distances)
        logger.info("Distance Threshold: %s", detector.thresholds)

        # test
        if (epoch + 1) % config.test_frequency == 0:

            detection_results = []

            for j, (feature, label) in enumerate(testloader):
                feature, label = net(feature.to(net.device)).view(1, -1), label.item()
                predicted_label, probability, distance = criterion.predict(feature)
                novelty = detector(predicted_label, probability, distance)

                detection_results.append((label, predicted_label, probability, distance, novelty))

                logger.debug("%5d: %d, %d, %7.4f, %7.4f, %s", j + 1, label, predicted_label, probability, distance, novelty)

            precision, recall = detector.evaluate(detection_results)

            cm = confusion_matrix(detector.results['true label'], detector.results['predicted label'], sorted(list(testset.label_set)))

            logger.info("Accuracy: %7.4f", accuracy_score(detector.results['true label'], detector.results['predicted label']))
            logger.info("Precision: %7.4f", precision)
            logger.info("Recall: %7.4f", recall)
            logger.info("Confusion Matrix: \n%s\n", cm)


def run_cel(config, device, trainset, testset):
    logger = logging.getLogger(__name__)

    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=0)

    # net = models.CNNNet(device=device)
    net = models.DenseNet(device=device, in_channels=config.in_channels, number_layers=config.layers, growth_rate=12, drop_rate=0.0)
    logger.info("DenseNet Channels: %d\n", net.channels)
    fc_net = models.LinearNet(device=device, in_features=net.channels * (config.tensor_view[1] // 8) * (config.tensor_view[2] // 8))

    cel = nn.CrossEntropyLoss()
    sgd = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

    pkl_path = os.path.join(config.running_path, "{}.pkl".format(config.running_path))
    if os.path.exists(pkl_path):
        state_dict = torch.load(pkl_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load state from file %s.", config.pkl_path)
        except RuntimeError:
            logger.error("Loading state from file %s failed.", config.pkl_path)

    for epoch in range(config.epoch_number):
        logger.info("Epoch number: %d", epoch + 1)

        # train
        running_loss = 0.0

        for i, (feature, label) in enumerate(trainloader):
            feature, label = feature.to(net.device), label.to(net.device)
            sgd.zero_grad()
            feature = net(feature).view(1, -1)
            feature = fc_net(feature)
            loss = cel(feature, label)
            loss.backward()
            sgd.step()

            running_loss += loss.item()

            logger.debug("[%d, %d] %7.4f", epoch + 1, i + 1, loss.item())

        torch.save(net.state_dict(), pkl_path)

        # test
        if (epoch + 1) % config.test_frequency == 0:

            labels_true = []
            labels_predicted = []

            for j, (feature, label) in enumerate(testloader):
                feature, label = feature.to(net.device), label.item()
                feature = net(feature).view(1, -1)
                _, predicted_label = fc_net(feature).max(dim=1)

                labels_true.append(label)
                labels_predicted.append(predicted_label)

                logger.debug("%5d: %d, %d", j + 1, label, predicted_label)

            cm = confusion_matrix(labels_true, labels_predicted, sorted(list(testset.label_set)))

            logger.info("Accuracy: %7.4f", accuracy_score(labels_true, labels_predicted))
            logger.info("Confusion Matrix: \n%s\n", cm)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, help="Config directory path.", required=True)
    parser.add_argument('--clear', help="Clear running path.", action="store_true")
    parser.add_argument('-e', '--epoch', type=int, help="Train epoch number.", default=None)

    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise RuntimeError("Config path not found!")

    with open("{}/{}.json".format(args.config, args.config)) as config_file:
        config = Config(**json.load(config_file))
        config.running_path = args.config
        log_path = os.path.join(config.running_path, "{}.log".format(config.running_path))
        pkl_path = os.path.join(config.running_path, "{}.pkl".format(config.running_path))

    if args.epoch:
        config.epoch_number = args.epoch

    if args.clear:
        try:
            os.remove(log_path)
            os.remove(pkl_path)
        except FileNotFoundError:
            pass

    logger = setup_logger(level=logging.DEBUG, filename=log_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = np.loadtxt(config.dataset_path, delimiter=',')
    np.random.shuffle(dataset[:config.train_test_split])
    np.random.shuffle(dataset[config.train_test_split:])

    trainset = models.DataSet(dataset[:config.train_test_split], config.tensor_view)
    testset = models.DataSet(dataset[config.train_test_split:], config.tensor_view)

    logger.info("\n%s", config)
    logger.info("Trainset size: %d", len(trainset))
    logger.info("Testset size: %d", len(testset))

    if config.loss_type == 'gcpl':
        gcpl = models.GCPLLoss(threshold=config.threshold, gamma=config.gamma, lambda_=config.lambda_)
        run_cpl(config, device, trainset, testset, gcpl)
    elif config.loss_type == 'pairwise_dce':
        pwdce = models.PairwiseDCELoss(threshold=config.threshold, gamma=config.gamma, tao=config.tao, b=config.b, beta=0.5, lambda_=config.lambda_)
        run_cpl(config, device, trainset, testset, pwdce)
    elif config.loss_type == 'cel':
        run_cel(config, device, trainset, testset)
    else:
        raise RuntimeError('Cannot find "{}" loss type.'.format(config.loss_type))


if __name__ == '__main__':
    main()
