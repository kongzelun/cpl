import os
import argparse
import json
import logging
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import models
from sklearn.metrics import accuracy_score, confusion_matrix


class Config(object):
    # read from json
    # train = True
    testonly = False

    dataset_path = None
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
    testfreq = 1
    train_test_split = None

    # derived
    running_path = None
    log_path = None
    model_path = None
    prototypes_path = None
    intra_class_distances_path = None

    device = "cpu"

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

    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=0)

    # net = models.CNNNet(device=device)
    net = models.DenseNet(device=torch.device(config.device), in_channels=config.in_channels, number_layers=config.layers, growth_rate=12, drop_rate=0.0)
    logger.info("DenseNet Channels: %d", net.channels)

    if config.loss_type == 'gcpl':
        criterion = models.GCPLLoss(threshold=config.threshold, gamma=config.gamma, lambda_=config.lambda_)
    elif config.loss_type == 'pdce':
        criterion = models.PairwiseDCELoss(threshold=config.threshold, tao=config.tao, b=config.b, lambda_=config.lambda_)
    else:
        raise RuntimeError('Cannot find "{}" loss type.'.format(config.loss_type))

    sgd = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

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

            distances = np.array(intra_class_distances, dtype=[('label', np.int32), ('distance', np.float32)])
            average_distance = np.average(distances['distance']).item()
            std_distance = distances['distance'].std().item()

            config.threshold = (average_distance + 4 * std_distance)
            config.gamma = 1 / average_distance
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

        # test
        if not config.testonly:
            # load saved intra class distances
            if os.path.exists(config.intra_class_distances_path):
                try:
                    intra_class_distances = torch.load(config.intra_class_distances_path)
                    logger.info("Load intra class distances from file '%s'.", config.intra_class_distances_path)
                except RuntimeError:
                    logger.error("Loading prototypes from file '%s' failed.", config.intra_class_distances_path)

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

            results = detector.results[np.isin(detector.results['true_label'], list(testset.label_set))]
            logger.info("Accuracy: %7.4f", accuracy_score(results['true_label'], results['predicted_label']))
            logger.info("True Positive: %d", true_positive)
            logger.info("False Positive: %d", false_positive)
            logger.info("False Negative: %d", false_negative)
            logger.info("Precision: %7.4f", precision)
            logger.info("Recall: %7.4f", recall)
            logger.info("Confusion Matrix: \n%s", cm)


def run_cel(config, trainset, testset):
    logger = logging.getLogger(__name__)

    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=0)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device(config.device)

    net = models.DenseNet(device=device, in_channels=config.in_channels, number_layers=config.layers, growth_rate=12, drop_rate=0.0)
    logger.info("DenseNet Channels: %d", net.channels)
    fc_net = models.LinearNet(device=device, in_features=net.channels * (config.tensor_view[1] // 8) * (config.tensor_view[2] // 8))

    cel = nn.CrossEntropyLoss()
    sgd = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

    # load saved model state dict
    if os.path.exists(config.model_path):
        state_dict = torch.load(config.model_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load state from file '%s'.", config.pkl_path)
        except RuntimeError:
            logger.error("Loading state from file '%s' failed.", config.pkl_path)

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

        torch.save(net.state_dict(), config.model_path)

        # test
        if (epoch + 1) % config.testfreq == 0 or config.testonly:

            labels_true = []
            labels_predicted = []

            for i, (feature, label) in enumerate(testloader):
                feature, label = feature.to(net.device), label.item()
                feature = net(feature).view(1, -1)
                _, predicted_label = fc_net(feature).max(dim=1)

                labels_true.append(label)
                labels_predicted.append(predicted_label)

                logger.debug("%5d: %d, %d", i + 1, label, predicted_label)

            cm = confusion_matrix(labels_true, labels_predicted, sorted(list(testset.label_set)))

            logger.info("Accuracy: %7.4f", accuracy_score(labels_true, labels_predicted))
            logger.info("Confusion Matrix: \n%s", cm)


def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--dir', type=str, help="Running directory path.", required=True)
    parser.add_argument('-e', '--epoch', type=int, help="Train epoch number.", default=1)
    parser.add_argument('-f', '--testfreq', type=int, help="Test frequency.", default=1)
    parser.add_argument('-t', '--testonly', help="Test only.", action="store_true")
    parser.add_argument('-c', '--clear', help="Clear running path.", action="store_true")

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
    config.epoch_number = args.epoch
    config.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if args.testonly:
        config.epoch_number = 1

    if args.clear:
        if input("Do you really want to clear the running directory? (y/[n])") == 'y':
            try:
                os.remove(config.log_path)
                os.remove(config.model_path)
                os.remove(config.prototypes_path)
                os.remove(config.intra_class_distances_path)
                # os.remove("{}/config_dump.json".format(args.dir))
            except FileNotFoundError:
                pass

    logger = setup_logger(level=logging.DEBUG, filename=config.log_path)

    dataset = np.loadtxt(config.dataset_path, delimiter=',')
    np.random.shuffle(dataset[:config.train_test_split])
    np.random.shuffle(dataset[config.train_test_split:])

    trainset = models.DataSet(dataset[:config.train_test_split], config.tensor_view)
    testset = models.DataSet(dataset[config.train_test_split:], config.tensor_view)

    # import torchvision
    # transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    # trainset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=True, transform=transform, download=False)
    # testset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=False, transform=transform, download=False)

    logger.info("********************************************************************************")
    logger.info("%s", config)
    logger.info("Trainset size: %d", len(trainset))
    logger.info("Testset size: %d", len(testset))

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


if __name__ == '__main__':
    main()
