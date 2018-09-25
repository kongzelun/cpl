import os
import argparse
import json
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import models


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


def train(config):
    logger = logging.getLogger(__name__)

    logger.info("%s", config)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = np.loadtxt(config.dataset_path, delimiter=',')
    np.random.shuffle(dataset[:config.train_test_split])
    np.random.shuffle(dataset[config.train_test_split:])

    trainset = models.DataSet(dataset[:config.train_test_split], config.tensor_view)
    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=2)

    testset = models.DataSet(dataset[config.train_test_split:], config.tensor_view)
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=2)

    # net = models.CNNNet(device=device)
    net = models.DenseNet(device=device, in_channels=config.in_channels, number_layers=8, growth_rate=12, drop_rate=0.0)
    logger.info("DenseNet Channels: %d", net.channels)

    prototypes = {}

    # cel = torch.nn.CrossEntropyLoss()
    gcpl = models.GCPLLoss(threshold=config.threshold, gamma=config.gamma, tao=config.tao, b=1.0, beta=0.5, lambda_=config.lambda_)
    sgd = optim.SGD(net.parameters(), lr=config.learning_rate, momentum=0.9)

    if os.path.exists(config.pkl_path):
        state_dict = torch.load(config.pkl_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load state from file %s.", config.pkl_path)
        except RuntimeError:
            logger.error("Loading state from file %s failed.", config.pkl_path)

    for epoch in range(config.epoch_number):
        logger.info("\nEpoch number: %d", epoch + 1)
        logger.info("Trainset size: %d", len(trainset))
        logger.info("%7.4f %7.4f %7.4f %7.4f", gcpl.threshold, gcpl.gamma, gcpl.tao, gcpl.lambda_)

        # train
        prototypes.clear()

        running_loss = 0.0
        distance_sum = 0.0

        for i, (feature, label) in enumerate(trainloader):
            feature = feature.to(net.device)
            sgd.zero_grad()
            feature = net(feature).view(1, -1)
            loss, min_distance = gcpl(feature, label.item(), prototypes)
            loss.backward()
            sgd.step()

            running_loss += loss.item()
            distance_sum += min_distance

            logger.debug("[%d, %d] %7.4f, %7.4f", epoch + 1, i + 1, loss.item(), min_distance)

        torch.save(net.state_dict(), config.pkl_path)

        average_distance = distance_sum / len(trainloader)

        gcpl.threshold = average_distance * 2
        gcpl.tao = average_distance * 2

        logger.info("Distance Average: %7.4f", average_distance)

        # count prototypes
        prototype_count = 0

        for c in prototypes:
            prototype_count += len(prototypes[c])

        logger.info("Prototypes Count: %d", prototype_count)

        # test
        if (epoch + 1) % config.test_frequency == 0:
            logger.info("Testset size: %d", len(testset))

            correct = 0

            for i, (feature, label) in enumerate(testloader):
                feature = net(feature.to(net.device)).view(1, -1)
                predicted_label, probability, min_distance = gcpl.predict(feature, prototypes)

                if label == predicted_label:
                    correct += 1

                logger.debug("%5d: %d, %d, %7.4f, %7.4f, %7.4f",
                             i + 1, label, predicted_label, probability, min_distance, correct / (i + 1))

            logger.info("Accuracy: %7.4f\n", correct / len(testloader))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--config', type=str, help="Config file path.", required=True)
    parser.add_argument('-e', '--epoch', type=int, help="Train epoch number.", default=None)

    args = parser.parse_args()

    if not os.path.exists("pkl"):
        os.mkdir("pkl")

    if not os.path.exists("config"):
        os.mkdir("config")

    with open("config/{}".format(args.config)) as config_file:
        config = models.Config(**json.load(config_file))

    if args.epoch:
        config.epoch_number = args.epoch

    # try:
    #     config_file = open("config/{}".format(args.config))
    #     config = models.Config(**json.load(config_file))
    # except FileNotFoundError:
    #     print("Can't find config file.")
    # finally:
    #     config_file.close()

    setup_logger(level=logging.DEBUG, filename=config.log_path)
    train(config)


if __name__ == '__main__':
    main()
