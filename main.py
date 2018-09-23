import os
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
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename=filename, mode='a')
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


def main():
    logger = setup_logger(level=logging.DEBUG, filename='log.txt')

    train_epoch_number = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = np.loadtxt(models.Config.dataset_path, delimiter=',')
    np.random.shuffle(dataset[:5000])
    np.random.shuffle(dataset[5000:])

    trainset = models.DataSet(dataset[:5000])
    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=24)

    testset = models.DataSet(dataset[5000:15000])
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=2)

    # net = models.CNNNet(device=device)
    net = models.DenseNet(device=device, number_layers=8, growth_rate=12, drop_rate=0.0)
    logger.info("DenseNet Channels: %d", net.channels)

    prototypes = {}

    # cel = torch.nn.CrossEntropyLoss()
    gcpl = models.GCPLLoss(threshold=models.Config.threshold, gamma=models.Config.gamma, tao=models.Config.tao, b=1.0, beta=0.5, lambda_=models.Config.lambda_)
    sgd = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if not os.path.exists("pkl"):
        os.mkdir("pkl")

    if os.path.exists(models.Config.pkl_path):
        state_dict = torch.load(models.Config.pkl_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load state from file %s.", models.Config.pkl_path)
        except RuntimeError:
            logger.error("Loading state from file %s failed.", models.Config.pkl_path)

    for epoch in range(train_epoch_number):
        logger.info("Trainset size: %d, Epoch number: %d, Threshold: %7.4f", len(trainset), epoch + 1, gcpl.threshold)

        prototypes.clear()

        running_loss = 0.0

        for i, (feature, label) in enumerate(trainloader):
            feature = feature.to(net.device)
            sgd.zero_grad()
            feature = net(feature).view(1, -1)
            loss, min_distance = gcpl(feature, label.item(), prototypes)
            loss.backward()
            sgd.step()

            running_loss += loss.item()

            logger.debug("[%d, %d] %7.4f, %7.4f", epoch + 1, i + 1, loss.item(), min_distance)

        torch.save(net.state_dict(), models.Config.pkl_path)

        prototype_count = 0

        for c in prototypes:
            prototype_count += len(prototypes[c])

        logger.info("Prototypes Count: %d", prototype_count)

        # if (epoch + 1) % 5 == 0:
        distance_sum = 0.0
        correct = 0

        for i, (feature, label) in enumerate(testloader):
            feature = net(feature.to(net.device)).view(1, -1)
            predicted_label, probability, min_distance = models.predict(feature, prototypes)

            if label == predicted_label:
                correct += 1

            distance_sum += min_distance

            logger.debug("%5d: %d, %d, %7.4f, %7.4f, %7.4f",
                         i + 1, label, predicted_label, probability, min_distance, correct / (i + 1))

        average_distance = distance_sum / len(testloader)

        logger.info("Distance Average: %7.4f", average_distance)
        logger.info("Accuracy: %7.4f\n", correct / len(testloader))
        gcpl.threshold = average_distance * 2
        gcpl.tao = average_distance * 2


if __name__ == '__main__':
    main()
