import os
import logging
import numpy as np
import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import DataLoader
import fashion_minst as nets
import functions


def setup_logger(level=logging.DEBUG):
    """
    Setup logger.
    -------------
    :param level:
    :return: logger
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def train(net, dataloader, criterion, optimizer, all_prototypes):
    logger = logging.getLogger(__name__)

    for i, (feature, label) in enumerate(dataloader):
        feature, label = feature.to(net.device), int(label)

        optimizer.zero_grad()

        # extract abstract feature through CNN.
        feature = net(feature).view(1, -1)

        closest_prototype = functions.assign_prototype(tensor(feature.data), label, all_prototypes, tensor(nets.THRESHOLD).to(net.device))

        loss = criterion(feature, label, all_prototypes, closest_prototype)
        loss.backward()
        optimizer.step()

        distance = functions.compute_distance(feature, closest_prototype.feature)

        logger.debug("%5d: Loss: %.4f, Distance: %.4f", i + 1, loss, distance)


def test(net, dataloader, all_prototypes, gamma):
    logger = logging.getLogger(__name__)

    correct = 0

    for i, (feature, label) in enumerate(dataloader):
        feature, label = feature.to(net.device), int(label)

        # extract abstract feature through CNN.
        feature = net(feature).view(1, -1)

        predicted_label, probability = predict(feature, all_prototypes, gamma)

        if label == predicted_label:
            correct += 1

        logger.debug("%5d: Label: %d, Prediction: %d, Probability: %.4f, Accuracy: %.4f",
                     i + 1, label, predicted_label, probability, correct / (i + 1))

    return correct / len(dataloader)


def predict(feature, all_prototypes, gamma):
    probabilities = {}

    for label in all_prototypes:
        probability = functions.compute_probability(feature, label, all_prototypes, gamma)
        probabilities[label] = probability

    predicted_label = max(probabilities, key=probabilities.get)

    return predicted_label, probabilities[predicted_label]


if __name__ == '__main__':
    LOGGER = setup_logger(level=logging.DEBUG)

    TRAIN_EPOCH_NUMBER = 1

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    DATASET = np.loadtxt(nets.DATASET_PATH, delimiter=',')

    TRAINSET = nets.CPLDataset(DATASET[:10000])
    TRAINLOADER = DataLoader(dataset=TRAINSET, batch_size=1, shuffle=True, num_workers=4)

    TESTSET = nets.CPLDataset(DATASET[10000:20000])
    TESTLOADER = DataLoader(dataset=TESTSET, batch_size=1, shuffle=False, num_workers=0)

    PROTOTYPES = {}

    cplnet = nets.CPLNet(device=DEVICE)
    gcpl = functions.GCPLLoss(gamma=1.0, lambda_=0.1)
    sgd = optim.SGD(cplnet.parameters(), lr=0.001, momentum=0.9)

    if not os.path.exists("pkl"):
        os.mkdir("pkl")

    if os.path.exists(nets.PKL_PATH):
        state_dict = torch.load(nets.PKL_PATH)
        try:
            cplnet.load_state_dict(state_dict)
            LOGGER.info("Load state from file %s.", nets.PKL_PATH)
        except RuntimeError:
            LOGGER.error("Loading state from file %s failed.", nets.PKL_PATH)

    for epoch in range(TRAIN_EPOCH_NUMBER):
        LOGGER.info("Trainset size: %d, Epoch number: %d", len(TRAINSET), epoch + 1)
        train(cplnet, TRAINLOADER, gcpl, sgd, PROTOTYPES)
        torch.save(cplnet.state_dict(), nets.PKL_PATH)

        accuracy = test(cplnet, TESTLOADER, PROTOTYPES, gcpl.gamma)

        prototype_count = 0

        for c in PROTOTYPES:
            for p in PROTOTYPES[c]:
                prototype_count += 1

        LOGGER.info("Accuracy: %.4f", accuracy)
        LOGGER.info("Prototype Count: %d", prototype_count)
