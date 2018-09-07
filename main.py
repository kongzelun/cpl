import os
import logging
import numpy as np
import torch
import torch.nn as nn
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
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message) s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def train(net, dataloader, criterion, optimizer, all_prototypes):
    logger = logging.getLogger(__name__)

    for i, (feature, label) in enumerate(dataloader):
        feature, label = feature.to(net.device), label.to(net.device)

        # extract abstract feature through CNN.
        feature = net(feature)

        assign_prototype(feature, label)

        optimizer.zero_grad()

        loss = criterion(feature, label, all_prototypes)


def assign_prototype(feature, label, all_prototypes):
    distance = nn.PairwiseDistance(p=2)
    if label not in all_prototypes:
        all_prototypes[label] = []
        prototype = functions.Prototype(label)
        prototype.update(feature)
        all_prototypes[label].append(prototype)
    else:
        minimum_distance = all_prototypes['threshold']
        closest_prototype = None
        has_prototype = False

        for p in all_prototypes[label]:
            d = distance(feature, p.feature)
            if d < minimum_distance:
                closest_prototype = p
                minimum_distance = d
                has_prototype = True

        if has_prototype:
            closest_prototype.update(feature)
        else:
            prototype = functions.Prototype(label)
            prototype.update(feature)
            all_prototypes[label].append(prototype)


if __name__ == '__main__':
    LOGGER = setup_logger(level=logging.DEBUG)

    TRAIN_EPOCH_NUMBER = 100

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    DATASET = np.loadtxt(nets.DATASET_PATH, delimiter=',')

    TRAINSET = nets.CPLDataset(DATASET)
    TRAINLOADER = DataLoader(dataset=TRAINSET, batch_size=1, shuffle=True, num_workers=2)

    TESTSET = nets.CPLDataset(DATASET)
    TESTLOADER = DataLoader(dataset=TESTSET, batch_size=1, shuffle=False, num_workers=2)

    PROTOTYPES = {'threshold': 1.0}

    net = nets.CPLNet(device=DEVICE)
    dce = functions.DCE(gamma=1.0)
    sgd = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if os.path.exists(nets.PKL_PATH):
        state_dict = torch.load(nets.PKL_PATH)
        try:
            net.load_state_dict(state_dict)
            LOGGER.info("Load state from file %s.", nets.PKL_PATH)
        except RuntimeError:
            LOGGER.error("Loading state from file %s failed.", nets.PKL_PATH)
