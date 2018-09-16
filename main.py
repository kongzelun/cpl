import os
import logging
import numpy as np
import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import DataLoader
import models
import functions


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


# def train(net, dataloader, criterion, optimizer):
#     logger = logging.getLogger(__name__)
#     loss_sum = 0.0
#
#     threshold = models.Config.threshold
#
#     all_prototypes = {}
#
#     # distances_sum = 0.0
#
#     for i, (feature, label) in enumerate(dataloader):
#         feature, label = feature.to(net.device), int(label)
#
#         optimizer.zero_grad()
#
#         # extract abstract feature through CNN.
#         feature = net(feature).view(1, -1)
#
#         closest_prototype_index, min_distance = functions.assign_prototype(tensor(feature.data), label, all_prototypes, threshold)
#
#         # if (i + 1) % 10 == 0:
#         #     threshold = distances_sum / 100
#         #     distances_sum = 0.0
#         # else:
#         #     distances_sum += min_distance
#
#         # distances_sum += min_distance
#         # threshold = distances_sum / (i + 1)
#
#         loss = criterion(feature, label, all_prototypes, all_prototypes[label][closest_prototype_index])
#
#         loss.backward()
#         optimizer.step()
#
#         loss_sum += loss
#
#         logger.debug("%5d: Loss: %6.4f, Distance: %6.4f, Threshold: %6.4f", i + 1, loss, min_distance, threshold)
#
#     logger.info("Loss Average: %6.4f", loss_sum / len(dataloader))
#
#     return all_prototypes

def train(net, dataloader, criterion, optimizer):
    logger = logging.getLogger(__name__)

    for i, (feature, label) in enumerate(dataloader):
        feature, label = feature.to(net.device), label.to(net.device)
        optimizer.zero_grad()
        out = net(feature).view(1, -1)
        loss = criterion(out, label)
        loss.backward()
        optimizer.step()

        logger.debug("%5d: Loss: %6.4f", i + 1, loss)


# def test(net, dataloader, all_prototypes, gamma):
#     logger = logging.getLogger(__name__)
#
#     correct = 0
#
#     for i, (feature, label) in enumerate(dataloader):
#         feature, label = feature.to(net.device), int(label)
#
#         # extract abstract feature through CNN.
#         feature = net(feature).view(1, -1)
#
#         predicted_label, probability = predict(feature, all_prototypes, gamma)
#
#         if label == predicted_label:
#             correct += 1
#
#         logger.debug("%5d: Label: %d, Prediction: %d, Probability: %.4f, Accuracy: %.4f",
#                      i + 1, label, predicted_label, probability, correct / (i + 1))
#
#     return correct / len(dataloader)

def test(net, dataloader):
    logger = logging.getLogger(__name__)

    correct = 0

    for i, (feature, label) in enumerate(dataloader):
        feature, label = feature.to(net.device), label.to(net.device)
        out = net(feature).view(1, -1)

        predicted_label = out.max(dim=1)[1]

        if label == predicted_label:
            correct += 1

        logger.debug("%5d: Label: %d, Prediction: %d, Accuracy: %.4f", i + 1, label, predicted_label, correct / (i + 1))

    return correct / len(dataloader)


def predict(feature, all_prototypes, gamma):
    probabilities = {}

    for label in all_prototypes:
        probability = functions.compute_probability(feature, label, all_prototypes, gamma)
        probabilities[label] = probability

    predicted_label = max(probabilities, key=probabilities.get)

    return predicted_label, probabilities[predicted_label]


def main():
    logger = setup_logger(level=logging.DEBUG, filename='log.txt')

    train_epoch_number = 100

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = np.loadtxt(models.Config.dataset_path, delimiter=',')

    trainset = models.FashionMnist(dataset[:5000])
    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=16)

    testset = models.FashionMnist(dataset[5000:10000])
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=16)

    # net = models.CNNNet(device=device)
    net = models.DenseNet(device=device, number_layers=6, growth_rate=8, drop_rate=0.1)
    cel = torch.nn.CrossEntropyLoss()
    # gcpl = functions.GCPLLoss(gamma=1.0, lambda_=0.1)
    # ddml = functions.PairwiseLoss(tao=10.0, b=1.0, beta=0.5)
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
        logger.info("Trainset size: %d, Epoch number: %d", len(trainset), epoch + 1)

        # CPL train
        # logger.info("Threshold: %f", models.Config.threshold)
        # prototypes = train(net, trainloader, gcpl, sgd)
        # torch.save(net.state_dict(), models.Config.pkl_path)

        # prototype_count = 0

        # for c in prototypes:
        #     prototype_count += len(prototypes[c])

        # logger.info("Prototype Count: %d", prototype_count)

        # accuracy = test(net, testloader, prototypes, gcpl.gamma)

        # CEL train
        train(net, trainloader, cel, sgd)

        torch.save(net.state_dict(), models.Config.pkl_path)
        accuracy = test(net, testloader)

        logger.info("Accuracy: %.4f", accuracy)


if __name__ == '__main__':
    main()
