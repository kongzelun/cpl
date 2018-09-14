import os
import logging
import numpy as np
import torch
from torch import tensor
import torch.optim as optim
from torch.utils.data import DataLoader
import fashion_minst as model
import functions

PROTOTYPES = {}


def setup_logger(level=logging.DEBUG, filename=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if filename is not None:
        file_handler = logging.FileHandler(filename=filename)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    return logger


def train(net, dataloader, criterion, optimizer, all_prototypes):
    logger = logging.getLogger(__name__)
    loss_sum = 0.0

    for i, (feature, label) in enumerate(dataloader):
        feature, label = feature.to(net.device), int(label)

        optimizer.zero_grad()

        # extract abstract feature through CNN.
        feature = net(feature).view(1, -1)

        closest_prototype_index, min_distance = functions.assign_prototype(tensor(feature.data), label, all_prototypes, tensor(model.Config.threshold).to(net.device))

        loss = criterion(feature, label, all_prototypes, all_prototypes[label][closest_prototype_index])
        loss.backward()
        optimizer.step()

        loss_sum += loss

        logger.debug("%5d: Loss: %.4f, Distance: %.4f", i + 1, loss, min_distance)

    logger.info("Loss Average: %.4f", loss_sum / len(dataloader))


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


def main():
    logger = setup_logger(level=logging.DEBUG, filename='log.txt')

    train_epoch_number = 10

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = np.loadtxt(model.Config.dataset_path, delimiter=',')

    trainset = model.FashionMnist(dataset[:5000])
    trainloader = DataLoader(dataset=trainset, batch_size=1, shuffle=True, num_workers=4)

    testset = model.FashionMnist(dataset[5000:10000])
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=0)

    # model = net.CNNNet(device=DEVICE)
    net = model.DenseNet(device=device, number_layers=6, growth_rate=12)
    gcpl = functions.GCPLLoss(gamma=1.0, lambda_=0.1)
    sgd = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    if not os.path.exists("pkl"):
        os.mkdir("pkl")

    if os.path.exists(model.Config.pkl_path):
        state_dict = torch.load(model.Config.pkl_path)
        try:
            net.load_state_dict(state_dict)
            logger.info("Load state from file %s.", model.Config.pkl_path)
        except RuntimeError:
            logger.error("Loading state from file %s failed.", model.Config.pkl_path)

    for epoch in range(train_epoch_number):
        logger.info("Trainset size: %d, Epoch number: %d", len(trainset), epoch + 1)

        PROTOTYPES.clear()

        train(net, trainloader, gcpl, sgd, PROTOTYPES)

        prototype_count = 0

        for c in PROTOTYPES:
            prototype_count += len(PROTOTYPES[c])

        logger.info("Prototype Count: %d", prototype_count)

        torch.save(net.state_dict(), model.Config.pkl_path)

        accuracy = test(net, testloader, PROTOTYPES, gcpl.gamma)

        logger.info("Accuracy: %.4f", accuracy)


if __name__ == '__main__':
    main()
