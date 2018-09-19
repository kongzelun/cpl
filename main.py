import os
import logging
import numpy as np
import torch
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


def train(net, dataloader, criterion, optimizer, pairwise=False):
    logger = logging.getLogger(__name__)
    loss_sum = 0.0

    threshold = models.Config.threshold

    gcpl = criterion[0]
    pwl = criterion[1]

    all_prototypes = {}

    logger.info("Threshold: %6.4f", threshold)

    if pairwise:

        for i, (s0, s1) in enumerate(dataloader):
            feature0, label0 = s0[0].to(net.device), int(s0[1])
            feature1, label1 = s1[0].to(net.device), int(s1[1])

            optimizer.zero_grad()

            # extract abstract feature through CNN.
            feature0 = net(feature0).view(1, -1)
            feature1 = net(feature1).view(1, -1)

            # if (i + 1) % 10 == 0:
            #     threshold = distances_sum / 100
            #     distances_sum = 0.0
            # else:
            #     distances_sum += min_distance

            # distances_sum += min_distance
            # threshold = distances_sum / (i + 1)

            loss0 = gcpl(feature0, label0, all_prototypes)
            loss1 = gcpl(feature1, label1, all_prototypes)
            loss = pwl(feature0, feature1, label0, label1)
            loss = loss + loss0 + loss1
            loss.backward()
            optimizer.step()

            loss_sum += loss

            logger.debug("%5d: Loss: %7.4f", i + 1, loss)
    else:
        for i, (feature, label) in enumerate(dataloader):
            feature, label = feature.to(net.device), int(label)

            optimizer.zero_grad()

            feature = net(feature).view(1, -1)

            loss = gcpl(feature, label, all_prototypes)

            loss.backward()
            optimizer.step()

            loss_sum += loss

            logger.debug("%5d: Loss: %7.4f", i + 1, loss)

    logger.info("Loss Average: %7.4f", loss_sum / len(dataloader))

    return all_prototypes


# def train(net, dataloader, criterion, optimizer):
#     logger = logging.getLogger(__name__)
#
#     loss_sum = 0.0
#
#     for i, (feature, label) in enumerate(dataloader):
#         feature, label = feature.to(net.device), label.to(net.device)
#         optimizer.zero_grad()
#         out = net(feature).view(1, -1)
#         loss = criterion(out, label)
#         loss.backward()
#         optimizer.step()
#
#         loss_sum += loss
#
#         logger.debug("%5d: Loss: %6.4f", i + 1, loss)
#
#     logger.info("Loss Average: %6.4f", loss_sum / len(dataloader))


def test(net, dataloader, all_prototypes, gamma):
    logger = logging.getLogger(__name__)

    distance_sum = 0.0

    correct = 0

    for i, (feature, label) in enumerate(dataloader):
        feature, label = feature.to(net.device), int(label)

        # extract abstract feature through CNN.
        feature = net(feature).view(1, -1)

        predicted_label, probability, min_distance = functions.predict(feature, all_prototypes, gamma)

        if label == predicted_label:
            correct += 1

        distance_sum += min_distance

        logger.debug("%5d: Label: %d, Prediction: %d, Probability: %7.4f, Distance: %7.4f, Accuracy: %7.4f",
                     i + 1, label, predicted_label, probability, min_distance, correct / (i + 1))

    logger.info("Distance Average: %7.4f", distance_sum / len(dataloader))

    return correct / len(dataloader)


# def test(net, dataloader):
#     logger = logging.getLogger(__name__)
#
#     correct = 0
#
#     for i, (feature, label) in enumerate(dataloader):
#         feature, label = feature.to(net.device), label.to(net.device)
#         out = net(feature).view(1, -1)
#
#         predicted_label = out.max(dim=1)[1]
#
#         if label == predicted_label:
#             correct += 1
#
#         logger.debug("%5d: Label: %d, Prediction: %d, Accuracy: %.4f", i + 1, label, predicted_label, correct / (i + 1))
#
#     return correct / len(dataloader)


def main():
    logger = setup_logger(level=logging.DEBUG, filename='log.txt')

    train_epoch_number = 10

    batch_size = 1

    pairwise = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = np.loadtxt(models.Config.dataset_path, delimiter=',')
    np.random.shuffle(dataset[:5000])

    trainset = models.DataSet(dataset[:5000], pairwise=pairwise)
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=24)

    testset = models.DataSet(dataset[5000:])
    testloader = DataLoader(dataset=testset, batch_size=1, shuffle=False, num_workers=2)

    # net = models.CNNNet(device=device)
    net = models.DenseNet(device=device, number_layers=16, growth_rate=16, drop_rate=0.0)
    # cel = torch.nn.CrossEntropyLoss()
    gcpl = functions.GCPLLoss(threshold=models.Config.threshold, gamma=0.1, lambda_=0.001)
    pwl = functions.PairwiseLoss(tao=10.0, b=2.0, beta=0.5)
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
        prototypes = train(net, trainloader, (gcpl, pwl), sgd, pairwise=pairwise)
        torch.save(net.state_dict(), models.Config.pkl_path)

        prototype_count = 0

        for c in prototypes:
            prototype_count += len(prototypes[c])

        logger.info("Prototype Count: %d", prototype_count)

        accuracy = test(net, testloader, prototypes, gcpl.gamma)
        logger.info("Accuracy: %7.4f\n", accuracy)

        # CEL train
        # train(net, trainloader, cel, sgd)
        # torch.save(net.state_dict(), models.Config.pkl_path)

        # if (epoch + 1) % 10 == 0:
        #     accuracy = test(net, testloader)
        #     logger.info("Accuracy: %.4f", accuracy)


if __name__ == '__main__':
    main()
