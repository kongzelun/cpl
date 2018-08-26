import torchvision
from torchvision import transforms
import numpy as np


def dump_csv(dataset, filename):
    data = []

    for x, y in dataset:
        i = np.append((x * 255).int().numpy(), int(y))
        data.append(i)

    data = np.array(data)
    np.random.shuffle(data)
    np.savetxt(filename, data, fmt='%d', delimiter=',')


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=True, transform=transform, download=False)
    testset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=False, transform=transform, download=False)

    dump_csv(trainset, 'data/fashion-mnist_train.csv')
    dump_csv(testset, 'data/fashion-mnist_test.csv')

    trainset = torchvision.datasets.CIFAR10(root="data/cifar10", train=True, transform=transform, download=False)
    testset = torchvision.datasets.CIFAR10(root="data/cifar10", train=False, transform=transform, download=False)

    dump_csv(trainset, 'data/cifar10_train.csv')
    dump_csv(testset, 'data/cifar10_test.csv')
