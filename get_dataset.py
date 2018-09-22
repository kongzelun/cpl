import os
import torchvision
from torchvision import transforms
import numpy as np


def dump_csv(dataset, filename, size=None):
    data = []

    for x, y in dataset:
        i = np.append((x * 255).int().numpy(), int(y))
        data.append(i)

        if size and len(data) >= size:
            break

    data = np.array(data)
    np.random.shuffle(data)
    np.savetxt(filename, data, fmt='%d', delimiter=',')


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor()])

    if not os.path.exists("data"):
        os.mkdir("data")

    download_fashion_mnist = True
    download_cifar10 = True

    if os.path.exists("data/fashion-mnist"):
        download_fashion_mnist = False

    if os.path.exists("data/cifar10"):
        download_cifar10 = False

    trainset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=True, transform=transform, download=download_fashion_mnist)
    testset = torchvision.datasets.FashionMNIST(root="data/fashion-mnist", train=False, transform=transform, download=download_fashion_mnist)

    dump_csv(trainset, 'data/fashion-mnist_train.csv')
    dump_csv(testset, 'data/fashion-mnist_test.csv')

    trainset = torchvision.datasets.CIFAR10(root="data/cifar10", train=True, transform=transform, download=download_cifar10)
    testset = torchvision.datasets.CIFAR10(root="data/cifar10", train=False, transform=transform, download=download_cifar10)

    dump_csv(trainset, 'data/cifar10_train.csv')
    dump_csv(testset, 'data/cifar10_test.csv')
