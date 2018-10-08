import os
from os.path import exists
import torchvision
from torchvision import transforms
import numpy as np

FashionMNIST_PATH = 'data/fashion-mnist'
CIFAR10_PATH = 'data/cifar10'
LSUN_PATH = 'data/lsun'
SVHN_PATH = ''


def dump_csv(dataset, filename, size=None):
    data = []

    for x, y in dataset:
        i = np.append(x.numpy(), int(y))
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

    download_fashion_mnist = not exists(FashionMNIST_PATH)
    download_cifar10 = not exists(CIFAR10_PATH)
    download_lsun = not exists(LSUN_PATH)

    trainset = torchvision.datasets.FashionMNIST(root=FashionMNIST_PATH, train=True, transform=transform, download=download_fashion_mnist)
    testset = torchvision.datasets.FashionMNIST(root=FashionMNIST_PATH, train=False, transform=transform, download=download_fashion_mnist)

    dump_csv(trainset, 'data/fashion-mnist_train.csv')
    dump_csv(testset, 'data/fashion-mnist_test.csv')

    trainset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=True, transform=transform, download=download_cifar10)
    testset = torchvision.datasets.CIFAR10(root=CIFAR10_PATH, train=False, transform=transform, download=download_cifar10)

    dump_csv(trainset, 'data/cifar10_train.csv')
    dump_csv(testset, 'data/cifar10_test.csv')

    trainset = torchvision.datasets.LSUN(root=LSUN_PATH, classes='train', transform=transform)
    dump_csv(trainset, 'data/lsun_train.csv')

