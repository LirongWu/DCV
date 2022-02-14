import os
import torch
import numpy as np
import pandas as pd

from PIL import Image
import scipy.io as sio
import torchvision.datasets as datasets


def GetData(args):

    if args["data_name"] == "mnist-test":
        dataloader = datasets.MNIST(root="../data", train=False, download=True, transform=None)
        data_train = dataloader.data.reshape(-1, 28 * 28).float() / 255
        label_train = dataloader.targets

    elif args["data_name"] == "mnist-full":
        dataloader = datasets.MNIST(root="../data", train=True, download=True, transform=None)
        data_train = dataloader.data.reshape(-1, 28 * 28).float() / 255
        label_train = dataloader.targets

        dataloader = datasets.MNIST(root="../data", train=False, download=True, transform=None)
        data_test = dataloader.data.reshape(-1, 28 * 28).float() / 255
        label_test = dataloader.targets

        data_train = np.concatenate((data_train, data_test), axis=0)
        label_train = np.concatenate((label_train, label_test), axis=0)

    elif args["data_name"] == "coil100rgb":
        path = "../data/coil-100"
        fig_path = os.listdir(path)

        label = []
        data = np.zeros((100 * 72, 128 * 128 * 3))
        for i, path_i in enumerate(fig_path):
            if "obj" in path_i:
                I = Image.open(path + "/" + path_i)
                I_array = np.array(I).reshape(1, -1)
                data[i] = I_array
                label.append(int(fig_path[i].split("__")[0].split("obj")[1]))

        data = data / 255
        label = np.array(label) - 1.0
        data_train, data_test = data, data
        label_train, label_test = label, label

    elif args["data_name"] == "usps":
        data = sio.loadmat('../data/USPS/usps_resampled.mat')
        x_train, y_train, x_test, y_test = data['train_patterns'].T, data['train_labels'].T, data['test_patterns'].T, data['test_labels'].T
        
        x = np.concatenate((x_train, x_test)).astype(np.float32)
        data_train = (x.reshape((x.shape[0], -1)) + 1.0) / 2.0

        y_train = [np.argmax(l) for l in y_train]
        y_test = [np.argmax(l) for l in y_test]
        label_train = np.concatenate((y_train, y_test)).astype(np.int32)
 
    elif args["data_name"] == "fashionmnist":
        x = np.load('../data/fashionmnist/data.npy').astype(np.float32)
        data_train = x.reshape((x.shape[0], -1))
        label_train = np.load('../data/fashionmnist/labels.npy').astype(np.int32)

    elif args["data_name"] == "har":
        x_train = pd.read_csv('../data/HAR/train/X_train.txt', sep=r'\s+', header=None)
        y_train = pd.read_csv('../data/HAR/train/y_train.txt', header=None)
        x_test = pd.read_csv('../data/HAR/test/X_test.txt', sep=r'\s+', header=None)
        y_test = pd.read_csv('../data/HAR/test/y_test.txt', header=None)

        data_train = np.concatenate((x_train, x_test)).astype(np.float32)
        label_train = np.concatenate((y_train, y_test)).astype(np.int32) - 1
        label_train = label_train.reshape((label_train.size,))

    elif args["data_name"] == "reuters-10k":
        data = np.load('../data/Reuters-10k/reuters-10k.npy', allow_pickle=True).item()

        data_train = data['data']
        label_train = data['label']
        data_train = data_train.reshape((data_train.shape[0], -1)).astype(np.float32)
        label_train = label_train.reshape((label_train.shape[0])).astype(np.int32)

    elif args["data_name"] == "pendigits":

        with open('../data/pendigits/pendigits.tra') as file:
            data = file.readlines()
        data = [list(map(float, line.split(','))) for line in data]
        data = np.array(data).astype(np.float32)
        data_train, labels_train = data[:, :-1], data[:, -1]

        with open('../data/pendigits/pendigits.tes') as file:
            data = file.readlines()
        data = [list(map(float, line.split(','))) for line in data]
        data = np.array(data).astype(np.float32)
        data_test, labels_test = data[:, :-1], data[:, -1]

        data_train = np.concatenate((data_train, data_test)).astype('float32')
        label_train = np.concatenate((labels_train, labels_test))
        data_train /= 100.
        label_train = label_train.astype(np.int32)

    data_train = torch.tensor(data_train)
    label_train = torch.tensor(label_train)

    return data_train.float(), label_train