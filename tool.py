import os
import time
import random as rd
import numpy as np
import matplotlib.pyplot as plt

import umap
from scipy.optimize import linear_sum_assignment as linear_assignment

import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DataSaver():
    def __init__(self):
        pass

    def SaveData(self, input_data, latent, label, numEpoch, path, name):

        if type(latent) == torch.Tensor:
            latent = latent.detach().cpu().numpy()
        if type(label) == torch.Tensor:
            label = label.detach().cpu().numpy()

        np.save(path + name + 'latent_2.npy', latent[0])
        np.save(path + name + 'latent_clu.npy', latent[1])

        if numEpoch < 1:
            if type(input_data) == torch.Tensor:
                input_data = input_data.detach().cpu().numpy()
            np.save(path + name + 'input.npy', input_data.astype(np.float16))
            np.save(path + name + 'label.npy', label.astype(np.float16))


class GIFPloter():
    def __init__(self):
        pass

    def PlotOtherLayer(self, fig, data, label, args, cluster, s):

        color_list = []
        for i in range(label.shape[0]):
            color_list.append(int(label[i]))

        ax = fig.add_subplot(1, 1, 1)
        if cluster is None:
            ax.scatter(data[:, 0], data[:, 1], c=label, s=s, cmap='rainbow_r')
        else:
            ax.scatter(data[:label.shape[0], 0], data[:label.shape[0], 1], c=label, s=s, cmap='rainbow_r')
            ax.scatter(data[label.shape[0]:, 0], data[label.shape[0]:, 1], c=list(np.arange(args['n_cluster'])), s=30, cmap='rainbow_r', edgecolors='k')

        plt.axis('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        plt.xticks([])
        plt.yticks([])

    def AddNewFig(self, latent, label, path, name, args, cluster=None):

        fig = plt.figure(figsize=(5, 5))

        if latent.shape[1] <= 2:
            if cluster is not None:
                latent = np.concatenate((latent, cluster), axis=0)
            self.PlotOtherLayer(fig, latent, label, args, cluster, s=0.3)  
        else:
            reducer = umap.UMAP(n_neighbors=5, min_dist=0.7,  metric='correlation')
            if latent.shape[0] > 20000:
                latent = latent[:10000]
                label = label[:10000]
            if cluster is not None:
                latent = np.concatenate((latent, cluster), axis=0)
            latent = reducer.fit_transform(latent)
            self.PlotOtherLayer(fig, latent, label, args, cluster, s=0.3)  
        
        plt.tight_layout()
        plt.savefig(path + name, dpi=300)
        plt.close()


def SetSeed(seed):

    SEED = seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    rd.seed(SEED)
    np.random.seed(SEED)


def GetPath(name):

    rest = time.strftime("%Y%m%d%H%M%S_", time.localtime()) + os.popen('git rev-parse HEAD').read()
    path = '../log/' + name + '/'
    if not os.path.exists(path):
        os.makedirs(path)

    return path


def SaveParam(path, param):

    import json
    paramDict = param
    paramStr = json.dumps(paramDict, indent=4)
    print(paramStr, file=open(path + '/param.txt', 'a'))


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)

    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.array(ind).T
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size