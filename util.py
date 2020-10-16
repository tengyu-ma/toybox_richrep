import os
import shutil
import conf
import torch
import pandas as pd


class HyperP:
    def __init__(self, lr, batch_size, num_workers, epochs):
        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs


def get_mean_std(data_loader):
    """ Calculate dataset mean and std
    """
    mean = 0.
    var = 0.
    nb_samples = 0.
    for batch_idx, (x, y_true, file) in enumerate(data_loader):
        batch_samples = x.size(0)
        x = x.view(batch_samples, x.size(1), -1)
        mean += x.mean(2).sum(0)
        var += x.var(2).sum(0)
        nb_samples += batch_samples
        print(f'\rCalculate mean and std: {batch_idx + 1}/{len(data_loader)}', end='')
    print('')

    mean /= nb_samples
    var /= nb_samples
    std = torch.sqrt(var)

    print('mean:', mean)
    print('std:', std)
    return mean, std


def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
