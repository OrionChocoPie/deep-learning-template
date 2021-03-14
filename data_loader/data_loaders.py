from base import BaseDataLoader
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision import datasets, transforms

import numpy as np
import torch

class Circles(Dataset):
    def __init__(self, n_samples, shuffle, noise, factor=.8):
        self.X, self.y = datasets.make_circles(n_samples=n_samples, shuffle=shuffle,
                                               noise=noise, factor=factor)

        self.X = StandardScaler().fit_transform(self.X)
        self.X, self.y = self.X.astype(np.float32), self.y.astype(np.int)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

class Moons(Dataset):
    def __init__(self, n_samples, shuffle, noise):
        self.X, self.y = datasets.make_moons(n_samples=n_samples, shuffle=shuffle,
                                               noise=noise)

        self.X = StandardScaler().fit_transform(self.X)
        self.X, self.y = self.X.astype(np.float32), self.y.astype(np.int)


    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(np.array(self.y[idx]))

class HomeTaskDataLoader(BaseDataLoader):
    def __init__(self, batch_size, shuffle=True, validation_split=0.0, num_workers=1, 
            training=True):
        self.dataset = Moons(5000, True, 0.2)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
