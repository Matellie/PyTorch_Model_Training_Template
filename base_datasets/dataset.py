import torch
from torch.utils.data import Dataset

from sklearn import datasets
from sklearn.preprocessing import StandardScaler

import numpy as np
import os

class BaseDataset(Dataset):

    def __init__(self):
        """
        Dataset parameters:
        - 
        """
        super(BaseDataset, self).__init__()

    def __getitem__(self, index):
        """
        Return dataset's items
        """
        pass
    
    def __len__(self):
        """
        Return length of the dataset
        """
        pass

class WineDataset(Dataset):
    """
    My best result: 82% accuracy

    model = base_model_code.SimpleNeuralNet(input_size=13, hidden_size=32, num_classes=3)
    loss = nn.CrossEntropyLoss

    batch_size = 256
    nb_workers = 0
    learning_rate = 0.00001
    nb_epochs = 100000
    """
    def __init__(self):
        super(WineDataset, self).__init__()

        xy = np.loadtxt(os.path.join('base_datasets', 'data', 'wine.csv'), delimiter=',', dtype=np.float32, skiprows=1)

        self.x = torch.from_numpy(xy[:, 1:])
        y = torch.from_numpy(xy[:, [0]]).type(torch.LongTensor)
        self.y = torch.flatten(y - 1)

        self.nb_samples = xy.shape[0]
        self.nb_features = xy.shape[1] - 1

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.nb_samples
    
    def get_nb_features(self):
        return self.nb_features
    
    def get_name(self):
        return 'Wine'

class CancerDataset(Dataset):

    def __init__(self):
        super(CancerDataset, self).__init__()

        xy = datasets.load_breast_cancer()
        X, Y = xy.data, xy.target

        sc = StandardScaler()
        X = sc.fit_transform(X)

        self.x = torch.from_numpy(X.astype(np.float32))
        Y = torch.from_numpy(Y.astype(np.float32))
        self.y = Y.view(Y.shape[0], 1)

        self.nb_samples = X.shape[0]
        self.nb_features = X.shape[1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nb_samples

    def get_nb_features(self):
        return self.nb_features
    
    def get_name(self):
        return 'Cancer'

class MNISTDataset(Dataset):
    """
    My best result: 97% accuracy

    model = base_model_code.SimpleNeuralNet(input_size=64, hidden_size=32, num_classes=10)
    loss = nn.CrossEntropyLoss

    batch_size = 1024
    nb_workers = 0
    learning_rate = 0.00001
    nb_epochs = 200000
    """
    def __init__(self):
        super(MNISTDataset, self).__init__()

        xy = datasets.load_digits()

        self.x = torch.from_numpy(xy.data.astype(np.float32))
        y = torch.from_numpy(xy.target).type(torch.LongTensor)
        self.y = torch.flatten(y)

        self.nb_samples = xy.data.shape[0]
        self.nb_features = xy.data.shape[1]

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.nb_samples

    def get_nb_features(self):
        return self.nb_features
    
    def get_name(self):
        return 'MNIST'