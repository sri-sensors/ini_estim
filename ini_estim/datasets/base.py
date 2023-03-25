from torch.utils.data import Dataset
import torch
import numpy as np
import pathlib


class TimeSeriesDataset(Dataset):
    def __init__(self, root="", noise=0.0):
        self.noise = noise
        self.root = pathlib.Path(root)

    @property
    def num_features(self):
        raise NotImplementedError

    @property
    def num_samples(self):
        raise NotImplementedError

    @property
    def variable_length(self):
        raise NotImplementedError
    

class XYCSVDataset(Dataset):
    def __init__(self, xpath, ypath, xdelim=",", ydelim=None, normalize=False, randomize=False):
        if ydelim is None:
            ydelim = xdelim
        
        self.x = self._load_csv(xpath, xdelim, normalize)
        self.y = self._load_csv(ypath, ydelim, normalize)
        if self.x.shape[0] != self.y.shape[0]:
            raise ValueError("X and Y must have same number of samples")
        if randomize:
            idx = torch.randperm(self.x.shape[0])
            self.x = self.x[idx]
            self.y = self.y[idx]
    
    @property
    def xfeatures(self):
        return self.x.shape[1]
    
    @property
    def yfeatures(self):
        return self.y.shape[1]
    
    def __len__(self):
        return self.x.shape[0]
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def _load_csv(self, path, delim, normalize=False):
        with open(path) as f:
            data = np.loadtxt(f, dtype=np.float32, delimiter=delim)
        
        if normalize:
            data = data - np.mean(data, 0)
            data = data / np.maximum(np.std(data, 0), 1e-6)
        
        return torch.from_numpy(data)
