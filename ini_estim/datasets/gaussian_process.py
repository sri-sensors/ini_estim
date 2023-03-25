import torch
from .base import TimeSeriesDataset
from torch.utils.data import Dataset
from ini_estim.ml.gaussian_process import generate_gp
import numpy as np


class GaussianProcess(TimeSeriesDataset):
    """ Generated gaussian process data with RBF kernel """
    variable_length = False
    def __init__(self, num_signals=512, num_features=1, signal_length=128, 
            min_std=1.0, max_std=10.0, train=True, **kwargs):
        """Generate RBF gaussian process signals

        Parameters
        ----------
        num_signals : int, optional
            The number of signals to generate, by default 512
        num_features : int, optional
            The number of features per signal, by default 1
        signal_length : int, optional
            The length of each signal, by default 128
        min_std : float, optional
            The minimum standard deviation of the RBF covariance, by default 1.0
        max_std : float, optional
            The maximum standard deviation of the RBF covariance, by default 10.0
        train : bool, optional
            Flag to generate training set or test set, by default True. Because
            the signals are generated, this flag sets the seed to the 
            internal random number generator.
        **kwargs
            Arguments to pass on to TimeSeriesDataset
        """
        super().__init__(**kwargs)
        
        seed = 12345 if train else 54321
        generator = np.random.default_rng(seed)
        stddevs = generator.uniform(min_std, max_std, num_signals)
        variances = stddevs**2
        signals = np.dstack([np.row_stack([
            generate_gp(
                    signal_length, generator=generator, variance=v
                ) for v in variances
            ]) for f in range(num_features)]
        )
        smax = max(np.abs(signals.max()), np.abs(signals.min()))
        signals /= smax
        self.signals = torch.from_numpy(signals).type(torch.float)
        self.labels = torch.from_numpy(stddevs).type(torch.float)

    @property
    def num_features(self):
        return self.signals.shape[2]
    
    @property
    def num_samples(self):
        return self.signals.shape[1]

    def __len__(self):
        return self.signals.shape[0] if self.signals is not None else 0
    
    def __getitem__(self, idx):
        data = self.signals[idx]
        if self.noise > 0.0:
            data = data + self.noise*data.std()*torch.randn_like(data)
        return data, self.labels[idx]

