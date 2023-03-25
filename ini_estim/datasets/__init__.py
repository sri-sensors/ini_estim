from .uci_har import UCIHARDataset
from .emg import KinematicEMGADL
from .luke import LukeHandDataset
from .gaussian_process import GaussianProcess
from collections import namedtuple
import torch


DatasetTuple = namedtuple("DatasetTuple",
        ["train_set", "val_set", "test_set", "num_features", "total_samples", "variable_length"]
        )
datasets = {
    "uci_har": UCIHARDataset,
    "emg_adl": KinematicEMGADL,
    "luke": LukeHandDataset,
    "gp": GaussianProcess
}


def get_dataset(name, data_folder, training_split=0.8, **kwargs):
    """ Retrieve dataset 
    
    Parameters
    ----------
    name : str
        The name of the dataset (from datasets.keys())
    data_folder : str
        The path of the dataset
    training_split : float
        Fraction of original training data to use for training. The rest 
        will be used for  validation. The default value is 0.8, meaning
        an 80/20 split between training and validation.
    **kwargs
        Extra arguments to pass on to the dataset
    
    Returns
    -------
    train_set, val_set, test_set, num_features, total_samples
        train_set and val_set are split from the training data, 
        retrieved with Dataset(train=True). test_set is the full test
        data, i.e. Dataset(train=False). 
    """
    from torch.utils.data import random_split
    full_dataset = datasets[name](root=data_folder, train=True, **kwargs)
    Ntot = len(full_dataset)
    Ntrain = int(0.8*Ntot)
    Nval = Ntot - Ntrain
    rng_state = torch.random.get_rng_state()
    torch.random.manual_seed(42)
    train_set, val_set = random_split(full_dataset, [Ntrain, Nval])
    torch.random.set_rng_state(rng_state)
    test_set = datasets[name](root=data_folder, train=False, **kwargs)
    num_samples = None if full_dataset.variable_length else full_dataset.num_samples
    return DatasetTuple(
        train_set, val_set, test_set, full_dataset.num_features, 
        num_samples, full_dataset.variable_length
    )

