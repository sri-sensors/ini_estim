import numpy as np
import os
from pathlib import Path
import scipy.io as sio


class MatReader(object):
    """Loads MATLAB .mat file and formats it for simple use.
    https://github.com/erkil1452/touch/blob/master/shared/dataset_tools.py
    """

    def __init__(self, flatten1d=True):
        self.flatten1D = flatten1d

    def loadmat(self, filename):
        meta = sio.loadmat(filename, struct_as_record=False)

        meta.pop('__header__', None)
        meta.pop('__version__', None)
        meta.pop('__globals__', None)

        meta = self._squeeze_item(meta)
        return meta

    def _squeeze_item(self, item):
        if isinstance(item, np.ndarray):
            if item.dtype == np.object:
                if item.size == 1:
                    item = item[0, 0]
                else:
                    item = item.squeeze()
            elif item.dtype.type is np.str_:
                item = str(item.squeeze())
            elif self.flatten1D and len(item.shape) == 2 and (
                    item.shape[0] == 1 or item.shape[1] == 1):
                # import pdb; pdb.set_trace()
                item = item.flatten()

            if isinstance(item, np.ndarray) and item.dtype == np.object:
                it = np.nditer(item, flags=['multi_index', 'refs_ok'],
                               op_flags=['readwrite'])
                while not it.finished:
                    item[it.multi_index] = self._squeeze_item(
                        item[it.multi_index])
                    it.iternext()

        if isinstance(item, dict):
            for k, v in item.items():
                item[k] = self._squeeze_item(v)
        elif isinstance(item, sio.matlab.mio5_params.mat_struct):
            for k in item._fieldnames:
                v = getattr(item, k)
                setattr(item, k, self._squeeze_item(v))

        return item


def read_signals_ucihar(filename):
    with open(filename, 'r') as fp:
        data = fp.read().splitlines()
        data = map(lambda x: x.rstrip().lstrip().split(), data)
        data = [list(map(float, line)) for line in data]
    return data


def read_labels_ucihar(filename):
    with open(filename, 'r') as fp:
        activities = fp.read().splitlines()
        activities = list(map(int, activities))
    return activities


def load_ucihar_data(folder):
    """Loads the UCI Human Activity Recognition (HAR) dataset.
    """
    train_folder = folder / 'train' / 'Inertial Signals'
    test_folder = folder / 'test' / 'Inertial Signals'
    labelfile_train = folder / 'train' / 'y_train.txt'
    labelfile_test = folder / 'test' / 'y_test.txt'
    train_signals, test_signals = [], []
    for input_file in os.listdir(train_folder):
        signal = read_signals_ucihar(train_folder.joinpath(input_file))
        train_signals.append(signal)
    train_signals = np.transpose(np.array(train_signals), (1, 2, 0))
    for input_file in os.listdir(test_folder):
        signal = read_signals_ucihar(test_folder.joinpath(input_file))
        test_signals.append(signal)
    test_signals = np.transpose(np.array(test_signals), (1, 2, 0))
    train_labels = read_labels_ucihar(labelfile_train)
    test_labels = read_labels_ucihar(labelfile_test)
    return train_signals, train_labels, test_signals, test_labels


