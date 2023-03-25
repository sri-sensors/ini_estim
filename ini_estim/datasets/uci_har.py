from pathlib import Path
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
import numpy as np
import torch
import torch.utils.data as data
import logging
from tqdm.auto import tqdm
from .base import TimeSeriesDataset
LOGGER = logging.getLogger(__name__)


class UCIHARDataset(TimeSeriesDataset):
    FILENAME = r"UCI HAR Dataset.zip"
    URL = r"https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip"
    SIGNAL_FILES = ['body_acc_x',
                    'body_acc_y',
                    'body_acc_z',
                    'body_gyro_x',
                    'body_gyro_y',
                    'body_gyro_z',
                    'total_acc_x',
                    'total_acc_y',
                    'total_acc_z']
    DATA_DIR = "UCI HAR Dataset"
    TRAIN_DIR = 'train/Inertial Signals/'
    TRAIN_LABELS = "train/y_train.txt"
    TEST_DIR = 'test/Inertial Signals/'
    TEST_LABELS = "test/y_test.txt"

    label_names = [
        'walking', 'walking upstairs', 'walking downstairs',
        'sitting', 'standing', 'laying'
    ]
    sampling_period = 0.02
    variable_length = False

    def __init__(self, root, train=True, download=True, **kwargs):
        """
        Parameters
        ----------
        root : str or Path
            Root directory of the dataset where "UCI HAR Dataset/train/y_train.txt" 
            and "UCI HAR Dataset/train/y_test.txt" exist.
        train : bool
            If True (default), the training set will be loaded, otherwise the
            test set will be loaded.
        download : bool
            If true (default), the data will be downloaded into the root 
            directory if has not been downloaded yet.
        **kwargs
            Arguments to pass on to TimeSeriesDataset
        """
        super().__init__(**kwargs)
        self.train = train
        self.signals = None
        self.labels = None
        self.root = Path(root)
        self.num_labels = len(self.label_names)
        if download:
            self.download()

        self.load()
    
    @property
    def num_features(self):
        return self.signals.shape[2]
    
    @property
    def num_samples(self):
        return self.signals.shape[1]

    def __len__(self):
        return self.signals.shape[0] if self.signals is not None else 0

    def __getitem__(self, idx):
        data = torch.from_numpy(self.signals[idx])
        if self.noise > 0.0:
            data = data + self.noise*data.std()*torch.randn_like(data)
        return data, self.labels[idx]

    @property
    def _zipfile(self):
        return self.root / self.FILENAME

    def load(self):
        data_loc = self.root / self.DATA_DIR
        LOGGER.info("Loading UCI HAR data from {}".format(self.root))
        signals = []
        datadir = self.TRAIN_DIR if self.train else self.TEST_DIR
        labelfile = self.TRAIN_LABELS if self.train else self.TEST_LABELS
        suffix = "_train.txt" if self.train else "_test.txt"
        
        for s in tqdm(self.SIGNAL_FILES):
            with open(data_loc / Path(datadir) / (s + suffix)) as f:
                signals.append(np.loadtxt(f, dtype=np.float32))
        self.signals = np.dstack(signals)
        with open(data_loc / labelfile) as f:
            self.labels = torch.from_numpy(np.loadtxt(f, dtype=np.int64) - 1)

    def download(self):
        zipfile = self._zipfile
        
        if zipfile.exists():
            if (self.root / self.DATA_DIR).exists():
                return
            else:
                self._extract_zip()

        with urlopen(self.URL) as request:
            chunk_size = 4096
            total_size = request.info().get('Content-Length').strip()
            total_size = int(total_size)
            print("Downloading data, total size: {}KB".format(total_size//1024))
            buf = BytesIO()
            bytes_left = total_size
            for _ in tqdm(range((total_size + chunk_size - 1)//chunk_size)):
                bytes_to_read = min(chunk_size, bytes_left)
                buf.write(request.read(bytes_to_read))
                bytes_left -= bytes_to_read

        with open(zipfile, 'wb') as f:
            f.write(buf.getbuffer())

        self._extract_zip()
    
    def _extract_zip(self):
        with ZipFile(self._zipfile, 'r') as f:
            f.extractall(self.root)

   