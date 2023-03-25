from io import BytesIO
from urllib.request import urlopen
import numpy as np
import torch
import torch.utils.data as data
from scipy import io
from pathlib import Path
import warnings
from tqdm.auto import trange
from .base import TimeSeriesDataset


class KinematicEMGADL(TimeSeriesDataset):
    """ Kinematic and EMG dataset for Activities of Daily Living

    Please see the paper below for a more thorough description of the dataset.
    Here is a brief summary (mostly copied from the the paper):

    A calibrated database of kinematics and EMG of the forearm and hand during
    activities of daily living.

    Authors: Jarque-Bou, Nestor; Vergara, Margarita; Sancho-Bru, Joaquín L.;
    Gracia-Ibáñez, Verónica; Roda-Sales, Alba

    Downloaded from: https://zenodo.org/record/3469380#.XqoKDJp7nOR
    "KIN-MUS UJI Dataset contains 572 recordings with anatomical angles and
    forearm muscle activity of 22 subjects while performing 26 representative
    activities of daily living. This dataset is, to our knowledge, the biggest
    currently available hand kinematics and muscle activity dataset to focus on
    goal-oriented actions. Data were recorded using a CyberGlove instrumented
    glove and surface EMG electrodes, both properly synchronised. Eighteen hand
    anatomical angles were obtained from the glove sensors by a validated
    calibration procedure. Surface EMG activity was recorded from seven
    representative forearm areas. The statistics verified that data were not
    affected by the experimental procedures and were similar to the data
    acquired under real-life conditions."


    Attributes
    ----------
    data
        Numpy record array with the fields defined as follows.
    fields
        The fields available in each returned record.
        'Subject', 'ADL', 'Phase', 'time', 'Kinematic_data', 'EMG_data'

        Subject - The subject ID. There are 22 unique subjects.
        ADL - The activity ID. Could be used for classification
            0 Collecting a coin and putting it into a change purse
            1 Opening and closing a zip
            2 Removing the coin from the change purse and leaving it on the table
            3 Catching and moving two different sized wooden cubes
            4 Lifting and moving an iron from one marked point to another
            5 Taking a screwdriver and turning a screw clockwise 360° with it
            6 Taking a nut and turning it until completely inserted inside the bolt
            7 Taking a key, placing it in a lock and turning it counter-clockwise 180°
            8 Turning a door handle 30°
            9 Tying a shoelace
            10 Unscrewing two lids and leaving them on the table
            11 Passing two buttons through their respective buttonhole using both hands
            12 Taking a bandage and putting it on his/her left arm up to the elbow
            13 Taking a knife with the right hand and a fork with the left hand and splitting a piece of clay (sitting)
            14 Taking a spoon with the right hand and using it 5 times to eat soup (sitting)
            15 Picking up a pen from the table, writing his/her name and putting the pen back on the table (sitting)
            16 Folding a piece of paper with both hands, placing it into an envelope and leaving it on the table (sitting)
            17 Taking a clip and putting it on the flap of the envelope (sitting)
            18 Writing with the keypad (sitting)
            19 Picking up the phone, placing it to his/her ear and hanging up the phone (sitting)
            20 Pouring 1L of water from a carton into a jug (sitting)
            21 Pouring water from the jug into the cup up to a marked point (sitting)
            22 Pouring the water from the cup back into the jug (sitting)
            23 Putting toothpaste on the toothbrush
            24 Using a spray over the table 5 times
            25 Cleaning the table with a cloth for 5 seconds
        Phase - The phase of movement. 
            0 reaching
            1 manipulation
            2 releasing
        time - The timestamp.
        Kinematic_data - Calibrated anatomical angles
        EMG_data - Normalised sEMG signal for the seven representative spot 
            areas
    signals
        A list of torch tensors (float32) of the concatenated kinematic and emg
        data. Each array is (N x 25), where N is the sequence length of the 
        particular array. The last dimension corresponds to the different 
        kinematic or emg signals. This is what is returned by __getitem__,
        along with the corresponding activity.
        
    """
    URL = r"https://zenodo.org/record/3469380/files/KIN_MUS_UJI.mat?download=1"
    MATFILE = "KIN_MUS_UJI.mat"
    SUBDIR = "kinematics_EMG_forearm_and_hand"
    adl_descriptions = [
        "Collecting a coin and putting it into a change purse",
        "Opening and closing a zip",
        "Removing the coin from the change purse and leaving it on the table",
        "Catching and moving two different sized wooden cubes",
        "Lifting and moving an iron from one marked point to another",
        "Taking a screwdriver and turning a screw clockwise 360° with it",
        "Taking a nut and turning it until completely inserted inside the bolt",
        "Taking a key, placing it in a lock and turning it counter-clockwise 180°",
        "Turning a door handle 30°",
        "Tying a shoelace",
        "Unscrewing two lids and leaving them on the table",
        "Passing two buttons through their respective buttonhole using both hands",
        "Taking a bandage and putting it on his/her left arm up to the elbow",
        "Taking a knife with the right hand and a fork with the left hand and splitting a piece of clay (sitting)",
        "Taking a spoon with the right hand and using it 5 times to eat soup (sitting)",
        "Picking up a pen from the table, writing his/her name and putting the pen back on the table (sitting)",
        "Folding a piece of paper with both hands, placing it into an envelope and leaving it on the table (sitting)",
        "Taking a clip and putting it on the flap of the envelope (sitting)",
        "Writing with the keypad (sitting)",
        "Picking up the phone, placing it to his/her ear and hanging up the phone (sitting)",
        "Pouring 1L of water from a carton into a jug (sitting)",
        "Pouring water from the jug into the cup up to a marked point (sitting)",
        "Pouring the water from the cup back into the jug (sitting)",
        "Putting toothpaste on the toothbrush",
        "Using a spray over the table 5 times",
        "Cleaning the table with a cloth for 5 seconds"
    ]
    phase_descriptions = ["reaching", "manipulating", "releasing"]
    # train/test split obtained using np.random.choice
    train_ids = [0, 1, 3, 4, 5, 8, 10, 12, 13, 15, 16, 17, 18, 19, 20]
    test_ids = [2, 6, 7, 9, 11, 14, 21]
    num_features = 25
    sampling_period = 0.01
    variable_length = True

    def __init__(self, root, train=True, label="ADL", phase=None, download=True, **kwargs):
        """

        Parameters
        ----------
        root : str or Path
            Root directory of the dataset where KIN_MUS_UJI.mat exists.
        train : bool, optional
            Flag to set if the training set (True) or test set (False) will be 
            loaded, by default True. Train/test split is approximately 70/30,
            and is split by a random partition on the subjects ids, so that
            the test set has different subjects than the train set.  
        label : str
            String to indicate which parameter to use for the data label. This
            can be either "Phase" or the default "ADL".
        phase : int
            If label == "ADL", setting this value to 0, 1, or 2 will restrict
            signals to those matching the phase value. By default, this is None,
            allowing all phases.
        adl : int
            if label == "Phase", setting this value to an ADL id (0 - 25) will
            restrict signals to those matching the adl value. By default, this
            is None, allowing all ADL.
        noise : float, optional
            Optional noise level to add in terms of relative standard deviation.
        download : bool, optional
            Flag to download the data automatically if it's not found in the 
            root directory, by default True.
        **kwargs
            Arguments to pass on to TimeSeriesDataset
        """
        super().__init__(**kwargs)
        if label not in ["ADL", "Phase"]:
            raise ValueError("label argument must be either \"ADL\" or \"Phase\"")
        elif label == "ADL":
            self.label_names = self.adl_descriptions
        else:
            self.label_names = self.phase_descriptions
        self._train = train
        self.root = Path(root)
        if download:
            self.download()

        data = io.loadmat(
            self.root / self.MATFILE, struct_as_record=True, squeeze_me=True
            )['EMG_KIN_v4']
        
        # clean up - remove NaN data and start at 0 instead of 1
        badidx = np.flatnonzero([np.isnan(a).all() for a in data[:]['time']])
        data = np.delete(data, badidx)
        for k in ['Subject', 'ADL', 'Phase']:
            data[k] -= 1

        # get only the train or test set
        data_ids = self.train_ids if train else self.test_ids
        idx = np.isin(data[:]['Subject'].astype(np.int), data_ids)

        self.data = data[idx]
        self.fields = self.data.dtype.names
        
        # get the signals and labels for training
        self.signals = [
            torch.from_numpy(
                np.hstack([r["Kinematic_data"], r["EMG_data"]]).astype(np.float32)
                ) 
            for r in self.data
            ]
        self._num_samples = max([d.shape[0] for d in self.signals])
        self.phase = torch.from_numpy(self.data[:]["Phase"].astype(np.int64))
        self.adl = torch.from_numpy(self.data[:]["ADL"].astype(np.int64))
        self.labels = torch.from_numpy(self.data[:][label].astype(np.int64))

        # filter phase / adl
        if label == "ADL" and phase is not None:
            idx = np.flatnonzero(self.phase == phase)
            self.signals = [self.signals[i] for i in idx]
            self.phase = self.phase[idx]
            self.adl = self.adl[idx]
            self.labels = self.labels[idx]
        elif label == "Phase" and adl is not None:
            idx = np.flatnonzero(self.adl == adl)
            self.signals = [self.signals[i] for i in idx]
            self.phase = self.phase[idx]
            self.adl = self.adl[idx]
            self.labels = self.labels[idx]

    @property
    def train(self):
        return self._train

    def download(self):
        outfile = self.root / self.MATFILE
        if outfile.exists():
            return
        self.root = self.root / self.SUBDIR
        outfile = self.root / self.MATFILE
        if outfile.exists():
            return
        
        self.root.mkdir(parents=True, exist_ok=True)

        with urlopen(self.URL) as request:
            chunk_size = 4096
            total_size = request.info().get('Content-Length').strip()
            total_size = int(total_size)
            print("Downloading data, total size: {}KB".format(total_size//1024))
            buf = BytesIO()
            bytes_left = total_size
            nchunks = (total_size + chunk_size - 1) // chunk_size
            for _ in trange(nchunks):
                bytes_to_read = min(chunk_size, bytes_left)
                buf.write(request.read(bytes_to_read))
                bytes_left -= bytes_to_read

        with open(self.root / self.MATFILE, 'wb') as f:
            f.write(buf.getbuffer())

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        data = self.signals[idx]
        if self.noise > 0.0:
            data = data + self.noise*data.std()*torch.randn_like(data)
        return data, self.labels[idx]

    @property
    def num_samples(self):
        return self._num_samples