import logging
import re
import numpy as np
from pathlib import Path
import torch
import torch.utils.data as data
from tqdm.auto import tqdm, trange
from .base import TimeSeriesDataset


LOGGER = logging.getLogger(__name__)


def get_line_count(filename):
    line_count = 0
    with open(filename) as f:
        for l in f:
            line_count += 1
    return line_count


class LukeHandDataset(TimeSeriesDataset):
    DATA_DIR = "SRI LUKE Hand Dataset"
    SUB_DIR = "deka_data"
    SIGNAL_FILES = ['index_lateral_force_sensor',
                    'index_tip_force_sensor',
                    'middle_tip_force_sensor',
                    'ring_tip_force_sensor',
                    'pinky_tip_force_sensor',
                    'distal_palm_force_sensor',
                    'proximal_palm_force_sensor',
                    'hand_edge_force_sensor',
                    'hand_dorsal_force_sensor',
                    'thumb_ulnar_force_sensor',
                    'thumb_radial_force_sensor',
                    'thumb_tip_force_sensor',
                    'thumb_dorsal_force_sensor',
                    'thumb_pitch_position',
                    'thumb_roll_position',
                    'wrist_rotator_position',
                    'wrist_flexor_position',
                    'index_finger_position',
                    'mrp_fingers_aggregate_position']
    LABEL_NAME_FILES = {
        "grasp": "grasp_labels.txt",
        "object": "object_labels.txt"
    }
    TRAIN_DIR = 'train/sensor_position_signals'
    TRAIN_LABELS = {
        "grasp": "train/grasp_info_train.txt",
        "object": "train/object_info_train.txt"
    }
    TEST_DIR = 'test/sensor_position_signals'
    TEST_LABELS = {
        "grasp": "test/grasp_info_test.txt",
        "object": "test/object_info_test.txt"
    }
    variable_length = True
    SIGNAL_RANGES = np.array([   
                        [ 0.000000e+00,  2.200000e+01],
                        [ 0.000000e+00,  2.230000e+01],
                        [ 1.000000e-01,  1.090000e+01],
                        [ 1.000000e-01,  7.900000e+00],
                        [ 1.000000e-01,  5.000000e+00],
                        [ 0.000000e+00,  2.550000e+01],
                        [ 0.000000e+00,  5.800000e+00],
                        [ 0.000000e+00,  2.550000e+01],
                        [ 0.000000e+00,  2.550000e+01],
                        [ 0.000000e+00,  1.800000e+00],
                        [ 0.000000e+00,  2.320000e+01],
                        [ 0.000000e+00,  2.550000e+01],
                        [ 0.000000e+00,  3.400000e+00],
                        [-1.418100e-02,  1.712604e+00],
                        [-7.636000e-03,  1.699787e+00],
                        [-4.652390e-01,  4.281500e-02],
                        [-3.081600e-02,  4.936010e-01],
                        [-5.617800e-02,  1.323996e+00],
                        [-1.036300e-02,  1.315542e+00]
                        ], dtype=np.float32)


    def __init__(self, root, train=True, label="object", **kwargs):
        """Class to hold sensor data from the LUKE hand.
        Parameters
        ----------
        root : str or Path
            Root directory of the dataset where "train/y_train.txt"
            and "/train/y_test.txt" exist.
        train : bool
            If True (default), the training set will be loaded, otherwise the
            test set will be loaded.
        label : str
            "grasp" - use grasp as label
            "object" - use object as label
        **kwargs
            Arguments to pass on to TimeSeriesDataset

        """
        super().__init__(**kwargs)
        self.train = train
        self.signals = None
        self.label = label.lower()
        self.labels = None
        self.root = Path(root)
        self.grasp = None
        self.grasp_names = None
        self.object = None
        self.object_names = None

        self.load()

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        data = torch.from_numpy(self.signals[idx])
        if self.noise > 0.0:
            data = data + self.noise*data.std()*torch.randn_like(data)
        return data, self.labels[idx]

    @property
    def num_features(self):
        return len(self.SIGNAL_FILES)

    def load(self):
        """Loads sensor data from root"""
        if (self.root.name.lower() != self.DATA_DIR.lower()):
            data_loc = self.root / self.DATA_DIR
        else:
            data_loc = self.root

        if not (data_loc.exists()):
            self.root = self.root / self.SUB_DIR
            data_loc = self.root / self.DATA_DIR
        LOGGER.info("Loading data from {}".format(self.root))
        
        if self.train:
            data_dir = self.TRAIN_DIR
            label_file = self.TRAIN_LABELS[self.label]
        else:
            data_dir = self.TEST_DIR
            label_file = self.TEST_LABELS[self.label]
        label_name_file = self.LABEL_NAME_FILES[self.label]

        suffix = ".txt"
        
        print("Loading...")
        self.label_names = self._get_label_names(data_loc, label_name_file)
        self.labels = self._get_labels(data_loc, label_file)
        self.num_labels = len(self.label_names)
        # get all the labels just in case ...
        self.grasp_names = self._get_label_names(
            data_loc, self.LABEL_NAME_FILES.get("grasp")
            )
        label_dict = self.TRAIN_LABELS if self.train else self.TEST_LABELS
        self.grasp = self._get_labels(data_loc, label_dict["grasp"])
        self.object_names = self._get_label_names(
            data_loc, self.LABEL_NAME_FILES.get("object")
            )
        self.object = self._get_labels(data_loc, label_dict["object"])
        self.signals = []

        # Get the line count first.
        line_count = get_line_count(
            data_loc / Path(data_dir) / (self.SIGNAL_FILES[0] + suffix)
            )
        
        file_handles = [
                open(data_loc / Path(data_dir) / (s + suffix)) 
                for s in self.SIGNAL_FILES
            ]
        try:
            for i in trange(line_count):
                tmp = []
                for f in file_handles:
                    tmp.append(np.fromstring(f.readline(), sep=" ", dtype=np.float32))
                tmp = np.column_stack(tmp)

                self.signals.append(self._normalize(tmp))
        finally:
            for f in file_handles:
                f.close()        

    def _get_labels(self, data_loc, label_file):
        with open(data_loc / label_file) as f:
            labels = torch.from_numpy(np.loadtxt(f, dtype=np.int64) - 1)
        return labels

    def _get_label_names(self, data_loc, label_name_file):
        regex = re.compile("(\d+) [a-zA-Z_]")
        with open(data_loc / label_name_file, 'r') as f:
            label_lines = [l.strip() for l in f if re.match(regex, l.strip())]
        label_names = ['']*len(label_lines)
        for l in label_lines:
            idx, label = l.split(' ')
            label_names[int(idx)-1] = label
        return label_names
    
    def _normalize(self, data_sample):
        r = np.diff(self.SIGNAL_RANGES).transpose()
        data_sample = (data_sample - (self.SIGNAL_RANGES[:,0][None,:]))
        data_sample /= r
        return data_sample

