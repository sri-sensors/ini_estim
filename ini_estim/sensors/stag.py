import logging
import pickle
from pathlib import Path

import numpy as np
from scipy import interpolate as interp

from ini_estim import DATA_DIR
from ini_estim.utilities.dataset_utils import MatReader


class TactileGloveSensor:
    """Class to load & manage data from MIT Scalable Tactile Glove (STAG)
    sensor array
    """

    NAMES = {"blindfolded_lite", "classification_lite", "handposes_lite"}

    def __init__(self, name='classification_lite', scaling='linear'):
        """Initializes TactileGloveSensor object"""
        self.base_path = Path(DATA_DIR).joinpath('stag')
        self.n_sensors = 548
        self.offset = 510
        self.scaling = scaling
        self.metadata = None
        self.load_metadata(name)
        self.sensor_idx = None
        self.get_sensor_idx()


    @property
    def xind(self):
        return self.sensor_idx[0]

    @property
    def yind(self):
        return self.sensor_idx[1]

    @property
    def num_recordings(self):
        return len(self.metadata['recordings'])

    def load_metadata(self, name='handposes_lite'):
        """Loads metadata
        :arg name: selects data directory (Directories class instance)
        """
        self.metadata = MatReader().loadmat(self.base_path.joinpath(name).
                                            joinpath('metadata.mat'))

    def print_recordings(self):
        for r in self.metadata['recordings']:
            print(r)

    @property
    def recordings(self):
        return self.metadata['recordings']

    def get_pressure_data(self, rec_id, t_resample=None):
        """Returns the sensor data for a given recording id
        :arg rec_id - numeric value for recording id
        :returns p array of pressure data sized (t, 32, 32)
        :returns t_data array of times sampled sized (t)
        """

        idx = self.metadata['recordingId'] == rec_id
        p = self.metadata['pressure'][idx]
        t_data = self.metadata['ts'][idx]
        if self.scaling == 'linear':
            max_pressure = np.amax(p)
            p = (p - self.offset)/(max_pressure - self.offset)
        logging.getLogger(__name__).info(
            'Sampled at {:0.2f} Hz'.format(1/np.mean(np.diff(t_data))))
        pdata = p[:, self.xind, self.yind]
        if t_resample is None:
            return pdata, t_data
        else:
            func = interp.interp1d(
                t_data, pdata, axis=0, fill_value=pdata[-1,:], bounds_error=False
            )
            tr = np.arange(t_data[0], t_data[-1] + t_resample, t_resample)
            pr = func(tr)
            return pr, tr

    def get_sensor_idx(self):
        filename = self.base_path.joinpath('sensor_coords.pickle')
        with open(filename, 'rb') as f:
            self.sensor_idx = pickle.load(f)