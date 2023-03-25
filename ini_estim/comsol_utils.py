import numpy as np
import pandas as pd
from pathlib import Path
import os

from ini_estim import DATA_DIR

basepath = Path(DATA_DIR) / 'comsol_data'
f_name = '14elcuff_singleelon_1mAthrough_approx5x5mm2.txt'
FILENAME = basepath / f_name

base2d = Path(DATA_DIR) / 'cuff_fem' / 'exports'
f_name2d = '14elcuff_singleelon_1mAthrough_el5.txt'
FILENAME_2D = base2d / f_name2d

BASEDIR_USEA = Path(DATA_DIR) / 'USEA'

USEA_ELEC_COORDS = {
    1: (-1.8, -1.4),
    2: (-1.4, -1.4),
    3: (-1, -1.4),
    4: (-0.6, -1.4),
    5: (-0.2, -1.4),
    6: (0.2, -1.4),
    7: (0.6, -1.4),
    8: (1, -1.4),
    9: (1.4, -1.4),
    10: (1.8, -1.4),
    11: (-1.8, -1),
    12: (-1.4, -1),
    13: (-1, -1),
    14: (-0.6, -1),
    15: (-0.2, -1),
    16: (0.2, -1.0),
    17: (0.6, -1.0),
    18: (1.0, -1.0),
    19: (1.4, -1.0),
    20: (1.8, -1.0),
    21: (-1.8, -0.6),
    22: (-1.4, -0.6),
    23: (-1.0, -0.6),
    24: (-0.6, -0.6),
    25: (-0.2, -0.6),
    26: (0.2, -0.6),
    27: (0.6, -0.6),
    28: (1.0, -0.6),
    29: (1.4, -0.6),
    30: (1.8, -0.6),
    31: (-1.8, -0.2),
    32: (-1.4, -0.2),
    33: (-1.0, -0.2),
    34: (-0.6, -0.2),
    35: (-0.2, -0.2),
    36: (0.2, -0.2),
    37: (0.6, -0.2),
    38: (1.0, -0.2),
    39: (1.4, -0.2),
    40: (1.8, -0.2),
    41: (-1.8, 0.2),
    42: (-1.4, 0.2),
    43: (-1.0, 0.2),
    44: (-0.6, 0.2),
    45: (-0.2, 0.2),
    46: (0.2, 0.2),
    47: (0.6, 0.2),
    48: (1.0, 0.2),
    49: (1.4, 0.2),
    50: (1.8, 0.2),
    51: (-1.8, 0.6),
    52: (-1.4, 0.6),
    53: (-1.0, 0.6),
    54: (-0.6, 0.6),
    55: (-0.2, 0.6),
    56: (0.2, 0.6),
    57: (0.6, 0.6),
    58: (1.0, 0.6),
    59: (1.4, 0.6),
    60: (1.8, 0.6),
    61: (-1.8, 1.0),
    62: (-1.4, 1.0),
    63: (-1.0, 1.0),
    64: (-0.6, 1.0),
    65: (-0.2, 1.0),
    66: (0.2, 1.0),
    67: (0.6, 1.0),
    68: (1.0, 1.0),
    69: (1.4, 1.0),
    70: (1.8, 1.0),
    71: (-1.8, 1.4),
    72: (-1.4, 1.4),
    73: (-1.0, 1.4),
    74: (-0.6, 1.4),
    75: (-0.2, 1.4),
    76: (0.2, 1.4),
    77: (0.6, 1.4),
    78: (1.0, 1.4),
    79: (1.4, 1.4),
    80: (1.8, 1.4),
    81: (-1.8, 1.8),
    82: (-1.4, 1.8),
    83: (-1.0, 1.8),
    84: (-0.6, 1.8),
    85: (-0.2, 1.8),
    86: (0.2, 1.8),
    87: (0.6, 1.8),
    88: (1.0, 1.8),
    89: (1.4, 1.8),
    90: (1.8, 1.8),
}


def load_usea_data(electrode_index, cx_mm=0, cy_mm=0, data_to_mm=1):
    """Loads the USEA data for a given electrode intex. Returns the
    xy-coordinates (in mm, if a value for data_to_mm is given) and the
    potential (mv) at those coordinates, with the option of re-centering the
    array at 0, 0.
    """
    file_name = 'VfromTerminal{}.gz'.format(electrode_index)
    full_name = os.path.join(BASEDIR_USEA, file_name)
    fem_data = pd.read_csv(full_name)

    x = fem_data['x'].to_numpy() * data_to_mm - cx_mm
    y = fem_data['y'].to_numpy() * data_to_mm - cy_mm
    v = fem_data['v'].to_numpy()
    return x, y, v


def load_comsol_2d(electrode_index):
    fn = '14elcuff_singleelon_1mAthrough_el{}.txt'.format(electrode_index)
    full_name = base2d / fn
    data = pd.read_csv(full_name, delim_whitespace=True,
                       skiprows=9, header=None, names=['x', 'y', 'z', 'v'])
    return data


def load_comsol_data(filename=FILENAME):
    """Loads exported COMSOL potential data and returns x-y-z oords and
    corresponding potential values
    :param: filename: complete path to .txt file with 3d data
    """
    data = pd.read_csv(filename, delim_whitespace=True,
                       skiprows=9, header=None, names=['x', 'y', 'z', 'v'])
    return data


def get_2d_potential(data, z_location=None):
    """Returns the 2d potential data at the specified z_location
    If z_location not specified, returns the location with the highest
    potential
    :param: data: 3d comsol data (from load_comsol_data)
    :param: z_location: position along z to return the 2d map
    """
    if z_location is None:
        z_location = data['z'][data['v'].max() == data['v']]
        data2d = np.asarray(data[data['z'] == z_location.iloc[0]])
    else:
        data2d = np.asarray(data[data['z'] == z_location])

    x = np.reshape(data2d[:, 0], (80, 80))
    y = np.reshape(data2d[:, 1], (80, 80))
    v = np.reshape(data2d[:, 3], (80, 80))
    return v, x, y


def rotate_coordinates(x, y, theta):
    """rotates the specified coordinates by angle theta (in radians)"""
    c, s = np.cos(theta), np.sin(theta)
    rotation_m = np.array([[c, -s], [s, c]])
    return np.dot(rotation_m, np.array([x, y]))


if __name__ == "__main__":
    print("See Jupyter notebooks for plotting demo (e.g., USEA demo.ipynb)")
