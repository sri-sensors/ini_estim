"""
Classes for generating sample sensor data

"""
import numpy as _np
import scipy.interpolate as _interp

from .generic import AbstractSensor, PressureSensor
from .stag import TactileGloveSensor


def resample_data(t, data, sample_freq):
    """Resamples a dataset to the target sampling frequency.

    Args:
        sample_freq Desired sampling frequency, Hz. """
    f_spline = _interp.interp1d(t, data, kind='cubic')
    t_new = _np.linspace(0, t[-1], t[-1] * sample_freq)
    return t_new, f_spline(t_new)

