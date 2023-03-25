import enum

import numpy as np
from scipy.signal import lfilter

from ini_estim.signal_processing import butter_lpf


class AbstractSensor:

    f_sample = 1000.0

    def generate(self, t: float):
        """
        :param t: Duration in seconds
        :return: numpy array of length int(T*self.f_sample)
        """
        raise NotImplementedError


class PressureSensor(AbstractSensor):
    class Modes(enum.IntEnum):
        RANDOM = 0
        PULSE = 1
        RAMP = 2
        SINE = 3

    def __init__(self, val_min: float, val_max: float, f_max: float,
                 noise_std: float, mode=None):
        """

        Create a pressure sensor.
        Default properties:
            f_sample = 2000.0
            mode = PressureSensor.Modes.RANDOM

        :param val_min: minimum value possible
        :param val_max: maximum value possible
        :param f_max: maximum frequency possible
        :param noise_std: standard deviation of noise
        :param mode: default mode (PressureSensor.Modes)
        """
        self.val_min = val_min
        self.val_max = val_max
        self.f_max = f_max
        self.noise_std = noise_std
        self.f_sample = 2000.0
        self.phi = 0.0
        if mode is None:
            self._mode = self.Modes.RANDOM
        else:
            self._mode = mode

    def set_mode(self, mode: Modes):
        """ Set the operating mode for the pressure sensor

        :param mode:
        """
        self._mode = mode

    def get_mode(self):
        return self._mode

    @property
    def range(self):
        return self.val_max - self.val_min

    def band_limit_signal(self, sig, f_sample):
        b, a = butter_lpf(self.f_max, f_sample)
        output = lfilter(b, a, sig)
        return output

    def generate(self, T):
        """
        Generate samples according to the configured parameters
            (mode and f_sample)

        :param T: Total observation length (seconds)
        :return: numpy array of length int(T*self.f_sample) of configured
            (RANDOM or PULSE) samples
        """
        if self._mode == self.Modes.RANDOM:
            return self.generate_random_samples(T)
        elif self._mode == self.Modes.PULSE:
            return self.generate_pulse(T, np.random.rand()*T)
        elif self._mode == self.Modes.RAMP:
            return self.generate_ramp(T)
        elif self._mode == self.Modes.SINE:
            return self.generate_sine(T)
        else:
            raise NotImplementedError

    def generate_sine(self, T, f_sample=None):
        """ Generate a sine

        :param T: Total observation length (seconds)
        :param f_sample: Sample rate (Hz). Defaults to internal f_sample.
        :return: numpy array of length int(T*f_sample)
        """
        if f_sample is None:
            f_sample = self.f_sample

        Nsamples = int(T*f_sample)
        t = np.arange(Nsamples)/f_sample
        out = np.sin(2*np.pi*self.f_max*t + self.phi)*0.5 + 0.5
        out = self.range*out + self.val_min
        return out

    def generate_ramp(self, T, f_sample=None):
        """ Generate a linear ramp

        :param T: Total observation length (seconds)
        :param f_sample: Sample rate (Hz). Defaults to internal f_sample.
        :return: numpy array of length int(T*f_sample)
        """
        if f_sample is None:
            f_sample = self.f_sample

        Nsamples = int(T*f_sample)
        return np.linspace(self.val_min, self.val_max, Nsamples)

    def generate_random_samples(self, T, f_sample=None):
        """ Generate random samples

        :param T: Total observation length (seconds)
        :param f_sample: Sample rate (Hz). Defaults to internal f_sample.
        :return: numpy array of length int(T*f_sample) of random samples
        """
        if f_sample is None:
            f_sample = self.f_sample
        Nsamples = int(T*f_sample)
        Nbw = int(max(f_sample/self.f_max, 1))
        Nbw = (Nsamples+Nbw-1)//Nbw
        output_bw = self.range*np.random.rand(Nbw) + self.val_min
        output = np.interp(np.arange(Nsamples),
                           np.linspace(0, Nsamples, Nbw, False),
                           output_bw)
        output += np.random.randn(Nsamples)*self.noise_std
        output = self.band_limit_signal(output, f_sample)
        return output

    def generate_pulse(self, T_total, T_pulse, A_rel=None, f_sample=None):
        """ Generate a pulse

        :param T_total: Total observation time
        :param T_pulse: Length of pulse (s)
        :param A_rel: Amplitude relative to full scale range (0 to 1), set to
            None (default behavior) to let A_rel be a random value.
        :param f_sample: Sample rate (Hz). Defaults to internal f_sample.
        :return: numpy array of length int(T*f_sample)
        """
        if A_rel is None:
            A_rel = np.random.rand()
        else:
            A_rel = np.clip(A_rel, 0, 1)

        if f_sample is None:
            f_sample = self.f_sample

        Nsamples = int(T_total*f_sample)
        Npulse = int(T_pulse*f_sample)
        Npulse = np.clip(Npulse, 0, Nsamples)
        ndiff = Nsamples - Npulse
        offset = ndiff//2
        output = np.zeros(Nsamples)
        output[offset:offset+Npulse] = A_rel
        output = output*self.range + self.val_min
        output += np.random.randn(Nsamples)*self.noise_std
        output = self.band_limit_signal(output, f_sample)
        return output