import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from torch.distributions.normal import Normal


class Nerve(nn.Module):
    """ Simplified nerve class for use with PyTorch """
    # NOTE: The values here are for reference and are considered constants. If 
    # they are changed for any reason, then the function fits must be updated.
    # ---------
    # ref list of currents (mA) that evoke an action potential for a 100us pulse
    # duration, current source 1mm away, a range of fiber diameters (um)
    FIBER_DIAMETERS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 15, 16, 17, 18, 19,
                       20]
    THRESHOLD_CURRENT_D = [2, 1.2, 0.85, 0.66, 0.53, 0.46, .4, 0.37, 0.34, 0.31,
                           0.292, 0.27, 0.26, 0.252, 0.245, 0.236, 0.226]
    # ref list of currents (mA) that evoke an action potential for a 20um fiber
    # diameter, current source 1mm away, a range of pulse durations (ms)
    PULSE_DURATIONS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
    THRESHOLD_CURRENT_PD = [1.075, 0.6, 0.3, 0.226, 0.18, 0.14, 0.127]
    MIN_FIBER_D = 0.1
    MAX_FIBER_D = 30.0

    def __init__(
            self, diameter_mean=11.0, diameter_std=4.0, rho=300.0, 
            spontaneous_rate_hz=1.0
        ):
        """ Create a nerve

        Parameters
        ----------
        diameter_mean : float, optional
            Average neuron diameter in microns, by default 11.0.
        diameter_std : float, optional
            Neuron diameter standard deviation in microns, by default 4.0.
        rho : float, optional
            Resistivity in kOhm-cm, by default 300.0.
        spontaneous_rate_hz : float, optional
            Spontaneous firing rate in Hz, by default 1.0
        """
        super().__init__()
        self.diameter_mean = diameter_mean
        self.diameter_std = diameter_std
        self.rho = rho
        self.spontaneous_rate_hz = spontaneous_rate_hz
        self.dm_t = nn.Parameter(torch.tensor([diameter_mean]), requires_grad=False)
        self.ds_t = nn.Parameter(torch.tensor([diameter_std]), requires_grad=False)
        self.f = Normal(self.dm_t, self.ds_t)
        self.v_to_ma = 400*math.pi/rho
    
    @staticmethod
    def get_threshold(diameter, pulse_duration_ms):
        """ Get the threshold activation current

        Parameters
        ----------
        diameter : float, ndarray, torch.Tensor
            The neuron diameter in microns.
        pulse_duration_ms : float
            The stimulation pulse duration in milliseconds.

        Returns
        -------
        Threshold activation current : same type as diameter
            The stimulating current required to activate the neuron.
        """
        # Coefficients we're found by fitting function 'fun'
        # to THRESHOLD_CURRENT_D = fun(FIBER_DIAMETERS) from 
        # nerve_models.electrical_activation with scipy.optimize.curve_fit
        i_d = -0.00805161*diameter - 3.84887702/math.sqrt(diameter) + \
            7.74579978/diameter + 0.8655029
        scale = i_d/0.226
        # pulse duration coefficients found with curve fit for
        # PULSE_DURATIONS and THRESHOLD_CURRENT_PD
        return (0.00951643/pulse_duration_ms + 0.12270154)*scale
    
    def get_diameter_from_potential(self, v, pulse_duration_ms=0.1):
        """ Get the diameter of a neuron that would activated

        Parameters
        ----------
        v : float, ndarray, torch.Tensor
            The stimulation voltage
        pulse_duration_ms : float, optional
            The sitmulation pulse duration in milliseconds, by default 0.1

        Returns
        -------
        same type as v
            The required neuron diameter to be activated by the supplied 
            stimulation voltage.
        """
        i = v*self.v_to_ma
        i_max = self.get_threshold(self.MIN_FIBER_D, pulse_duration_ms)
        i_min = self.get_threshold(self.MAX_FIBER_D, pulse_duration_ms)
        i = torch.clamp(i, i_min, i_max)
        # coefficients found with scipy.optimize.curve_fit for FIBER_DIAMETERS
        # and THRESHOLD_CURRENT_D from nerve_models.electrical_activation
        out = -1.47858995*i -1.33870859/i + 1.09223674/(i*i) + 5.30642146
        return torch.relu(out)

    def forward(self, v, pulse_duration_ms=0.1):
        """ Calculates the activation probability for a given stimulus 

        Parameters
        ----------
        v : torch.Tensor
            The stimulation potential in volts.
        pulse_duration_ms : float, optional
            The stimulating pulse duration in ms, by default 0.1.
        """
        d = self.get_diameter_from_potential(v, pulse_duration_ms)
        return 1.0 - self.f.cdf(d)
