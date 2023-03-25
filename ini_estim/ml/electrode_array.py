import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import ini_estim.electrodes as _electrodes
from ini_estim.ml.nerve import Nerve
from typing import Union


class ElectrodeArray(nn.Module):
    """ElectrodeArray

    Simulates an electrode array and returns nerve responses.

    Attributes
    ----------
    num_sampling_points : int
        The number of sampling points within the nerve. This is also the number
        of output features.
    total_neurons : int
        The total number of neurons represented in the nerve.
    neurons_per_sample : float
        The equivalent number of neurons represented by each output feature.
    x, y : np.ndarray
        The x and y locations within the nerve of each sampling point.
    update_rate : float
        The update rate of the electrode array. This parameter is used to
        scale the noise level at the output (adding uncertainty to the
        spike count), by default 30Hz.
    positive_current_only : bool
        Enforce input current to always be positive, by default False.

    """
    ELECTRODE_TYPES = _electrodes.electrode_arrays
    def __init__(
        self, electrode_array, num_sampling_points=500, nerve=None,
        update_rate=30.0, input_scale=1.0, positive_current_only=False
    ):
        """ Create an electrode array

        Parameters
        ----------
        electrode_array: str or GenericElectrodeCuff
            The electrode array. This can either be a string or an instance of
            GenericElectrodeCuff. If a string, then ElectrodeArray will retrieve
            the class from ini_estim.electrodes.electrode_arrays.
        num_sampling_points : int
            The number of sampling points within the array, by default 500.
            ElectrodeArray will evenly distribute the points within the
            spatial bounds. 
        nerve : ini_estim.ml.Nerve
            The nerve to use for generating the output response. By default or
            if set to None, ElectrodeArray will use a Nerve instance with the
            default parameters.
        update_rate : float
            The update rate of the electrode array. This parameter is used to
            scale the noise level at the output (adding uncertainty to the
            spike count), by default 30Hz.
        input_scale : float
            pre-scale the input by input_scale, by default 1.0.
        positive_current_only : bool
            Enforce input current to always be positive, by default False.
        """
        super().__init__()
        if isinstance(electrode_array, str):
            electrode_array = _electrodes.electrode_arrays[electrode_array]
            self.electrode_array = electrode_array()
        else:
            self.electrode_array = electrode_array
        
        self.positive_current_only = positive_current_only
        self.x, self.y = self.electrode_array.get_uniform_points(
            num_sampling_points)
        
        weights = torch.from_numpy(np.dstack(
                    self.electrode_array.get_potential(self.x, self.y)
                ).squeeze().astype(np.float32)
            )
        
        # scale by the current reference to get mA scale:
        weights *= 1e3 / self.electrode_array.CURRENT_REFERENCE_UA

        if weights.dim() == 1:
            weights = weights.view((-1, 1))
        self.ma_to_v = nn.Parameter(weights, requires_grad=False)
        self.input_scale = nn.Parameter(torch.tensor([input_scale]), requires_grad=False)
        self.nerve = Nerve() if nerve is None else nerve

        # compute neurons per sampling point assuming hexagonal packing
        neuron_area = math.pi*(self.nerve.diameter_mean*1e-3*0.5)**2
        self.total_neurons = math.floor((self.electrode_array.area/neuron_area)*0.74)
        self.num_sampling_points = weights.shape[0]
        self.neurons_per_sample = self.total_neurons/num_sampling_points
        self.update_rate = update_rate
        self.in_features = weights.shape[1]
        self.out_features = weights.shape[0]

        self.return_probablities = False
    
    def forward(self, input_current_ma):
        """ Get the neuron responses within the electrode array

        Parameters
        ----------
        input_current_ma : torch.Tensor
            The input current to supply.

        Returns
        -------
        torch.Tensor
            The equivalent spike count for each sampling point. 
        """
        
        if self.positive_current_only:
            input_current_ma = torch.relu(input_current_ma)
        volts = F.linear(self.input_scale*input_current_ma, self.ma_to_v, None)
        p = self.nerve(volts)
        if self.return_probablities:
            return p
            
        neuron_factor = self.neurons_per_sample/self.update_rate
        noise_rate = math.sqrt(self.nerve.spontaneous_rate_hz*neuron_factor)
        out = p*self.neurons_per_sample
        noise = torch.poisson(torch.full_like(out, noise_rate))
        return out + noise
