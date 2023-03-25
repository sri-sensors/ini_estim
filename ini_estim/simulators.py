from logging import getLogger
import numpy as np
from scipy.sparse import coo_matrix
import scipy.interpolate as interp
from typing import Union, Tuple, List

import ini_estim.mutual_information.histogram
import ini_estim.sensors as sensors
import ini_estim.encoders as encoders
import ini_estim.nerves as nerves
import ini_estim.sensors.generic
import ini_estim.stimulus as stimulus
import ini_estim.electrodes as electrodes
import ini_estim.nerve_models as nerve_models
import ini_estim.utilities.sampling as sampling

LOG = getLogger(__name__)


class Simulator:
    def __init__(self,
                 electrode_array: electrodes.GenericElectrodeCuff ,
                 encoder: encoders.SensorEncoder,
                 nerve_model: nerve_models.Nerve):
        self.electrode_array = electrode_array
        self.encoder = encoder
        self.nerve_model = nerve_model
        self.sampling_locations = np.array([[0, 0]])
        self.sensor_oversample = 1

    def generate_uniform_sampling_locations(self, num_locations=50):
        """ Generate neuron sampling locations

        Saves the locations internally to use for generating spikes or
        probability maps.

        Parameters
        ----------
        num_locations
            The number of sampling locations to generate

        Returns
        -------
        The sampling locations, shape = (num_locations, 2)
        """
        x, y = sampling.uniform_points_circle(
            num_locations, self.electrode_array.DIAMETER_MM
        )
        self.sampling_locations = np.concatenate(
            (x[:, None], y[:, None]), axis=1
        )
        return self.sampling_locations

    def get_probability_response(self, sensor_values, sample_rate, resample=True):
        """ Generate the probability response for each neuron

        Parameters
        ----------
        sensor_values
            The sensor values to feed the encoder. Sensor values are assumed
            to have dimensions [num_samples, num_sensors]
        sample_rate
            The sample rate of the sensor values. Sensor values will be
            resampled by linear interpolation to Simulator.sensor_oversample *
            encoder.sample_rate
        resample
            If True (default), sensor values will be resampled by linear
            interpolation to Simulator.sensor_oversample * encoder.sample_rate
        """
        u_stim = self.encode(sensor_values, sample_rate, resample)
        p_nerve = self.nerve_model.get_activation_probability(u_stim)
        return p_nerve

    def get_spike_response(self, sensor_values, sample_rate, resample=True):
        """ Generate the spike response for each neuron

        Parameters
        ----------
        sensor_values
            The sensor values to feed the encoder. Sensor values are assumed
            to have dimensions [num_samples, num_sensors]
        sample_rate
            The sample rate of the sensor values. Sensor values will be
            resampled by linear interpolation to Simulator.sensor_oversample *
            encoder.sample_rate
        resample
            If True (default), sensor values will be resampled by linear
            interpolation to Simulator.sensor_oversample * encoder.sample_rate
        """
        u_stim = self.encode(sensor_values, sample_rate, resample)
        spikes = self.nerve_model.get_spikes(
            u_stim, dt=1000.0/self.encoder.sample_rate
        )
        return spikes

    def encode(self, sensor_values, sample_rate, resample=True):
        """ Encode sensor values

        Parameters
        ----------
        sensor_values
            The sensor values to feed the encoder. Sensor values are assumed
            to have dimensions [num_samples, num_sensors]
        sample_rate
            The sample rate of the sensor values. Sensor values will be
            resampled by linear interpolation to Simulator.sensor_oversample *
            encoder.sample_rate
        resample
            If True (default), sensor values will be resampled by linear
            interpolation to Simulator.sensor_oversample * encoder.sample_rate
        """
        if resample:
            Tsensor = sensor_values.shape[0] / sample_rate
            f = interp.interp1d(
                np.arange(0, Tsensor, 1 / sample_rate),
                sensor_values, 'linear', 0, copy=False, bounds_error=False,
                fill_value=sensor_values[-1, :], assume_sorted=True
            )
            new_sample_rate = self.sensor_oversample * self.encoder.sample_rate
            tn = np.arange(0, Tsensor, 1 / new_sample_rate)
            sensor_values = f(tn)
            sample_rate = new_sample_rate
        u_encoder = self.encoder.encode_series(sensor_values, sample_rate, True)
        electrode_weights = self.electrode_array.get_potential(
            self.sampling_locations[:, 0],
            self.sampling_locations[:, 1]
        )
        electrode_weights = np.column_stack(electrode_weights).transpose()
        u_stim = u_encoder @ electrode_weights
        return u_stim


class SimulatorOld:
    """Class to run the full simulation.

    Instantiates Sensor, Encoder, NeuralStimulator, Nerve objects and
    orchestrates how they run together.
    """
    def __init__(self,
                 encoder: encoders.Encoder,
                 stimulator: stimulus.NeuralStimulator,
                 nerve: nerves.Nerve):
        """Initializes Simulator class"""
        self.sensors = []   # type: List[sensors.AbstractSensor]
        self.connectivity = []
        self.encoder = encoder
        self.stimulator = stimulator
        self.nerve = nerve
        self._last_input_data = None

    def add_sensor(self,
                   sensor: ini_estim.sensors.generic.AbstractSensor,
                   electrode_indexes: Union[int, List[int]],
                   connectivity_weights: Union[float, List[float]]):
        """ Add a sensor to the simulator.

        :param sensor: A sensor
        :param electrode_indexes: The electrode or list of electrodes that
            the sensor is connected to.
        :param connectivity_weights: The weight or list of weights that the
            sensor is connected to the electrode(s). Weights are usually
            between 0 and 1.
        :return:
        """
        self.sensors.append(sensor)
        sensor_index = len(self.sensors) - 1
        if (isinstance(electrode_indexes, int) and
                isinstance(connectivity_weights, float)):
            self.connectivity.append(
                [sensor_index, electrode_indexes, connectivity_weights]
            )
        elif len(electrode_indexes) == len(connectivity_weights):
            for e, c in zip(electrode_indexes, connectivity_weights):
                self.connectivity.append(
                    [sensor_index, e, c]
                )
        else:
            raise ValueError(
             "Invalid arguments for electrode_indexes or "
             "connectivity_weights."
            )

        return sensor_index

    def remove_sensor(self, sensor_index):
        raise NotImplementedError

    @property
    def electrodes(self):
        return self.stimulator.cfg.electrodes

    def run(self, duration, neuron_idx=None):
        """Runs simulation

        At each timestep: Draw sample, pass to the encoder, propagate result to
        electrode array, run a neural response model.

        :param duration: The duration in seconds.
        """
        if not len(self.sensors):
            raise RuntimeError("No sensors have been defined!")

        dt = self.nerve.dt
        sample_rate = 1e3/dt

        # generate all the sensor data at the nerve sample rate.
        for s in self.sensors:
            s.f_sample = sample_rate
        sdata = np.array([s.generate(duration) for s in self.sensors])

        # map to the electrodes
        weight_indexes = np.array(self.connectivity)
        w = weight_indexes[:,2].ravel()     # weights
        r = weight_indexes[:,1].ravel()     # electrodes
        c = weight_indexes[:,0].ravel()     # sensors
        wmshape = (len(self.electrodes), len(self.sensors))
        weight_matrix = coo_matrix((w, (r, c)), shape=wmshape)
        sdata_mapped = weight_matrix.dot(sdata)

        # encode
        sdata_encoded = [self.encoder.encode_sample(s) for s in sdata_mapped]

        stim = [
            stimulus.StimulationSequence(e, a, f)
            for e, (f, a) in enumerate(sdata_encoded)
        ]
        if neuron_idx is None:
            neuron_idx = self.stimulator.choose_neuron_coordinates_randomly(
                self.nerve.num_neurons, 0)
        self.stimulator.apply_multi_stimulus(
            self.nerve, neuron_idx, stim
        )
        self._last_input_data = sdata
        return neuron_idx, sdata

    def finish(self, sensor_bins=5):
        """
        Signals that simulation is over and a score needs to be computed.

        :param int sensor_bins: The number of histogram bins to use for the
            sensor data. The default value is 5.
        :return:
            entropy_of_sensor_data (bits/second)
            entropy_of_spike_data (bits/second)
            mutual_info_sensor_and_spike (bits/second)
        """
        sdata = self._last_input_data
        if sdata is None:
            raise RuntimeError("No data to compute mutual information!")

        sdata = self._last_input_data.transpose()
        # spike trains are stored as time indexes for each nerve

        spike_trains = [s[1] for s in self.nerve.spike_trains]
        spikes = np.zeros((sdata.shape[0], len(spike_trains)))
        for i, s in enumerate(spike_trains):
            spikes[s,i] = 1

        sample_rate = 1000.0/self.nerve.dt
        edata = ini_estim.mutual_information.histogram.entropy(sdata, sensor_bins) * sample_rate
        espikes = ini_estim.mutual_information.histogram.entropy(spikes, 2) * sample_rate
        info = ini_estim.mutual_information.histogram.hist_mutual_info(sdata, spikes, bins=[sensor_bins, 2]) * sample_rate
        return edata, espikes, info



        # mi.spike_triggered_average_multi(
        #     self.sensor.data,
        #     self.nerve.spike_trains,
        #     filter_length=250)


        # From Nora's routine to compute mutual info.
        # self.calc_temporal_sta(stimulus, spikes, filter_length)
        # self.calc_nl_and_noise(stimulus, spikes, window, nbins)
        # self.calc_info(L=1000)