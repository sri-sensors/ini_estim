import attr
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import random

from ini_estim.deprecated import activation
from ini_estim.nerve_models.nerve import Nerve
from ini_estim.nerve_models.electrical_activation import ElectricalActivation
from ini_estim import comsol_utils
from ini_estim import nerves
from ini_estim.signal_processing import apply_butter_lpf_sym

PLOTTING = True


@attr.s(auto_attribs=True)
class NeuralStimulatorConfig(object):
    """A simple class to hold neural stimulator data."""
    electrodes: List[int]
    frequency_min: float
    frequency_max: float
    amplitude_min: float
    amplitude_max: float
    pulse_width_min: float
    pulse_width_max: float
    update_rate: float
    reference_potential_map: List[np.ndarray]
    reference_potential_xc: np.ndarray
    reference_potential_yc: np.ndarray


@attr.s(auto_attribs=True)
class StimulationSequence:
    """ Class for stimulation parameters """
    electrode: int
    amplitudes: np.ndarray
    frequencies: np.ndarray

@attr.s(auto_attribs=True)
class StimulationOneShot(object):
    """A simple class to hold stimulation pulse data"""
    electrode: int
    stim_amplitude: float
    pulse_width: float
    frequency: float
    duration: float


def _load_fem_for_round_cuff():
    """Loads the FEM for the round cuff

    Returns a dictionary with keys reference_potential_map,
    reference_potential_xc, reference_potential_yc.
    """
    fem_data = comsol_utils.load_comsol_2d(0)
    # keep only the values within the nerve diameter (=3mm) THIS IS
    # CURRENTLY HARD-CODED TO MATCH THE COMSOL FEM DATA
    valid_indices = np.sqrt(
        fem_data['x'] ** 2 + fem_data['y'] ** 2) < 1.5 * 1e-3
    xc = fem_data['x'][valid_indices].values * 1e3  # Scaled to mm
    yc = fem_data['y'][valid_indices].values * 1e3  # Scaled to mm
    v = [fem_data['v'][valid_indices].values * 1e3]  # mV

    for electrode_idx in range(1, 14):
        # load the ref potential map for the electrode index
        fem_data = comsol_utils.load_comsol_2d(electrode_idx)
        v.append(fem_data['v'][valid_indices].values * 1e3)   # mV

    return {'reference_potential_map': v,
            'reference_potential_xc': xc,
            'reference_potential_yc': yc,
            }

def _get_electrodes_within_nerve(nerve_radius=1.5):
    """Returns a list of electrodes that are positioned within the nerve.
    THIS RADIUS IS CURRENTLY HARD-CODED TO MATCH THE COMSOL FEM DATA.
    """
    electrode_list = []
    for i in comsol_utils.USEA_ELEC_COORDS:
        # If electrode is not contained within the nerve, do not plot.
        if np.linalg.norm([*comsol_utils.USEA_ELEC_COORDS[i]]) > nerve_radius:
            continue
        electrode_list.append(i)
    return electrode_list


def _load_fem_for_usea(cx_mm=1.8, cy_mm=1.4, data_to_mm=1000,
                       nerve_radius=1.5):
    """Loads the FEM for the uSEA

    Returns a dictionary with keys reference_potential_map,
    reference_potential_xc, reference_potential_yc.
    """
    electrode_list = _get_electrodes_within_nerve(
        nerve_radius=nerve_radius)
    (x, y, v) = comsol_utils.load_usea_data(electrode_list[0],
                                            cx_mm,
                                            cy_mm,
                                            data_to_mm)
    # keep only the values within the nerve diameter (=3mm) THIS IS
    # CURRENTLY HARD-CODED TO MATCH THE COMSOL FEM DATA
    valid_indices = np.sqrt(x ** 2 + y ** 2) < nerve_radius
    xc = x[valid_indices] # Scaled to mm
    yc = y[valid_indices]  # Scaled to mm
    v = [v[valid_indices]]  # mV

    for electrode_idx in electrode_list[1:]:
        # load the ref potential map for the electrode index
        fem_data = comsol_utils.load_usea_data(electrode_idx,
                                               cx_mm,
                                               cx_mm,
                                               data_to_mm)
        v.append(fem_data[2][valid_indices])   # mV

    return {'reference_potential_map': v,
            'reference_potential_xc': xc,
            'reference_potential_yc': yc,
            }


ROUND_CUFF = NeuralStimulatorConfig(
    electrodes=[*range(14)],
    frequency_min=0,
    frequency_max=500,
    amplitude_min=0,
    amplitude_max=5,  # TODO: check what this should be. Guess.
    pulse_width_min=0,
    pulse_width_max=0.300,  # ms
    update_rate=33.0,  # Hz
    **_load_fem_for_round_cuff(),
)

# USEA = NeuralStimulatorConfig(
#     electrodes=_get_electrodes_within_nerve(),
#     frequency_min=0,
#     frequency_max=500,
#     amplitude_min=0,
#     amplitude_max=5,  # TODO: check what this should be. Guess.
#     pulse_width_min=0,
#     pulse_width_max=0.300,  # ms
#     update_rate=33.0,  # Hz
#     **_load_fem_for_usea(),
# )


class NeuralStimulator(object):
    """Class to hold the hardware specifications and the reference potentials.
    """
    def __init__(self, stimulator_config: NeuralStimulatorConfig):
        self.cfg = stimulator_config
        self.stimulus = None
        self.pulse_width = 0.1
        self.nerve = Nerve()
        self.estim = ElectricalActivation()

    def generate_stimulus_oneshot(self, electrode_idx, stim_amplitude,
                                  frequency, pulse_width, stimulus_duration):
        """

        :param electrode_idx:
        :param stim_amplitude:
        :param frequency:
        :param pulse_width:
        :param stimulus_duration:
        :return:
        """

        self.stimulus = StimulationOneShot(
            electrode=electrode_idx,
            stim_amplitude=stim_amplitude,
            pulse_width=pulse_width,
            frequency=frequency,
            duration=stimulus_duration
        )

    def get_hardware_params(self):
        """Returns f_min, f_max, a_min, a_max."""
        return {'f_min': self.cfg.frequency_min,
                'f_max': self.cfg.frequency_max,
                'a_min': self.cfg.amplitude_min,
                'a_max': self.cfg.amplitude_max
                }

    def apply_multi_stimulus(self, nerve: nerves.Nerve, neuron_idx,
                             stimulations: List[StimulationSequence]):
        """

        :param nerve: nerves.Nerve instance.
        :param neuron_idx: neuron indexes
        :param stimulations: List of stimulations that define electrode,
            amplitude, frequency, pulse width
        :return:
        """
        # make sure all the amplitudes have the same length.
        sequence_length_all = [len(s.amplitudes) for s in stimulations]
        if not len(np.unique(sequence_length_all)) == 1:
            raise ValueError(
                "All stimulation sequences must be the same length."
            )

        stim_sequence_length = sequence_length_all[0]
        t_original = np.arange(0, stim_sequence_length) * nerve.dt
        f_original = 1000.0 / nerve.dt
        dt_step = 1000.0 / self.cfg.update_rate
        n_steps = int(stim_sequence_length*nerve.dt/dt_step)
        t_step = np.arange(0, n_steps)*dt_step

        amplitudes = [
            apply_butter_lpf_sym(s.amplitudes, self.cfg.update_rate, f_original)
            for s in stimulations
        ]

        frequencies = [
            apply_butter_lpf_sym(s.frequencies, self.cfg.update_rate, f_original)
            for s in stimulations
        ]

        electrodes = [s.electrode for s in stimulations]
        for i, t in enumerate(t_step):
            i0 = np.searchsorted(t_original, t)
            if i == len(t_step)-1:
                i1 = stim_sequence_length
            else:
                i1 = np.searchsorted(t_original, t_step[i+1])
            N = i1 - i0
            ampls = [a[i0] for a in amplitudes]
            freqs = [f[i0] for f in frequencies]
            if i0 != nerve.neurons[0].time_counter:
                print("Time counter mismatch! {} != {}".
                      format(i0, nerve.neurons[0].time_counter))
            pmap = self.get_activation_probability_map_multi(
                ampls, electrodes, neuron_idx, self.pulse_width
            )
            nerve.step(p_spike_applied_field=pmap, n_steps=N)

    def apply_continuous_stimulus(self,
                                  nerve: nerves.Nerve,
                                  neuron_idx,
                                  stimulation: StimulationSequence):
        """
        Apply continuous amplitude/frequency to neurons at the neuron time step.

        :param nerve: nerves.Nerve instance.
        :param neuron_idx: neuron indexes
        :param stimulation: StimulationSequence defining which electrode and
            the stimulation amplitude/frequency/pulse width
        """
        # The stimulus can only be updated at the update rate, so the amplitudes
        # and frequencies must be resampled to match the update rate. This is
        # done by applying a low pass filter and resampling to the update rate.
        t_original = np.arange(0, len(stimulation.amplitudes))*nerve.dt
        f_original = 1000.0/nerve.dt
        dt_new = 1000.0/self.cfg.update_rate
        Nsteps = int(len(stimulation.amplitudes)*nerve.dt/dt_new)
        t_new = np.arange(0, Nsteps)*dt_new

        amplitudes = apply_butter_lpf_sym(
            stimulation.amplitudes, self.cfg.update_rate, f_original)
        amplitudes = np.interp(t_new, t_original, amplitudes)
        frequencies = apply_butter_lpf_sym(
            stimulation.frequencies, self.cfg.update_rate, f_original)
        frequencies = np.interp(t_new, t_original, frequencies)

        for i, (a, f) in enumerate(zip(amplitudes, frequencies)):
            pmap = self.get_activation_probability_map(
                a, stimulation.electrode, neuron_idx, self.pulse_width
            )
            iout0 = np.searchsorted(t_original, t_new[i])
            if i == len(amplitudes)-1:
                iout1 = len(amplitudes)
            else:
                iout1 = np.searchsorted(t_original, t_new[i] + dt_new)
            N = iout1 - iout0
            nerve.step(p_spike_applied_field=pmap, n_steps=N)

    def get_activation_probability_map_multi(self,
                                             amplitudes,
                                             electrodes,
                                             neuron_idx,
                                             pulse_width):
        """

        :param amplitudes:
        :param electrodes:
        :param neuron_idx:
        :param pulse_width:
        :return:
        """
        potential_maps = np.vstack([
            self.cfg.reference_potential_map[e]*a
            for e, a in zip(electrodes, amplitudes)
        ])
        pmap_all = np.sum(potential_maps, axis=0)
        potential_per_neuron = np.array([pmap_all[i] for i in neuron_idx])
        d_threshold = activation.get_diameter_from_potential(
            potential_per_neuron,
            pulse_width)
        return activation.get_probability_map(d_threshold)

    def get_activation_probability_map(self, amplitude, electrode_idx, neuron_idx,
                                       pulse_width):
        potential_map = self.cfg.reference_potential_map[electrode_idx] * amplitude
        potential_per_neuron = np.array([potential_map[i] for i in neuron_idx])
        d_threshold = self.estim.get_diameter_from_potential(
            potential_per_neuron,
            pulse_width)
        return self.nerve.get_activation_probability(d_threshold)

    def apply_oneshot_stimulus(self,
                               a_nerve: nerves.Nerve,
                               neuron_idx,
                               timesteps):
        """Applies a stimulus to an instance of the Nerve class."""
        probability_map = self.generate_activation_probabilities(neuron_idx)
        for t in range(timesteps):
            if divmod(t,
                      int(1/self.stimulus.frequency*1000/a_nerve.dt))[1] == 0 \
                    and t < self.stimulus.duration/a_nerve.dt:
                a_nerve.step(p_spike_applied_field=probability_map)
            else:
                a_nerve.step(p_spike_applied_field=np.array([0]))
        return a_nerve

    def generate_activation_probabilities(self, neuron_idx):
        """Returns the probabilities for activation by the defined stimulus at
        the neuron locations given by neuron_idx
        """
        potential_map = self.cfg.reference_potential_map[self.stimulus.electrode] \
            * self.stimulus.stim_amplitude
        potential_per_neuron = np.array([potential_map[i] for i in neuron_idx])
        d_threshold = activation.get_diameter_from_potential(
            potential_per_neuron,
            self.stimulus.pulse_width)
        if PLOTTING:
            plt.scatter(self.cfg.reference_potential_xc[neuron_idx],
                        self.cfg.reference_potential_yc[neuron_idx], 10,
                        d_threshold)
            plt.colorbar()
            plt.title('Diameter at threshold for {} mA stimulus'.
                      format(self.stimulus.stim_amplitude))
            plt.show()
        return activation.get_probability_map(d_threshold)

    def plot_stimulus_potential_map(self, neuron_idx):
        """Plotting function"""
        # Show the map of potential values and where the neurons are located
        plt.scatter(self.cfg.reference_potential_xc,
                    self.cfg.reference_potential_yc, 10,
                    self.cfg.reference_potential_map[self.stimulus.electrode])
        plt.colorbar()
        for n, neuron in enumerate(neuron_idx):
            marker_str = '${}$'.format(n)
            plt.scatter(self.cfg.reference_potential_xc[neuron],
                        self.cfg.reference_potential_yc[neuron],
                        marker=marker_str)
        plt.show()

    def choose_neuron_coordinates_randomly(self, num_neurons, electrode=None):
        """Choose at random coordinates in the map that represent neurons"""
        if electrode is None:
            electrode = self.stimulus.electrode
        neuron_idx = random.sample(
            range(len(self.cfg.reference_potential_map[electrode])),
            num_neurons)
        neuron_idx.sort()
        return neuron_idx


if __name__ == "__main__":

    # Create nerve containing num_neurons each with a spontaneous firing rate sp
    n_neurons = 10
    sp = 1  # Probability of spontaneous firing (Hz)
    my_nerve = nerves.Nerve(num_neurons=n_neurons, spontaneous_rate=sp)

    # Create a stimulator object
    stimulator = NeuralStimulator(stimulator_config=ROUND_CUFF)
    # Generate the stimulus to apply to the nerve
    stimulator.generate_stimulus_oneshot(electrode_idx=0, stim_amplitude=0.2,
                                         frequency=200, pulse_width=0.1,
                                         stimulus_duration=200)

    # Generate neuron coordinates (associated with position within nerve, for
    # spatially varying application of stimulus
    neurons = stimulator.choose_neuron_coordinates_randomly(n_neurons)
    stimulator.plot_stimulus_potential_map(neurons)

    # Apply the stimulus
    ts = 5000  # timesteps are 0.5 ms
    dt = my_nerve.dt
    my_nerve = stimulator.apply_oneshot_stimulus(my_nerve, neurons, ts)
    # Apply the same stimulus but record a different duration, to append
    # multiple different stimuli
    my_nerve = stimulator.apply_oneshot_stimulus(my_nerve, neurons, 3000)
    # Get resulting spike trains
    x = my_nerve.spike_trains

    # Generate a raster plot
    fig, ax = plt.subplots()
    # create a horizontal plot
    ax.eventplot([[x_*dt for x_ in x[i][1]] for i in range(len(x))],
                 colors='black',
                 lineoffsets=1,
                 linelengths=1)
    plt.xlabel('time (ms)')
    plt.ylabel('nerve fiber')
    plt.title('spike rasters')
    plt.show()
