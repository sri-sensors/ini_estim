import numpy as np
import scipy.interpolate as interp
import scipy.stats as stats

from ini_estim.nerve_models.electrical_activation import ElectricalActivation


class Nerve:
    """Class that holds the Nerve (defined as a collection of individual
    neurons or nerve fibers) properties, such as resistivity and the range of
    neuron diameters within the nerve. Also models the spiking probability of
    the neurons given extraceullar stimulation and spontaneous spiking rates.
    """

    POST_SPIKE_FILTER = interp.interp1d(
        np.arange(0, 5.5, 0.5),
        [0, 0, 0, 0, 0.05, 0.1, 0.15, 0.2, 0.35, 0.5, 1.0],
        kind='linear', fill_value=(0.0, 1.0), bounds_error=False
    )

    def __init__(self):
        self.rho = 300.0  # resistivity kOhm cm
        self.diameter_mean = 11.0
        self.diameter_std_dev = 4.0
        self.spontaneous_rate_hz = 1.0
        self.e_stim = ElectricalActivation()

    def get_spikes(self, v_stim: np.ndarray, pulse_duration=0.1, dt=0.5):
        """
        Simulate the spiking response for a stimulus.

        Parameters
        ----------
        v_stim
            Stimulation potential in Volts. v_stim can be either 1-D for a
            single neuron, or 2-D for multiple neurons, in which case the second
            dimension is the number of neurons.
        pulse_duration
            Stimulation pulse duration in ms. Default is 0.1.
        dt
            Time step in ms. Default is 0.5.

        Returns
        -------
        An array of spikes with the same shape as v_stim.

        Notes
        -----
        The stimulation potential at a given instance represents a pulse, so if
        dt is changed, the effective number of pulses is also changed. That
        means, for example, if you decrease dt by half, then you must
        accordingly zero out every other v_stim sample to achieve the similar
        results.
        """
        if pulse_duration > dt:
            raise ValueError("Pulse duration cannot exceed time step.")

        if v_stim.ndim > 1:
            num_neurons = v_stim.shape[1]
            num_steps = v_stim.shape[0]
        else:
            num_neurons = 1
            num_steps = len(v_stim)

        spikes = np.zeros((num_steps, num_neurons), dtype=bool)
        last_spike = np.zeros(num_neurons) + 1000.0

        # do the simulation!
        p_stim = np.reshape(
            self.get_activation_probability(v_stim, pulse_duration),
            (num_steps, num_neurons)
        )
        srate_khz = 1e-3 * self.spontaneous_rate_hz
        # https://en.wikipedia.org/wiki/Poisson_distribution#Probability_of_events_for_a_Poisson_distribution
        p_rand = srate_khz * dt * np.exp(-srate_khz * dt)
        for i in range(num_steps):
            # firing probability from stimulus + random firing
            p_i = (p_stim[i, :] * (1 - p_rand) + p_rand) * \
                  self.POST_SPIKE_FILTER(last_spike)
            p_i = np.clip(p_i, 0.0, 1.0)

            # generate spikes
            fired = np.random.rand(num_neurons) <= p_i
            last_spike[fired] = 0.0
            last_spike[~fired] += dt
            spikes[i, :] = fired

        # make sure the dimensions match the stimulus
        spikes = np.reshape(spikes, v_stim.shape)
        return spikes

    def get_activation_probability(self, v_stim, pulse_dur_ms=0.1):
        """ Gets the probability of activation given a distribution of axon
        diameters in a nerve.

        Parameters
        ----------
        v_stim
            Stimulation potential in Volts
        pulse_dur_ms
            Pulse duration in ms

        Returns
        -------

        """
        d = self.e_stim.get_diameter_from_potential(v_stim, pulse_dur_ms,
                                                    self.rho)
        pmap = stats.norm(loc=self.diameter_mean,
                          scale=self.diameter_std_dev).cdf(d)
        return 1 - pmap
