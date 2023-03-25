"""Classes and functions that model the firing of a neuron which spikes with
spontaneous and evoked probabilites, held within a simulator.py Nerve class"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate.interpolate as interpolate


class Beta:
    STATE_VECTOR_500US = np.array(
        [1, 0, 0, 0, 0, 0.05, 0.1, 0.15, 0.2, 0.35, 0.5]
    )

    def __init__(self, dt=0.5):
        """Initializes activability state helper class Beta. Stores the state
        of the neuron activability, eg, whether the neuron is in its absolute
        refractory period or can be activated
        :param dt: timestep in ms
        """
        self.i = 0  # Index of the current activability state
        self._dt = dt
        self.state_vector = None
        self.init_states()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = val
        self.init_states()
        self.reset()

    def init_states(self):
        """Initializes activation state vector based on dt"""
        nstates = int(
            0.5/self._dt * len(Beta.STATE_VECTOR_500US) + 0.5
        )
        t_new = np.arange(nstates)*self._dt
        t_old = np.arange(len(Beta.STATE_VECTOR_500US))*0.5
        f = interpolate.interp1d(
            t_old, Beta.STATE_VECTOR_500US, kind='next',
            fill_value='extrapolate'
        )
        self.state_vector = f(t_new)

    def get_state(self):
        """Gets current state
        :return: beta, the activability
        """
        return self.state_vector[self.i]

    def step_state(self, fired):
        """
        Steps through the state diagram
        :param fired: Boolean True/False representing if a neuron fired or not
        :return: beta, the activability
        """
        if fired:
            self.i = 1
        else:
            if self.i == 0:
                pass
            elif self.i == self.state_vector.__len__() - 1:
                self.i = 0
            else:
                self.i += 1
        return self.state_vector[self.i]

    def reset(self):
        self.i = 0


class Nerve:
    def __init__(self, num_neurons, spontaneous_rate, dt=0.5):
        """

        :param num_neurons: number of neurons
        :param spontaneous_rate: spontaneous activation rate
        :param dt: time step in milliseconds
        """
        self.neurons = [Neuron(spontaneous_rate, dt) for _ in range(num_neurons)]
        self._num_neurons = num_neurons

    def step(self, p_spike_applied_field, n_steps=1):
        """If p_spike_applied_field is a scalar, it is applied to all neurons.
        If it is an array, it must have length equal to the number of neurons.
        Usage for multiple steps is not yet defined
        """
        if len(p_spike_applied_field) == 1:
            p_spike_applied_field = np.repeat(p_spike_applied_field,
                                              self.num_neurons)
        else:
            assert len(p_spike_applied_field) == self.num_neurons
        for _ in range(n_steps):
            for n, neuron in enumerate(self.neurons):
                neuron.step(p_applied_field=p_spike_applied_field[n])

    @property
    def spike_trains(self):
        """Returns the spike trains for all neuron fibers in the nerve"""
        return [(i, self.neurons[i].spike_train)
                for i in range(len(self.neurons))]

    @property
    def num_neurons(self):
        return self._num_neurons

    @property
    def spike_probability(self):
        """Returns the time-varying probability of firing for all neuron fibers
        in the nerve"""
        return [(i, self.neurons[i].p) for i in range(len(self.neurons))]

    @property
    def dt(self):
        """Get time resolution for nerve model"""
        return self.neurons[0].dt

    @dt.setter
    def dt(self, val):
        for n in self.neurons:
            n.dt = val


class Neuron:
    def __init__(self, p, dt=0.5):
        """
        Initializes the neuron class
        :param p: (time-dependent) firing rate of the nerve fiber (Hz)
        :param dt: time step in seconds
        """
        self.fired = 0
        self._dt = dt  # timestep, ms
        self.p0 = p*self.dt*1e-3  # Spontaneous firing rate
        self.p = [p*self.dt*1e-3]  # Probability of firing at any given timestep

        self.beta = Beta(self.dt)
        self.spike_train = []
        self.time_counter = 0

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, val):
        self._dt = val
        self.beta.dt = val
        self.reset_history()

    def reset_history(self):
        self.fired = 0
        self.p = [self.p0]
        self.beta.reset()
        self.spike_train = []
        self.time_counter = 0

    def step(self, p_applied_field=0, random_draw=None):
        """Steps through state transition
        :param p_applied_field: probability of firing as a result of an applied
            electric field
        :param random_draw: outcome of a uniform random number draw [0, 1]
        :return fired: whether or not the neuron fired at current timestep"""
        beta = self.beta.get_state()
        p_firing = beta*min(self.p0 + p_applied_field, 1)
        self.p.append(p_firing)
        if random_draw:
            fired = random_draw <= p_firing
        else:
            fired = np.random.binomial(1, p_firing)
        self.beta.step_state(fired)
        if fired:
            self.spike_train.append(self.time_counter)
        self.time_counter += 1


if __name__ == "__main__":
    sp = 1  # Probability of spontaneous firing
    applied_prob = np.array([0])
    my_nerve = Nerve(num_neurons=20, spontaneous_rate=sp)

    timesteps = 1000
    for _ in range(timesteps):
        my_nerve.step(p_spike_applied_field=applied_prob)

    x = my_nerve.spike_trains
    fig, ax = plt.subplots()
    # create a horizontal plot
    ax.eventplot([x[i][1] for i in range(len(x))], colors='black',
                 lineoffsets=1,
                 linelengths=1)
    plt.xlabel('timesteps')
    plt.ylabel('nerve fiber')
    plt.title('spike rasters')
    plt.show()

    plt.plot(my_nerve.spike_probability[0][1])
    plt.xlabel('timesteps')
    plt.ylabel('probability')
    plt.show()
