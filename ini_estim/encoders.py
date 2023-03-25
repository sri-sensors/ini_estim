import numpy as np
import scipy.interpolate as interp


class SensorEncoder:
    """
    Class that takes a sensor signal and produces an output appropriate for
    an electrode cuff

    """
    def __init__(self,
                 num_sensors: int,
                 num_electrodes: int,
                 control_levels: np.ndarray = np.linspace(0, 1.2, 10),
                 amin: float = 0.0,
                 amax: float = 1.0,
                 astep: float = 0.01,
                 fmin: float = 33.0,
                 fmax: float = 330.0):
        """

        Parameters
        ----------
        num_sensors
            The number of sensors to input.
        num_electrodes
            The number of electrodes to stimulate
        control_levels
            The list of discrete amplitudes that can be addressed by the
            stimulation system. Defined the same for all electrodes
        amin
            The minimum allowable amplitude
        amax
            The maximum allowable amplitude
        astep
            The minimum discretization of amplitudes allowed
        control_levels

        fmin
            The minimum allowable frequency in Hz. fmin is also taken as the
            electrode update rate, i.e. how often the sensor can change its
            output configuration. Update rate can be changed manually by
            setting property "update_rate".
        fmax
            The maximum allowable frequency in Hz. fmax is also taken as the
            output sampling rate. The output sampling rate can be updated
             manually by setting property "sample_rate".
        """
        self.num_sensors = num_sensors
        self.num_electrodes = num_electrodes
        self.control_levels = None
        self.amin = amin
        self.amax = amax
        self.astep = astep
        self.update_rate = fmin
        self.sample_rate = fmax
        self.fmin = fmin
        self.fmax = fmax
        self._validate_control_levels(control_levels)

    @property
    def samples_per_update(self):
        return self._get_samples_per_update()

    def _get_samples_per_update(self):
        samples_per_update = self.sample_rate / self.update_rate
        if not samples_per_update.is_integer():
            raise ValueError(
                "The ratio of update_rate and sample_rate must be an integer."
            )
        return int(samples_per_update)

    def get_num_output_samples(self, num_input_samples, sample_rate):
        """
        Compute the number of output samples

        Parameters
        ----------
        num_input_samples
        sample_rate

        Returns
        -------
        Number of output samples
        """
        N = num_input_samples*self.sample_rate/sample_rate
        N = np.ceil(N/self.samples_per_update)*self.samples_per_update
        return int(N)

    def _validate_control_levels(self, control_levels):
        """Confirms that the control levels in list are allowed by hardware"""
        # Convert
        if not all(np.diff(control_levels) >= self.astep):
            raise ValueError("Specified control levels are smaller than allowed"
                             " discretization for this hardware.")
        # set of possible control levels:
        cl = np.arange(self.amin, self.amax, self.astep)
        cl = np.append(cl, self.amax)

        # Adjust values to match only possible control levels
        adjusted_levels = np.empty(len(control_levels))
        for i, l in enumerate(control_levels):
            adjusted_levels[i] = min(cl, key=lambda x:abs(x-l))

        # Set control levels
        self.control_levels = adjusted_levels

    def encode_series(self, sensor_values: np.ndarray, sample_rate: float,
                      scale: bool):

        """
        Parameters
        ----------
        sensor_values
            The sensor values are a numpy ndarray with dimensions
            (num_samples, num_sensors).
        sample_rate
            The sample rate of the sensor values.
        scale
            Set to True to autoscale the values to 0 to 1, otherwise, the
            sensor_values are assumed to be pre-scaled and will be clipped
            to 0 to 1.

        Returns
        -------
        The encoded values
            A numpy ndarray with dimensions (num_electrodes, N)
            N is determined by the configured electrode sampling rate and in
            general is approximately = num_samples/(sample_rate*self.dt)
        """
        raise NotImplementedError("Encode must be used from a child class!")

    def map_control_levels(self, sensor_values: np.ndarray):
        """Maps the distribution of sensor data (a histogram) onto the electrode
        array control levels defined in self.

        :parameter sensor_values: The sensor values are a numpy ndarray with
            dimensions (num_sensors, num_samples).
        """
        raise NotImplementedError("Must be used from a child class!")


class LinearEncoder(SensorEncoder):
    WEIGHT_TYPES = {"constant", "linear", "random"}
    INPUT_WEIGHT_TYPES = {"random", "pca"}

    def __init__(self,
                 num_sensors: int,
                 num_electrodes: int = None,
                 amin: float = 0.0,
                 amax: float = 1.0,
                 fmin: float = 33.0,
                 fmax: float = 330.0):
        """
        Initializes basic encoder parameters, and sets encoding weights to
        default value of initialize_weights method.

        Parameters
        ----------
        num_electrodes
        amin
        amax
        fmin
        fmax
            See SensorEncoder.
        """
        if num_electrodes is None:
            num_electrodes = num_sensors
        super().__init__(num_sensors=num_sensors,
                         num_electrodes=num_electrodes,
                         amin=amin,
                         amax=amax,
                         fmin=fmin,
                         fmax=fmax)
        self._input_weights = None
        self._input_weight_type = "random"
        self._weights = None
        self.weight_type = "constant"
        self._f_constant = None

    def set_constant_frequency(self, f_constant=None):
        """ Set to use a constant frequency

        Parameters
        ----------
        f_constant : float
            The constant frequency to use. Set to None to disable and use the
            frequency determined by the output weights.
        """
        if f_constant is None:
            self._f_constant = None
        else:
            self._f_constant = np.clip(f_constant, self.fmin, self.fmax)

    @property
    def input_weights(self):
        return self._input_weights

    @property
    def output_weights(self):
        return self._weights

    @property
    def all_weights(self):
        if self._input_weights is not None and self._weights is not None:
            return np.concatenate((self._input_weights.ravel(),
                                   self._weights.ravel()))
        else:
            return None

    def initialize_input_weights(
            self, weight_type=None, sensor_values=None
    ):
        """

        Parameters
        ----------
        weight_type : str
            Value can be "identity", "pca", or "random". If "identity", the
            number of sensors must be less than (or equal) to the number of
            electrods. If "pca", principal component analysis is performed
            on the sample sensor values and the first num_electrodes
            principal components are used as the input weights. If "random",
            the input weights are a random matrix mapping num_sensors to
            num_electrodes, the columns of which have norm=1.
        sensor_values : np.ndarray
            Sample of sensor values. sensor_values is only required if using
            "pca" weight type. Dimension is assumed to be
            (num_samples x num_sensors). If omitted, weight_type will change
            to "random" if "pca" was chosen.

        Notes
        -----
        If weight_type is omitted, then the most appropriate weight type will
        be chosen automatically according to:
            num_sensors <= num_electrodes: identity
            sensor_values are supplied: pca
            otherwise: random

        """
        if (weight_type is None or
                weight_type.lower() not in ["identity", "pca", "random"]):
            if self.num_sensors <= self.num_electrodes:
                weight_type = "identity"
            elif sensor_values is not None:
                weight_type = "pca"
            else:
                weight_type = "random"

        if (weight_type.lower() == "identity" and
                self.num_sensors <= self.num_electrodes):
            self._input_weight_type = "identity"
            self._input_weights = np.zeros((self.num_sensors + 1,
                                            self.num_electrodes))
            for i in range(self.num_sensors):
                self._input_weights[i, i] = 1.0
        elif weight_type.lower() == "pca" and sensor_values is not None:
            self._input_weight_type = "pca"
            mu = np.mean(sensor_values, axis=0)[None, :]
            svm = (sensor_values - mu)
            C = svm.transpose() @ svm / svm.shape[0]
            S, V = np.linalg.eig(C)
            w = V[:, 0:self.num_electrodes]
            mu_mapped = mu @ w
            self._input_weights = np.concatenate((w, -mu_mapped), 0)
        else:
            self._input_weight_type = "random"
            # extra dimension is a bias term
            r = self.num_sensors + 1
            c = self.num_electrodes
            self._input_weights = np.random.rand(r, c) - 0.5
            self._input_weights /= np.linalg.norm(
                self._input_weights, axis=0
            )[None, :]

    def initialize_weights(self, sensor_sample_rate, weight_type=None):
        """
        Initialize the sensor weights for each update

        Parameters
        ----------
        sensor_sample_rate
        weight_type

        Returns
        -------

        """
        samples_per_update = int(sensor_sample_rate/self.update_rate)
        if weight_type is None:
            weight_type = self.weight_type

        if weight_type == "constant":
            self._weights = np.zeros((2, samples_per_update))
            self._weights[:, :] = 1/samples_per_update
        elif weight_type == "linear":
            X = np.column_stack((np.ones(samples_per_update),
                                 np.linspace(-0.5, 0.5, samples_per_update)))
            self._weights = np.linalg.pinv(X)
        elif weight_type == "random":
            self._weights = np.random.rand(2, samples_per_update)-0.5
        else:
            raise ValueError(
                "weight_type must be in {}".format(self.WEIGHT_TYPES)
            )

        # normalize the weights - using L1 norm preserves averaging weights
        self._weights /= np.sum(np.abs(self._weights), 1)[:, None]
        self.weight_type = weight_type

    @property
    def weights(self):
        return self._weights

    def encode_series(self,
                      sensor_values: np.ndarray,
                      sample_rate: float,
                      scale: bool = True):
        """

        Parameters
        ----------
        sensor_values
            The sensor values are a numpy ndarray with dimensions
            (num_sensors, num_samples).
        sample_rate
            The sample rate of the sensor values.
        scale
            Set to True to autoscale the values to 0 to 1, otherwise, the
            sensor_values are assumed to be pre-scaled and will be clipped
            to 0 to 1.

        Returns
        -------
        The encoded values
            A numpy ndarray with dimensions (num_electrodes, N)
            N is determined by the configured electrode sampling rate and in
            general is approximately = num_samples*self.update_rate/sample_rate
        """
        if sensor_values.ndim == 1:
            sensor_values = sensor_values[:, np.newaxis]

        if scale:
            smin = np.min(sensor_values, 0)[np.newaxis, :]
            smax = np.max(sensor_values, 0)[np.newaxis, :]
            sensor_values = (sensor_values - smin)/(smax - smin)

        # - the below check is no longer necessary
        # num_sensors = sensor_values.shape[1]
        # if num_sensors != self.num_electrodes:
        #     raise ValueError("Number of sensors and electrodes must match.")

        sensor_samples_per_update = int(sample_rate/self.update_rate)
        if (self._weights is None or
                self._weights.shape[1] != sensor_samples_per_update):
            self.initialize_weights(sample_rate)
        if self._input_weights is None:
            self.initialize_input_weights(sensor_values=sensor_values)

        num_samples = sensor_values.shape[0]
        sensor_values = np.concatenate(
            (sensor_values, np.ones((num_samples, 1))),
            axis=1
        )
        # map to number of electrodes with input weights.
        sensor_values = sensor_values @ self._input_weights

        Tsensor = num_samples/sample_rate
        t_sensor = np.arange(0, num_samples)/sample_rate
        dt_sensor = 1/sample_rate
        dt_update = 1/self.update_rate
        Nupdates = int(np.ceil(Tsensor*self.update_rate))
        samples_per_update = self._get_samples_per_update()

        t_sensor_tmp = np.arange(0, sensor_samples_per_update)*dt_sensor
        f_sensor = interp.interp1d(
            t_sensor, sensor_values, 'nearest', axis=0,
            fill_value='extrapolate', assume_sorted=True
        )
        param_scale = np.array([[self.amax - self.amin], [self.fmax - self.fmin]])
        param_offs = np.array([[self.amin], [self.fmin]])

        out = np.zeros((samples_per_update*Nupdates, self.num_electrodes))
        if self._f_constant is not None:
            de = np.empty(self.num_electrodes)
            de[:] = self.sample_rate / self._f_constant

        for i in np.arange(Nupdates):
            offset = i*samples_per_update
            s_values = f_sensor(t_sensor_tmp)
            params = self._weights @ s_values
            params = np.clip(params, 0, 1)*param_scale + param_offs
            if self._f_constant is None:
                de = self.sample_rate/params[1, :]

            for e in np.arange(self.num_electrodes):
                inds = np.arange(0, samples_per_update, de[e]).astype(np.int)
                out[inds+offset, e] = params[0, e]
            t_sensor_tmp += dt_update

        return out

    def map_control_levels2(self, sensor_values: np.ndarray):
        """See method description in base class."""
        # assumes a linear distribution of control levels.
        if any(np.diff(self.control_levels, 2) > 0.1):
            raise ValueError("map_control_levels is not defined for nonlinear "
                             "control level choice")

        sensor_distribution = np.histogram(sensor_values,
                                           bins=len(self.control_levels),
                                           density=True)

        return sensor_distribution[0] * np.diff(sensor_distribution[1])

    def map_control_levels(self, prob_dist, bin_centers, scale: bool = False):
        """See method description in base class."""
        out_dist = np.zeros(len(self.control_levels))
        for prob, center in zip(prob_dist, bin_centers):
            if scale:
                center = (center - bin_centers[0])/\
                         (bin_centers[-1] - bin_centers[0])
            out_ix = np.nonzero(self.control_levels == self.encode_a(center))
            out_dist[out_ix] += prob
        return out_dist

    def encode_a(self, c):
        """Implements traditional linear encoding on a single sample.
        (Amplitude only)

        Args:
            c: sensor input at time t. (scaled between 0 and 1)
        """
        c = np.clip(c, 0.0, 1.0)
        a = c * (self.amax - self.amin) + self.amin

        # Match to closest value of allowed control values
        distances = abs(self.control_levels - a)
        return self.control_levels[np.nonzero(distances == min(distances))]


def map_control_levels(control_levels, prob_dist, bin_centers, encoding_fn):
    out_dist = np.zeros(len(control_levels))
    for prob, center in zip(prob_dist, bin_centers):
        out_ix = np.where(control_levels == encoding_fn(center))
        out_dist[out_ix] += prob
    return out_dist


def linear_encode(c, a_min, a_max):
    """Implements traditional linear encoding. (Amplitude only)

    Args:
        c: sensor input at time t. (scaled between 0 and 1)
        f_min: minimum frequency allowed by hardware (Hz)
        f_max: maximum frequency allowed by hardware (Hz)
        a_min: minimum amplitude allowed by hardware (mA)
        a_max: maximum amplitude allowed by hardware (mA)
    """
    c = np.clip(c, 0.0, 1.0)
    #f = c*(f_max - f_min) + f_min
    a = c*(a_max - a_min) + a_min
    return a


def get_data_distribution(sensor_values, n_bins: int = 100):
    """Returns the probability density function and bin center values for a
    vector of data"""
    n, bins = np.histogram(sensor_values, n_bins, density=True)
    return n*np.diff(bins), bins[0:-1] + np.diff(bins)/2


class Encoder(object):
    """Class to represent the encoding of sensor signals to electrode
    activation patterns
    """
    def __init__(self, kind, stimulator_params=None):
        """Initialize Encoder class"""
        self._kind = kind
        self._stimulator_parameters = stimulator_params
        self.electrode = 'cuff'

    def encode_sample(self, sensor_input):
        """

        :param sensor_input:
        :return: frequency, amplitude
        """
        if self._kind == 'linear':
            return self._linear_encode(c=sensor_input,
                                       **self._stimulator_parameters)
        elif self._kind == 'scaled_linear':
            return self._scaled_linear_encode(c=sensor_input,
                                              **self._stimulator_parameters)
        elif self._kind == 'biomimetic1':
            return self._biomimetic1_encode()
        else:
            raise NotImplementedError

    def set_encoder_kind(self, kind, parameters):
        self._kind = kind
        self._stimulator_parameters = parameters

    @staticmethod
    def _linear_encode(c, f_min, f_max, a_min, a_max):
        """Implements traditional linear encoding.

        Args:
            c: sensor input at time t. (scaled between 0 and 1)
            f_min: minimum frequency allowed by hardware (Hz)
            f_max: maximum frequency allowed by hardware (Hz)
            a_min: minimum amplitude allowed by hardware (mA)
            a_max: maximum amplitude allowed by hardware (mA)
        """
        c = np.clip(c, 0.0, 1.0)
        f = c*(f_max - f_min) + f_min
        a = c*(a_max - a_min) + a_min
        return f, a

    @staticmethod
    def _scaled_linear_encode(c, f_min, f_max, a_min, a_max):
        """Implements scaled traditional linear encoding.

        Args:
            c: sensor input at time t. (scaled between 0 and 1)
            f_min: minimum frequency allowed by hardware (Hz)
            f_max: maximum frequency allowed by hardware (Hz)
            a_min: minimum amplitude allowed by hardware (mA)
            a_max: maximum amplitude allowed by hardware (mA)
        """
        c = np.clip(2*c, 0.0, 1.0)
        f = c*(f_max - f_min) + f_min
        a = c*(a_max - a_min) + a_min
        return f, a

    @staticmethod
    def _biomimetic1_encode(self, c,  f_min, f_max, a_min, a_max):
        """Implements Biomimetic 1 encoding
        Args:
            c: sensor input vector for 2 time points [t_-1, t_0]
            f_min: minimum frequency allowed by hardware (Hz)
            f_max: maximum frequency allowed by hardware (Hz)
            a_min: minimum amplitude allowed by hardware (mA)
            a_max: maximum amplitude allowed by hardware (mA)
        """
        if len(c) != 2:
            raise ValueError("Sensor input must be a vector of length 2")
        c = np.clip(c, 0.0, 1.0)
        vt = c[1] - c[0]
        if vt < 0:
            f = c[1]*(f_max - f_min) + f_min
            a = c[1] * (a_max - a_min) + a_min
        else:
            f = (5*vt + c[1])*(f_max - f_min) + f_min
            a = (5*vt + c[1]) * (a_max - a_min) + a_min
        return f, a

    @staticmethod
    def _biomimetic2_encode(self, c, f_min, f_max, a_min, a_max):
        """Implements Biomimetic 2 encoding
        Args:
            c: sensor input vector for 2 time points [t_-3, t_-2, t_-1, t_0]
            f_min: minimum frequency allowed by hardware (Hz)
            f_max: maximum frequency allowed by hardware (Hz)
            a_min: minimum amplitude allowed by hardware (mA)
            a_max: maximum amplitude allowed by hardware (mA)
        """
        if len(c) != 4:
            raise ValueError("Sensor input must be a vector of length 4")
        c = np.clip(c, 0.0, 1.0)
        v = np.diff[c]
        a = np.diff[v]

        f = 186*c[3] - 185*c[2] + 1559*v[2] - 360*v[1] - 109*v[0] + 364*a[-1] \
            +170*a[-2]
        a = a_min

        return f, a

if __name__ == "__main__":
    encoder = LinearEncoder(num_electrodes=1, amax=1.2)

    mu, sigma = 3., 1.  # mean and standard deviation
    s = np.random.lognormal(mu, sigma, 1000)
    pdf, bin_centers = get_data_distribution(s)
    stim_probs = encoder.map_control_levels(pdf, bin_centers)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 1)
    ax[0].stem(encoder.control_levels, stim_probs, use_line_collection=True)
    ax[1].bar(bin_centers, pdf)
    plt.show()



