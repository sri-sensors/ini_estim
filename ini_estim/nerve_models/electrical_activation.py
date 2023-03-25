import numpy as np
import scipy.interpolate as interp


class ElectricalActivation:
    """Class containing functions that estimate the probability of activation
    of a neuron given an extracellular applied electric field.

    Reference values for electrical activation thresholds are from McNeal 1976
    IEEE Trans. Biomed. Eng. BME-23:(4) 329-37.
    """

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

    def __init__(self):
        self.cm_to_m = 0.1
        self.thresh_fun = interp.interp1d(
            self.FIBER_DIAMETERS,
            self.THRESHOLD_CURRENT_D,
            kind='quadratic',
            fill_value='extrapolate'
        )
        self.diam_fun = interp.interp1d(
            self.THRESHOLD_CURRENT_D,
            self.FIBER_DIAMETERS,
            kind='linear',
            fill_value='extrapolate'
        )

    def get_threshold(self, diameter, pulse_duration):
        """ Calculates the threshold current of activation for a given neuron
        diameter and pulse duration.

        Parameters
        ----------
        diameter
            Fiber diameter in microns
        pulse_duration
            Pulse duration in ms

        Returns
        -------
        Threshold current (mA)
        """
        # Compute threshold value for fiber diameter d for a 100us pulse
        i_d = self.thresh_fun(diameter)
        # Scale the threshold values to match the value for diameter d
        i_thresh_scaled = np.array(self.THRESHOLD_CURRENT_PD)*i_d/0.226
        pulse_fun = interp.interp1d(
            self.PULSE_DURATIONS, i_thresh_scaled, kind='quadratic',
            fill_value='extrapolate'
        )
        return pulse_fun(pulse_duration)

    def get_diameter_from_potential(self, v_stim, pulse_dur_ms=0.1, rho=300.0):
        """ Returns the nerve fiber diameter that would be activated at a given
        potential.

        Parameters
        ----------
        v_stim
            Stimulation potential in Volts

        pulse_dur_ms
            Pulse duration in ms

        rho
            resistivity kOhm cm

        Returns
        -------
        Nerve fiber diameter
        """
        a_to_ma = 1e3    # A to mA conversion
        i = v_stim * 4 * np.pi * self.cm_to_m / rho * a_to_ma
        i_max = self.get_threshold(self.MIN_FIBER_D, pulse_dur_ms)
        i_min = self.get_threshold(self.MAX_FIBER_D, pulse_dur_ms)
        i = np.clip(i, i_min, i_max)
        out = self.diam_fun(i)
        out[out < 0] = 0.0
        return out

    @staticmethod
    def get_potential(rho, current, distance):
        """Calculates the applied extracellular potential field (mV) resulting
        from a point source

        :param current: applied current (uA)
        :param rho: extracellular resisitvity of the medium (kOhm cm)
        :param distance: distance from any node to the point source (cm)
        """
        r = (rho*0.25/np.pi)/distance
        return current*r
