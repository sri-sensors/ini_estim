import attr
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate

import ini_estim.mutual_information.theory
from ini_estim import comsol_utils, mutual_information
from ini_estim.nerve_models import electrical_activation as activation
from ini_estim.nerve_models import nerve
from ini_estim.stimulus import ROUND_CUFF

DEBUG = True


@attr.s(auto_attribs=True)
class StimulationParameters(object):
    electrodes: list
    stim_amplitude: np.ndarray
    pulse_width: list


@attr.s(auto_attribs=True)
class StimulationDefinition(object):
    electrodes: list
    stim_amplitude: np.ndarray
    pulse_width: list
    probabilities: np.ndarray


DEFAULT_PARAMS = StimulationParameters(
    electrodes=[*range(14)],
    stim_amplitude=np.asarray([0, 1]),
    pulse_width=[0.1]
    # probabilities=1/2 * np.ones((14, 2))
)


def _compute_activation_probability(electrode_idx: int,
                                    stim_amplitude: float,
                                    pulse_width: float,
                                    neuron_i: list):
    """Computes the activation probability."""
    potential_map = ROUND_CUFF.reference_potential_map[electrode_idx] \
        * stim_amplitude
    potential_per_neuron = np.array([potential_map[i] for i in neuron_i])

    activation_object = activation.ElectricalActivation()
    nerve_object = nerve.Nerve()
    diameter_th = activation_object.get_diameter_from_potential(
        potential_per_neuron,
        pulse_width
    )
    probability_map = nerve_object.get_activation_probability(diameter_th)
    return probability_map


def _compute_activation_probability_by_interpolation(axon_positions,
                                                     stim_amplitude,
                                                     pulse_width,
                                                     electrode_index):
    """Computes the activation probability.
    :params
        axon_positions:
        stim_amplitude: stimulus amplitude in mA
        pulse_width: pulse width in ms
        electrode_index:

    """
    raise NotImplementedError
    # # Rotate the reference map
    # x = np.reshape(NERVE_CUFF.reference_potential_map[1], 6400)*1e3
    # y = np.reshape(NERVE_CUFF.reference_potential_map[2], 6400)*1e3
    # theta = electrode_index * 2*math.pi/NERVE_CUFF.n_electrodes
    # x_rot, y_rot = comsol_utils.rotate_coordinates(x, y, theta)
    #
    # # Scale according to pulse amplitude and width
    # v_ref = NERVE_CUFF.reference_potential_map[0]*stim_amplitude*1e3
    # diameter_th = activation.get_diameter_from_potential(v_ref, pulse_width)
    # probability_map = activation.get_probability_map(diameter_th)
    #
    # # Interpolate probabilities AINT WORKING
    # # interpolating_fn = interpolate.interp2d(x_rot, y_rot, probability_map)
    # interpolating_fn = interpolate.interp2d(x.flatten(), y.flatten(),
    #                                         probability_map.flatten(),
    #                                         kind="cubic")
    # pm_new = interpolating_fn(
    #     axon_positions[:, 0], axon_positions[:, 1]
    # ).diagonal()
    # plt.scatter(axon_positions[:, 0], axon_positions[:, 1], 10, pm_new)
    # plt.show()
    # return


def compute_transfer_matrix(electrodes, stim_amplitude, pulse_width,
                            neuron_i):
    """Computes the transfer matrix for a stimulating electrode array and a
    nerve.

    :params
        electrodes:
        stim_amplitude:
        pulse_width
        neuron_i
    """
    n_axons = len(neuron_i)
    n_controls = len(electrodes) * \
        len(stim_amplitude) * \
        len(pulse_width)

    transition_matrix = np.zeros((n_axons, n_controls))

    m_counter = 0
    for idx, el in enumerate(electrodes):
        for stim_amp in stim_amplitude:
            for pw in pulse_width:
                transition_matrix[:, m_counter] = (
                    _compute_activation_probability(el,
                                                    stim_amp,
                                                    pw,
                                                    neuron_i))
                m_counter += 1

    return transition_matrix


def _compute_transfer_matrix(stimulation_params: StimulationDefinition,
                             neuron_i):
    """Computes the transfer matrix for a stimulating electrode array and a
    nerve.

    :params
        stimulation_params:
        neuron_idx:
    """
    n_axons = len(neuron_i)
    n_controls = len(stimulation_params.electrodes) * \
        len(stimulation_params.stim_amplitude) * \
        len(stimulation_params.pulse_width)

    transition_matrix = np.zeros((n_axons, n_controls))

    m_counter = 0
    for idx, el in enumerate(stimulation_params.electrodes):
        for stim_amplitude in stimulation_params.stim_amplitude:
            for pulse_width in stimulation_params.pulse_width:
                transition_matrix[:, m_counter] = (
                    _compute_activation_probability(el,
                                                    stim_amplitude,
                                                    pulse_width,
                                                    neuron_i))
                m_counter += 1

    return transition_matrix


def _compute_covariance_matrix(stimulation_params: StimulationDefinition):
    """Computes the covariance matrix using the defined probability distribution
    of stimulation inputs
    :params
    """
    n_amps = len(stimulation_params.stim_amplitude)
    n_pulse_widths = len(stimulation_params.pulse_width)
    n_controls = len(stimulation_params.electrodes) *\
        len(stimulation_params.stim_amplitude) * \
        len(stimulation_params.pulse_width)
    covariance_matrix = np.zeros((n_controls, n_controls))
    for m in range(n_controls):
        (electrode_idx_m, k_m) = divmod(m, n_amps*n_pulse_widths)
        for n in range(n_controls):
            (electrode_idx_n, k_n) = divmod(n, n_amps * n_pulse_widths)
            if electrode_idx_m == electrode_idx_n:
                if m == n:
                    covariance_matrix[m, n] = \
                        stimulation_params.probabilities[
                            electrode_idx_m, k_m] * \
                        (1-stimulation_params.probabilities[
                            electrode_idx_m, k_m])
                else:
                    covariance_matrix[m, n] = \
                        -stimulation_params.probabilities[electrode_idx_m, k_m]\
                        * stimulation_params.probabilities[electrode_idx_m, k_n]
    return covariance_matrix


def compute_covariance_matrix(probabilities, n_controls, n_electrodes):
    """computes the covariance matrix using the define probability distribution
    of stimulation inputs, does not use StimulationDefinition structure.
    """
    ctrls_per_elec = n_controls/n_electrodes
    if not int(ctrls_per_elec) == ctrls_per_elec:
        raise ValueError("n_controls must be divisible by n_electrodes")
    covariance_matrix = np.zeros((n_controls, n_controls))
    for m in range(n_controls):
        (electrode_idx_m, k_m) = divmod(m, ctrls_per_elec)
        idx_m = int(electrode_idx_m * ctrls_per_elec + k_m)
        for n in range(n_controls):
            (electrode_idx_n, k_n) = divmod(n, ctrls_per_elec)
            idx_n = int(electrode_idx_n * ctrls_per_elec + k_n)
            if electrode_idx_m == electrode_idx_n:
                if m == n:
                    covariance_matrix[m, n] = probabilities[idx_m] * \
                                              (1-probabilities[idx_n])
                else:
                    covariance_matrix[m, n] = -probabilities[idx_m] * \
                                              probabilities[idx_n]
    return covariance_matrix


def minfo(probabilities, transition_matrix, n_electrodes):
    """Computes the mutual information given the probability distribution of
    command values
    :param probabilities is a vector length (#electrodes * #control levels)
    :param transition_matrix is a matrix that describes the probability
    :param n_electrodes
    of activation for a given stimulation pattern
    """

    sigma_v = 0
    sigma_d = 0.15
    n_controls = transition_matrix.shape[1]
    covariance_matrix = compute_covariance_matrix(probabilities,
                                                  n_controls,
                                                  n_electrodes)
    return mutual_information.theory.linear_gaussian_mi(transition_matrix,
                                                        covariance_matrix,
                                                        sigma_v, sigma_d)


def _compute_transition_matrix_phase1_report(
        stimulation_params: StimulationParameters,
        neuron_i: list):
    """Computes the channel transition matrix of a nerve and a stimulator,
    using a uniform probability distribution and the StimulationParameters
    structure. Maintained here to generate the plots submitted for the Phase1
    final report.

    :params
        neuron_idx: a list of indices (max index is 17692) representing the axon
            positions
        stimulation_params:
    :"""
    # axon_positions = _place_axons(n_axons, nerve_diameter=nerve_diameter)
    n_axons = len(neuron_i)  # TODO: Interpolation.
    n_controls = len(stimulation_params.electrodes) *\
        len(stimulation_params.stim_amplitude) * \
        len(stimulation_params.pulse_width)
    controls_per_electrode = len(stimulation_params.stim_amplitude) * \
        len(stimulation_params.pulse_width)

    # For now, all control levels are equally probable
    p_control = 1/controls_per_electrode
    transition_matrix = np.zeros((n_controls, n_axons))
    covariance_matrix = np.zeros((n_controls, n_controls))
    m_counter = 0
    for idx, el in enumerate(stimulation_params.electrodes):
        for stim_amplitude in stimulation_params.stim_amplitude:
            for pulse_width in stimulation_params.pulse_width:
                transition_matrix[m_counter, :] = (
                    _compute_activation_probability(el,
                                                    stim_amplitude,
                                                    pulse_width,
                                                    neuron_i))
                m_counter += 1

    n_amps = len(stimulation_params.stim_amplitude)
    n_pulse_widths = len(stimulation_params.pulse_width)
    for m in range(n_controls):
        (electrode_idx_m, k_m) = divmod(m, n_amps*n_pulse_widths)
        for n in range(n_controls):
            (electrode_idx_n, k_n) = divmod(n, n_amps * n_pulse_widths)
            if electrode_idx_m == electrode_idx_n:
                if m == n:
                    covariance_matrix[m, n] = p_control*(1-p_control)
                else:
                    covariance_matrix[m, n] = -p_control*p_control

    return transition_matrix, covariance_matrix


def vary_max_stim_amp(max_stim_amps, neuron_pos):
    """
    :param max_stim_amps: list of maximum stimulation amplitude values to test
    :param neuron_pos: list of neuron indices
    :return: corresponding mi
    """
    for ni, n in enumerate(neuron_pos):
        mi = np.empty(len(max_stim_amps))

        # Center (almost) electrode only
        neuron_list = [n]

        for i, a in enumerate(max_stim_amps):
            stim_params = StimulationParameters(
                electrodes=[7],
                stim_amplitude=np.asarray([0, a]),  # np.linspace(0.01, 1, 25),
                pulse_width=[0.1],
            )
            transition_m, cov_m = _compute_transition_matrix_phase1_report(
                stimulation_params=stim_params,
                neuron_i=neuron_list
            )

            # Define the noise
            sigma_v = 0
            sigma_d = 0.15
            mi[i] = ini_estim.mutual_information.theory.linear_gaussian_mi(
                transition_m.T,
                cov_m,
                sigma_v,
                sigma_d
            )
        plt.plot(max_stim_amps, mi, '-', label='{}'.format(ni))
        plt.xlabel('Maximum stimulation amplitude')
        plt.ylabel('Mutual information (bits)')
    plt.legend()
    plt.show()


def vary_output_noise(noise_values, stim_params):
    """
    :param noise_values: list of output noise values to test
    :param stim_params:
    :return: corresponding mi
    """
    mi = np.empty(len(noise_values))

    # Center electrodes only
    # neuron_list = [*range(int(17692/2)-76, int(17692/2)-74),
    #                *range(int(17692/2)+74, int(17692/2)+76)]
    neuron_list = [int(17692/2)-76]

    transition_m, cov_m = _compute_transition_matrix_phase1_report(
        stimulation_params=stim_params,
        neuron_i=neuron_list
    )
    ref_noise = calculate_output_noise(transition_m.T, cov_m, len(neuron_list))
    noise_20db = calculate_output_noise(transition_m.T, cov_m, len(neuron_list),
                                        snr=20)
    noise_3db = calculate_output_noise(transition_m.T, cov_m, len(neuron_list),
                                       snr=3)
    print('10dB noise std deviation is {}'.format(ref_noise))
    print('The shape of the connectivity matrix is {}'.
          format(transition_m.shape))
    print('Rank of connectivity matrix is {}'.format(
        np.linalg.matrix_rank(transition_m)))

    for i, n in enumerate(noise_values):
        # Define the noise
        sigma_v = 0
        sigma_d = n

        print('Computing the channel capacity {}... '.format(i))
        mi[i] = ini_estim.mutual_information.theory.linear_gaussian_mi(
            transition_m.T,
            cov_m,
            sigma_v,
            sigma_d
        )
        print('Channel capacity is {}'.format(mi[i]))
    plt.show()
    plt.plot(noise_values, mi, 'k')
    plt.axvline(x=ref_noise, color='r')
    plt.text(ref_noise, 8, '10dB')
    plt.axvline(x=noise_3db, color=[0.8, 0.8, 0.8])
    plt.text(noise_3db, 8, '3dB')
    plt.axvline(x=noise_20db, color=[0.8, 0.8, 0.8])
    plt.text(noise_20db-0.05, 8, '20dB')
    plt.xlabel('Output noise standard deviation')
    plt.ylabel('Mutual information (bits)')
    plt.show()
    return mi


def vary_which_electrode():
    mi = np.empty(14)
    # Center electrodes only
    neuron_list = [*range(int(17692/2)-76, int(17692/2)-74),
                   *range(int(17692/2)+74, int(17692/2)+76)]

    plt.figure(1)
    for e in range(14):
        stim_params = StimulationParameters(
            electrodes=[e],
            stim_amplitude=np.asarray([0, 0.5]),  # np.linspace(0.01, 1, 25),
            pulse_width=[0.1],
        )

        transition_m, cov_m = _compute_transition_matrix_phase1_report(
            stimulation_params=stim_params,
            neuron_i=neuron_list
        )

        if DEBUG:
            plt.subplot(3, 5, e+1)
            plt.scatter(ROUND_CUFF.reference_potential_xc,
                        ROUND_CUFF.reference_potential_yc,
                        10, ROUND_CUFF.reference_potential_map[e])

            plt.scatter(ROUND_CUFF.reference_potential_xc[neuron_list],
                        ROUND_CUFF.reference_potential_yc[neuron_list],
                        50, 'r')
            plt.axis('off')
            # plt.scatter(ROUND_CUFF.reference_potential_xc[neuron_list],
            #             ROUND_CUFF.reference_potential_yc[neuron_list],
            #             50, [*range(len(neuron_list))])
            # plt.colorbar()
            plt.title('Active electrode #{}'.format(e))

        print('The shape of the connectivity matrix is {}'.
              format(transition_m.shape))
        print('Rank of connectivity matrix is {}'.format(
            np.linalg.matrix_rank(transition_m)))

        # Define the noise
        sigma_v = 0
        sigma_d = 0.15

        mi[e] = ini_estim.mutual_information.theory.linear_gaussian_mi(
            transition_m.T,
            cov_m,
            sigma_v,
            sigma_d
        )
        print('Mutual information is {}'.format(mi[e]))
    plt.show()
    plt.plot(range(14), mi)
    plt.xlabel('Active electrode in cuff')
    plt.ylabel('Mutual information (bits)')
    plt.show()


def neuron_positions(neuron_i):
    """Displays the neuron positions on the nerve

    :param neuron_i: list of neurons to plot
    """
    data = [x / 10 for x in ROUND_CUFF.reference_potential_map[0]]
    for electrode in range(1, 14):
        data = [sum(x)
                for x in zip(data,
                             [x/10 for x in
                              ROUND_CUFF.reference_potential_map[electrode]])]

    # Show the map of potential values and where the neurons are located
    plt.scatter(ROUND_CUFF.reference_potential_xc,
                ROUND_CUFF.reference_potential_yc, 10,
                ROUND_CUFF.reference_potential_map[7])

    plt.colorbar()
    # plt.set_cmap('jet')
    for n, neuron in enumerate(neuron_i):
        marker_str = '${}$'.format(n)
        # plt.scatter(ROUND_CUFF.reference_potential_xc[neuron],
        #             ROUND_CUFF.reference_potential_yc[neuron],
        #             10, 'r')
        plt.scatter(ROUND_CUFF.reference_potential_xc[neuron],
                    ROUND_CUFF.reference_potential_yc[neuron],
                    marker=marker_str)
    plt.axis('off')
    plt.show()


def calculate_output_noise(w, cs, num_neurons, snr=10, sigma_v=0):
    numerator = w @ cs @ w.T + sigma_v ** 2 * w @ w.T
    var = np.trace(numerator)/num_neurons/snr
    return np.sqrt(var)


def find_closest_to(x, y, all_x, all_y):
    xy_diffs = np.concatenate(((x - all_x)[:, np.newaxis],
                              (y - all_y)[:, np.newaxis]),
                              axis=1)
    xy_distances = np.linalg.norm(xy_diffs, axis=1)
    return int(np.where(xy_distances == np.min(xy_distances))[0][0])


def find_indices_near_electrodes():
    radius = 1.4
    angle = 2*math.pi/14
    idx = []
    for e in range(14):
        x_pos = radius*math.cos(e*angle)
        y_pos = radius*math.sin(e*angle)

        idx.append(find_closest_to(x_pos,
                                   y_pos,
                                   ROUND_CUFF.reference_potential_xc,
                                   ROUND_CUFF.reference_potential_yc)
                   )

    return idx


def vary_number_of_electrodes(neuron_i):
    """Check the result of activating one or many electrodes
    :param neuron_i:
    """
    mi = np.empty(14)

    # Define the noise
    sigma_v = 0
    sigma_d = 0.15

    for ni, n in enumerate(range(14)):
        stim_params = StimulationParameters(
            electrodes=[*range(n)],
            stim_amplitude=np.asarray([0, 0.5]),  # np.linspace(0.01, 1, 25),
            pulse_width=[0.1],
        )

        transition_m, cov_m = _compute_transition_matrix_phase1_report(
            stimulation_params=stim_params,
            neuron_i=neuron_i
        )

        mi[ni] = ini_estim.mutual_information.theory.linear_gaussian_mi(
            transition_m.T,
            cov_m,
            sigma_v,
            sigma_d)

    plt.plot([*range(14)], mi, 'k.-')
    plt.xlabel('Number of active electrodes')
    plt.ylabel('Mutual information (bits)')
    plt.show()
    return mi


def vary_number_of_output_neurons_multiple_elecs_levels():
    num_neurons = np.linspace(1, 13000, 20)
    sigma_v = 0
    sigma_d = 0.15
    stim_params = StimulationParameters(
        electrodes=[*range(14)],
        stim_amplitude=np.pad(np.linspace(0, 1.2, 20), (20 - 2, 0),
                              'constant', constant_values=(0, 0)),
        pulse_width=[0.1],
    )

    mi = np.empty(len(num_neurons))
    for i, nn in enumerate(num_neurons):
        print('Computing MI for {} neurons'.format(nn))
        neuron_i = np.random.choice([*range(17691)], int(nn), replace=False)
        transition_m, cov_m = _compute_transition_matrix_phase1_report(
            stimulation_params=stim_params,
            neuron_i=neuron_i
        )
        mi[i] = ini_estim.mutual_information.theory.linear_gaussian_mi(
            transition_m.T,
            cov_m,
            sigma_v,
            sigma_d)
    plt.plot(num_neurons, mi, 'k')
    plt.xlabel('Number of output neurons')
    plt.ylabel('Mutual information (bits)')
    plt.show()
    return mi, num_neurons


def vary_number_of_output_neurons():
    num_neurons = np.linspace(1, 13000, 20)
    sigma_v = 0
    sigma_d = 0.15
    stim_params = StimulationParameters(
        electrodes=[0],
        stim_amplitude=np.asarray([0, 5]),  # np.linspace(0.01, 1, 25),
        pulse_width=[0.1],
    )

    mi = np.empty(len(num_neurons))
    for i, nn in enumerate(num_neurons):
        print('Computing MI for {} neurons'.format(nn))
        neuron_i = np.random.choice([*range(17691)], int(nn), replace=False)
        transition_m, cov_m = _compute_transition_matrix_phase1_report(
            stimulation_params=stim_params,
            neuron_i=neuron_i
        )
        mi[i] = ini_estim.mutual_information.theory.linear_gaussian_mi(
            transition_m.T,
            cov_m,
            sigma_v,
            sigma_d
        )
    plt.plot(num_neurons, mi, 'k')
    plt.xlabel('Number of output neurons')
    plt.ylabel('Mutual information (bits)')

    c = np.polyfit(np.log(num_neurons), mi, 1)

    logfit_mi = c[0]*np.log(num_neurons) + c[1]
    plt.plot(num_neurons, logfit_mi)
    plt.show()
    return mi, num_neurons


def vary_stimulation_levels(neuron_i):
    # Define the noise
    sigma_v = 0
    sigma_d = 0.15
    stim_levels = [int(x) for x in range(2, 21, 1)]
    mi = np.empty(len(stim_levels))
    for i, sl in enumerate(stim_levels):

        stim_params = StimulationParameters(
            electrodes=[0],
            stim_amplitude=np.pad(np.linspace(0, 1.2, sl), (sl-2, 0),
                                  'constant', constant_values=(0, 0)),
            pulse_width=[0.1],
        )
        print('For {} stimulus levels, stim amps are {}'.
              format(sl, stim_params.stim_amplitude))

        transition_m, cov_m = _compute_transition_matrix_phase1_report(
            stimulation_params=stim_params,
            neuron_i=neuron_i
        )

        mi[i] = ini_estim.mutual_information.theory.linear_gaussian_mi(
            transition_m.T,
            cov_m,
            sigma_v,
            sigma_d
        )
    plt.plot(stim_levels, mi, 'k.-')
    plt.xlabel('Number of stimulus levels (linspace)')
    plt.ylabel('MI (bits)')
    plt.title('{} output neuron(s)'.format(len(neuron_i)))
    plt.show()


def debugging_plot(neuron_i):
    # Define the noise
    sigma_v = 0
    sigma_d = 0.15
    stim_levels = [int(x) for x in range(2, 21, 1)]
    for ni, n in enumerate(neuron_i):
        mi = np.empty(len(stim_levels))
        for i, sl in enumerate(stim_levels):
            stim_params = StimulationParameters(
                electrodes=[7],
                stim_amplitude=np.pad(np.linspace(0, 1.6, sl), (sl - 2, 0),
                                      'constant', constant_values=(0, 0)),
                # stim_amplitude=np.pad(
                #     np.pad(np.asarray([0, 1.6]), (0, sl - 2),
                #            'linear_ramp', end_values=(0, 5)
                #            ), (sl - 2, 0),'constant', constant_values=(0, 0)
                # ),
                # stim_amplitude=np.pad(np.asarray([0, 1.6]), (0, sl - 2),
                #                       'linear_ramp', end_values=(0, 5)),
                # stim_amplitude=np.linspace(0.01, 1, 25),
                pulse_width=[0.1],
            )

            trans_m, cov_m = _compute_transition_matrix_phase1_report(
                stimulation_params=stim_params,
                neuron_i=[n]
            )

            mi[i] = ini_estim.mutual_information.theory.linear_gaussian_mi(
                trans_m.T,
                cov_m,
                sigma_v,
                sigma_d
            )
        plt.plot(stim_levels, mi, '.-', label='{}'.format(ni))
        plt.xlabel('Number of stimulus levels (linspace)')
        plt.ylabel('MI (bits)')
    plt.title('Change location of output neuron(s)')
    plt.legend(loc='upper right')
    plt.show()


def plot_a(stim_params, neuron_i):

    transition_m, cov_m = _compute_transition_matrix_phase1_report(
        stimulation_params=stim_params,
        neuron_i=neuron_i
    )
    # plt.subplot(1, 2, 1)
    plt.imshow(transition_m)
    plt.colorbar()
    plt.title('A')
    plt.show()
    # plt.subplot(1, 2, 2)
    plt.imshow(cov_m)
    plt.colorbar()
    plt.title('K')
    plt.show()


def generate_probability_plot():
    sigma_v = 0
    sigma_d = 0.15
    neuron_i = [*range(int(17692 / 2) - 150, int(17692 / 2), 48)]
    # neuron_list = [int(17692 / 2) - 76]
    p_high = np.linspace(0, 1, 11)
    mi = np.empty(len(p_high))
    for i, p in enumerate(p_high):
        stim_params = StimulationDefinition(
            electrodes=[0],
            stim_amplitude=np.asarray([0, 2]),
            pulse_width=[0.1],
            probabilities=np.asarray([[1-p, p]])
        )

        a_matrix = _compute_transfer_matrix(stim_params, neuron_i)
        k_matrix = _compute_covariance_matrix(stim_params)
        mi[i] = mutual_information.theory.linear_gaussian_mi(a_matrix,
                                                             k_matrix,
                                                             sigma_v,
                                                             sigma_d)

    plt.plot(p_high, mi, '.-')
    plt.xlabel('Probability of high state')
    plt.ylabel('MI (bits)')
    plt.show()


def compute_mi(p):
    """Computes the mutual information as a function of the command sequence
    probability distribution
    :param p
    :returns mi: mutual information
    """
    sigma_v = 0
    sigma_d = 0.15
    neuron_i = [*range(int(17692 / 2) - 150, int(17692 / 2), 48)]

    stim_params = StimulationDefinition(
        electrodes=[0],
        stim_amplitude=np.asarray([0, 2]),
        pulse_width=[0.1],
        probabilities=np.asarray([[1-p, p]])
    )
    a_matrix = _compute_transfer_matrix(stim_params, neuron_i)
    k_matrix = _compute_covariance_matrix(stim_params)
    return -1*mutual_information.theory.linear_gaussian_mi(a_matrix,
                                                           k_matrix,
                                                           sigma_v,
                                                           sigma_d)


if __name__ == "__main__":
    generate_probability_plot()
    neuron_idx = [*range(int(17692 / 2) - 150, int(17692 / 2), 48)]
    s_params = StimulationDefinition(
        electrodes=[0, 7],
        stim_amplitude=np.asarray([0, 0.2, 0.8, 2]),
        pulse_width=[0.1],
        probabilities=np.asarray([[0.5, 0, 0, 0.5],
                                  [0.25, 0.25, 0.25, 0.25]])
    )
    A = _compute_transfer_matrix(s_params, neuron_idx)
    K = _compute_covariance_matrix(s_params)

    plt.subplot(1, 2, 1)
    plt.imshow(A)
    plt.colorbar()
    plt.title('A')
    plt.subplot(1, 2, 2)
    plt.imshow(K)
    plt.colorbar()
    plt.title('K')
    plt.show()

    # Generate figures for final report.
    fig4 = 1  # Generate figure showing example transfer & covariance matrices
    if fig4:
        # Plot A and K
        neuron_idx = [*range(int(17692 / 2) - 150, int(17692 / 2), 48)]
        neuron_positions(neuron_idx)
        s_params = StimulationParameters(
            electrodes=[0, 7],
            stim_amplitude=np.asarray([0, 0.2, 0.8, 2]),
            pulse_width=[0.1],
        )
        plot_a(s_params, neuron_idx)

    fig5 = 0
    if fig5:
        # Plot the influence on output noise on the mutual information
        s_params = StimulationParameters(
            electrodes=[0],
            stim_amplitude=np.asarray([0, 25]),  # np.linspace(0.01, 1, 25),
            pulse_width=[0.1],
        )
        vary_output_noise(noise_values=np.logspace(-3, 0.1, 40),
                          stim_params=s_params)

    fig6 = 0
    if fig6:
        # Plot the spatial influence of stimulation amplitude on mi
        neuron_idx = [*range(int(17692 / 2) - 150, int(17692 / 2) - 0, 20)]
        neuron_positions(neuron_idx)
        vary_max_stim_amp(np.linspace(0, 2, 100), neuron_idx)

    fig7 = 0
    if fig7:
        neuron_idx = [*range(int(17692 / 2) - 150, int(17692 / 2) - 0, 20)]
        neuron_positions(neuron_idx)
        debugging_plot(neuron_idx)

    # Generate figure showing the effect of increasing the number of stimulus
    # levels for multiple output neurons arranged along the nerve axis
    fig8 = 0
    if fig8:
        neuron_idx = [*range(int(17692 / 2) - 150, int(17692 / 2) - 0, 10)]
        neuron_positions(neuron_idx)
        vary_stimulation_levels(neuron_idx)

    # Generate figure showing the effect of increasing the number of electrodes
    # for neurons positioned near electrodes
    fig9 = 0
    if fig9:
        neuron_idx = find_indices_near_electrodes()
        neuron_positions(neuron_idx)
        vary_number_of_electrodes(neuron_idx)

    fig10 = 0  # How does adding output channels affect mutual information
    if fig10:
        vary_number_of_output_neurons_multiple_elecs_levels()
        vary_number_of_output_neurons()

    # How does adding stimulation levels change things for 1 output neuron?
    single_neuron = [int(17692/2)-76]
