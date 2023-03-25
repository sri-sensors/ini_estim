import numpy as np


def spike_triggered_average_multi(stim, spike_trains, filter_length):
    """

    :param stim: numpy array of stimulus values
    :param spike_trains: multi array of spike indexes in the form
        [(neuron_index_0, spike_indexes_0),
         (neuron_index_1, spike_indexes_1),
         ...
         (neuron_index_N-1, spike_indexes_N-1)
        ]
    :param filter_length: length of spike triggered average
    :return: multi array of spike triggered averages in the form
        [(neuron_index_0, spike_triggered_average_0),
         (neuron_index_1, spike_triggered_average_1),
         ...
         (neuron_index_N-1, spike_triggered_average_N-1)
        ]
    """
    out = []
    for t in spike_trains:
        idx = t[0]
        spikes = t[1]
        sta = spike_triggered_average(stim, spikes, filter_length)
        out.append((idx, sta))

    return out


def spike_triggered_average(stim, spikes, filter_length):
    """

    :param stim: numpy array of stimulus values
    :param spikes: array (or list) of spike indexes
    :param filter_length: length of spike triggered average
    :return: a filter_length vector of spike triggered averaged stimulus values.
    """
    out = np.zeros(filter_length)
    count = 0
    for s in spikes:
        idx_min = s - filter_length
        if idx_min < 0:
            continue

        out += stim[idx_min:idx_min+filter_length]
        count += 1

    if count>0:
        out = out/count

    return out