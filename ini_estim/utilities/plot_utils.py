# Functions to plot electrode array geometries, potentials, and neurons
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
from matplotlib.collections import PatchCollection


def plot_cuff_array_with_neurons(n_electrodes, inner_r, neuron_pos,
                                 colormap=None, active_electrodes: list = None,
                                 fig_params=None, title_str=None, cbar_str=None,
                                 save_name=None):
    """Plots the electrode locations for a round cuff electrode array
    and the positions of the output neurons.
    :param n_electrodes : number of electrodes in the round cuff (evenly spaced)
    :param inner_r : the radius of the nerve (mm)
    :param neuron_pos : tuple (x,y) containing a list of the x and y coordinates
                        of the neurons (mm)
    :param colormap : a vector (length = len(x)) with scaled values to assign to
                      neurons
    :param active_electrodes: list of active electrodes. color-codes these red
    :param fig_params: (figure, axis) tuple
    :param title_str: title for the plot. if empty, a default title is used
    :param cbar_str: string label for the colorbar. If None, colorbar is assumed
                     to represent the potential from active electrodes.
    :param save_name: full path filename to save figure. if populated, figure is
                      saved. otherwise, figure is not saved.
    """
    if fig_params is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig, ax = fig_params

    if colormap is not None:
        im = ax.scatter(neuron_pos[0], neuron_pos[1], 10, colormap)
        cbar = fig.colorbar(im, ax=ax)
        if cbar_str is not None:
            cbar.ax.set_ylabel(cbar_str)
        else:
            cbar.ax.set_ylabel('potential (mV) from active electrode(s)')
    else:
        ax.scatter(neuron_pos[0], neuron_pos[1], 10)

    angular_len = np.floor(360 / n_electrodes) - 5
    center_locs = np.linspace(0, 360, n_electrodes, endpoint=False)

    patches = []
    for c in center_locs:
        patches += Wedge((0, 0), inner_r + 0.050, c - angular_len / 2,
                         c + angular_len / 2, width=0.050),

    colors = ['k'] * len(patches)  # 100*np.random.rand(len(patches))
    if active_electrodes is not None:
        for e in active_electrodes:
            colors[e] = 'r'

    p = PatchCollection(patches, alpha=1, fc=colors)
    ax.add_collection(p)
    ax.set_xlim(inner_r * np.asarray([-1.1, 1.1]))
    ax.set_ylim(inner_r * np.asarray([-1.1, 1.1]))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    if title_str is not None:
        ax.set_title(title_str)
    else:
        ax.set_title('{}-electrode cuff array with {} output neurons'.
                     format(n_electrodes, len(neuron_pos[0])))
    if save_name is not None:
        fig.savefig('{}.png'.format(save_name), bbox_inches='tight')
    plt.show()


def plot_fine_array_with_neurons(n_electrodes, width, height, neuron_pos,
                                 colormap=None, active_electrodes: list = None,
                                 cbar_str=None, title_str=None, save_name=None,
                                 map_on=True):
    """Plots the electrode locations for a round cuff electrode array
    and the positions of the output neurons.
    :param n_electrodes : number of electrodes in the round cuff (evenly spaced)
    :param width : the width of the FINE array (mm)
    :param height : the width of the FINE array (mm)
    :param neuron_pos : tuple (x,y) containing a list of the x and y coordinates
                        of the neurons (mm)
    :param colormap : a vector (length = len(x)) with scaled values to assign to
                      neurons
    :param active_electrodes: list of active electrodes. color-codes these red
        :param title_str: title for the plot. if empty, a default title is used
    :param cbar_str
    :param save_name: full path filename to save figure. if populated, figure is
                      saved.
    """

    fig, ax = plt.subplots(figsize=(16, 6))
    if colormap is not None:
        im = plt.scatter(neuron_pos[0], neuron_pos[1], 10, colormap)
        if map_on:
            cbar = fig.colorbar(im, ax=ax)
            if cbar_str is not None:
                cbar.ax.set_ylabel(cbar_str)
            else:
                cbar.ax.set_ylabel('potential (mV) from active electrode(s)')
    else:
        plt.scatter(neuron_pos[0], neuron_pos[1], 10)

    electrode_w = 0.5  # Currently fixed at 0.5mm
    corner_locs, step = np.linspace(-width / 2, width / 2,
                                    int(n_electrodes / 2) + 1, retstep=True)
    corner_locs += step / 2 - electrode_w / 2
    bounds = Rectangle((-5, -height / 2), 10, height, fill=False)

    patches = []
    # patches.append(Rectangle((-width/2, -height/2), width, height, fill=False))
    for c in corner_locs[:int(n_electrodes / 2)]:
        patches += Rectangle((c, -0.85), electrode_w, 0.1),
    for c in corner_locs[:int(n_electrodes / 2)]:
        patches += Rectangle((c, 0.75), electrode_w, 0.1),
    colors = ['k'] * len(patches)  # 100*np.random.rand(len(patches))
    if active_electrodes is not None:
        for e in active_electrodes:
            colors[e] = 'r'

    p = PatchCollection(patches, alpha=1, fc=colors)
    ax.add_collection(p)
    ax.add_artist(bounds)
    ax.set_xlim(width / 2 * np.asarray([-1.1, 1.1]))
    ax.set_ylim(height / 2 * np.asarray([-1.1, 1.1]))
    ax.set_aspect('equal', 'box')
    ax.set_xlabel('mm')
    ax.set_ylabel('mm')
    if title_str is not None:
        ax.set_title(title_str)
    else:
        ax.set_title('{}-electrode FINE array with {} output neurons'.format(
            n_electrodes, neuron_pos[0].size))
    if save_name is not None:
        fig.savefig('{}.png'.format(save_name), bbox_inches='tight')
    plt.show()


def plot_transition_matrix(A, stim_amps=None):
    fig, ax = plt.subplots(1, 1, figsize=(16,6))
    im = ax.imshow(A, aspect='auto')
    ax.set_ylabel('Neuron index')
    ax.set_xlabel('Control level')
    if stim_amps is not None:
        labels = np.tile(stim_amps, int(A.shape[1]/stim_amps.shape[0]))
        ax.set_xticks([*np.arange(0, A.shape[1], 2)])
        ax.set_xticklabels(labels[::2])
        ax.set_xlabel('stimulus amplitude (mA)')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('probability of activation')