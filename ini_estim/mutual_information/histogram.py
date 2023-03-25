import numpy as np


def hist_mutual_info(x_data: np.ndarray, y_data: np.ndarray,
                     bins=10):
    """

    :param x_data: (N,X) numpy array, where N is the number of data
        samples and X is the dimensionality of the data.
        See numpy.histogramdd
    :param y_data: (N,Y) numpy array, similar to x_data.
    :param bins: see numpy.histogramdd for possible values.
    :return: mutual info from joint histogram
    """
    x_dim = x_data.shape[1]
    y_dim = y_data.shape[1]

    if x_data.shape[0] != y_data.shape[0]:
        raise ValueError('x_data and y_data must have the '
                         'same number of data points.')

    Npts = x_data.shape[0]

    try:
        if len(bins) == 2 and x_dim + y_dim != 2:
            bins = [bins[0]]*x_dim + [bins[1]]*y_dim
    except (TypeError, AttributeError):
        pass

    # Get probability distributions
    pxy, bin_edges = np.histogramdd(
        np.hstack((x_data, y_data)), bins, density=False)
    pxy /= Npts
    px, _ = np.histogramdd(x_data, bin_edges[0:x_dim], density=False)
    px /= Npts
    py, _ = np.histogramdd(y_data, bin_edges[x_dim:], density=False)
    py /= Npts

    # Compute p(x)*p(y) using array broadcasting
    for dim in range(y_dim):
        px = np.expand_dims(px, px.ndim)
    for dim in range(x_dim):
        py = np.expand_dims(py, 0)
    px_py = px*py

    # Compute mutual information.
    # - exclude places where pxy = 0 since Ixy at that point is 0.
    idx = np.flatnonzero(pxy)
    pxy = pxy.ravel()[idx]
    px_py = px_py.ravel()[idx]
    Ixy = np.sum(pxy*(np.log2(pxy)-np.log2(px_py)))
    return Ixy


def entropy(x, bins=None):
    """

    :param x: the data
        x can be either single dimensional or 2-dimensional. In the case that
        x is 2-dimensional, x will be interpreted as in np.histogramdd:

        * When an array, each row is a coordinate in a D-dimensional space -
          such as ``histogramgramdd(np.array([p1, p2, p3]))``.
        * When an array_like, each element is the list of values for single
          coordinate - such as ``histogramgramdd((X, Y, Z))``.

        The first form should be preferred.
    :param bins: Same argument as 'bins' in np.histogram, or np.histogramdd
    :return: the computed entropy
    """
    if np.ndim(x) == 1:
        bins = 'fd' if bins is None else bins
        p, bin_edges = np.histogram(x, bins=bins, density=False)
        p = p / np.sum(p)
    else:
        p, edges = np.histogramdd(x, bins=bins, density=False)
        p = p.ravel()
        p = p / np.sum(p)

    idx = np.flatnonzero(p)
    return -np.sum(p[idx] * np.log2(p[idx]))


def hist_mutual_info_lowmem(x_data: np.ndarray, y_data: np.ndarray, bins='fd'):
    """

    :param x_data:
    :param y_data:
    :param bins: See numpy.histogram_bin_edges
    :return:
    """
    x_dim = x_data.shape[1]
    y_dim = y_data.shape[1]
    x_edges = [
        np.histogram_bin_edges(x, bins) for x in x_data.transpose()
    ]
    y_edges = [
        np.histogram_bin_edges(y, bins) for y in x_data.transpose()
    ]
    raise NotImplementedError