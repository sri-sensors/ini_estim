"""
Mutual Information and Entropy estimators that use nearest neighbor methods
"""

import numpy as np
from scipy.special import digamma, gamma
from sklearn.neighbors import NearestNeighbors


def normalize_data(data, mean=None, n_neighbors=3):
    """ Normalize data for use with MI nn methods

    Normalize the data so that the average radius to the to Nth neighbor is 1.0.

    Parameters
    ----------
    data : ndarray (n_samples, n_dims)
        The input data
    mean : float or ndarray (1, n_dims)
        The mean to subtract from the data before normalization (will be re-added
        after). If None (default) the mean will be estimated from the data.
    n_neighbors : int
        The nearest neighbor to normalize by. Default is 3.
    """
    if mean is None:
        mean = np.mean(data, axis=0)[None,:]
    data = data - mean
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)
    nn.fit(data)
    r = nn.kneighbors()[0]
    r = np.nextafter(r[:, -1], 0)
    data /= r.mean()
    return data


def mi_discrete(x, y):
    """ MI between discrete datasets

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_dim_x)
        Samples of a discrete random variable.
    y : ndarray, shape (n_samples, n_dim_y)

    Returns
    -------
    mi : float
        Estimated mutual information
    """
    if x.shape[0] != y.shape[0]:
        raise ValueError('x_data and y_data must have the '
                         'same number of data points.')
    Npts = x.shape[0]
    if x.ndim < 2:
        x = x.reshape((-1, 1))
    if y.ndim < 2:
        y = y.reshape((-1, 1))
    xy = np.hstack((x,y))

    hx = entropy_discrete(x)
    hy = entropy_discrete(y)
    hxy = entropy_discrete(xy)
    return hx + hy - hxy


def entropy_discrete(x):
    """ Entropy of discrete dataset

    Parameters
    ----------
    x : ndarray, shape (n_samples, n_dim_x)
        Samples of a discrete random variable.

    Returns
    -------
    entropy : float
        Estimated shannon entropy in bits
    """
    Npts = x.shape[0]
    if x.ndim < 2:
        x = x.reshape((-1, 1))
    hx = 0.0
    for lbl in np.unique(x, axis=0):
        mask = np.all(x == lbl, axis=1)
        count = np.count_nonzero(mask)
        p = count / Npts
        if p > 0.0:
            hx -= p*np.log2(p)
    return hx


def mi_ross(c: np.ndarray, d: np.ndarray, n_neighbors: int=3):
    """ MI between continuous and discrete datasets

    This function is mostly a modified version of scikit-learn's implementation
    in the sklearn.feature_selection module.

    Parameters
    ----------
    c : ndarray, shape (n_samples, n_dim)
        Samples of a continuous random variable.
    d : ndarray, shape (n_samples, n_dim)
        Samples of a discrete random variable.
    n_neighbors : int
        Number of nearest neighbors to search for each point, see [1]_.

    Returns
    -------
    mi : float
        Estimated mutual information. If it turned out to be negative it is
        replaced by 0.

    Notes
    -----
    True mutual information can't be negative. If its estimate by a numerical
    method is negative, it means (providing the method is adequate) that the
    mutual information is close to 0 and replacing it by 0 is a reasonable
    strategy.

    References
    ----------
    .. [1] B. C. Ross "Mutual Information between Discrete and Continuous
       Data Sets". PLoS ONE 9(2), 2014.
    """
    N = c.shape[0]
    if c.ndim < 2:
        c = c.reshape((-1, 1))
    if d.ndim < 2:
        d = d.reshape((-1, 1))
    radius = np.empty(N)
    label_counts = np.empty(N)
    k_all = np.empty(N)
    nn = NearestNeighbors()
    for label in np.unique(d, axis=0):
        mask = np.all(d == label, axis=1)
        count = np.count_nonzero(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count

    mask = label_counts > 1
    N = np.count_nonzero(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]

    nn.set_params(algorithm='kd_tree')
    nn.fit(c)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    m_all = np.array([i.size for i in ind])
    minfo = (digamma(N) + np.mean(digamma(k_all)) -
             np.mean(digamma(label_counts)) - np.mean(digamma(m_all + 1)))
    return max(0, minfo)


def mi_ksg(x: np.ndarray, y: np.ndarray, n_neighbors: int=3, normalize=False):
    """ MI between continuous random variables

    Parameters
    ----------
    x : np.ndarray (n_samples, n_dimensions)
        Samples of input continuous random variables
    y : np.ndarray (n_samples, m_dimensions)
        Samples of output continuous random variables
    n_neighbors : int
        Number of nearest neighbors to search for at each point.
    normalize : bool, optional
        Boolean to normalize the data before computing MI, False by default.

    Returns
    -------
    mi : float
        The estimated mutual information.

    Notes
    -----
    This function is mostly a modified version of scikit-learn's implementation
    in the sklearn.feature_selection module.

    References
    ----------
    [1] A. Kraskov, H. Stogbauer and P. Grassberger, "Estimating mutual
        information". Phys. Rev. E 69, 2004.

    """
    n_samples = x.shape[0]
    if x.ndim < 2:
        x = x.reshape((-1, 1))
    if y.ndim < 2:
        y = y.reshape((-1, 1))
    if normalize:
        x = normalize_data(x, n_neighbors=n_neighbors)
        y = normalize_data(y, n_neighbors=n_neighbors)
    xy = np.hstack((x, y))

    # Below is just a straight copy of scikit-learn's _compute_mi_cc
    # -------
    # Here we rely on NearestNeighbors to select the fastest algorithm.
    nn = NearestNeighbors(metric='chebyshev', n_neighbors=n_neighbors)

    nn.fit(xy)
    radius = nn.kneighbors()[0]
    radius = np.nextafter(radius[:, -1], 0)

    # Algorithm is selected explicitly to allow passing an array as radius
    # later (not all algorithms support this).
    nn.set_params(algorithm='kd_tree')

    nn.fit(x)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    nx = np.array([i.size for i in ind])

    nn.fit(y)
    ind = nn.radius_neighbors(radius=radius, return_distance=False)
    ny = np.array([i.size for i in ind])

    mi = (digamma(n_samples) + digamma(n_neighbors) -
          np.mean(digamma(nx + 1)) - np.mean(digamma(ny + 1)))
    # mi = (digamma(n_samples) + digamma(n_neighbors) -
    #       np.mean(digamma(nx + 1) + digamma(ny + 1)))
    # mi is in nats, convert to bits
    mi *= np.log2(np.e)
    return max(0, mi)


def knn_entropy(x, k=1):
    """ Estimate entropy using the Kozachenko-Leonenko knn estimator

    Parameters
    ----------
    x : np.ndarray (n_samples, n_dimensions)
        Samples of input continuous random variables
    k : int
        Number of nearest neighbors to search for at each point.

    Returns
    -------
    Estimated entropy

    References
    ----------
    [1] Kozachenko, L. F. & Leonenko, N. N. 1987 Sample estimate of entropy
        of a random vector. Probl. Inf. Transm. 23, 95-101.
    [2] https://arxiv.org/pdf/1506.06501v1.pdf
    [3] http://www.cs.tut.fi/~timhome/tim/tim/core/differential_entropy_kl_details.htm

    Notes
    -----
    Reference [2] features an alternate procedure that is potentially more
    accurate for high-dimensionality data, but requires more computation.

    """
    n_samples = x.shape[0]
    if x.ndim < 2:
        x = x.reshape((-1, 1))
    D = x.shape[1]
    c_d = float(2**D)

    nn = NearestNeighbors(metric='chebyshev', n_neighbors=k)
    nn.fit(x)
    distances, _ = nn.kneighbors(n_neighbors=k)
    d = distances[:, -1]
    d = d[d>0]
    H = digamma(n_samples) - digamma(k) + D/d.size*np.sum(np.log(d)) + np.log(c_d)
    return H*np.log2(np.e)
