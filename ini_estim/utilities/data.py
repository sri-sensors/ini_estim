import numpy as np

as_strided = np.lib.stride_tricks.as_strided


def slide_array(a: np.ndarray, w: int, skip: int = 1):
    """

    Parameters
    ----------
    a
    w
    skip

    Returns
    -------

    """
    if (a.ndim > 1):
        raise ValueError("input array must have only a single dimension")
    b = as_strided(a, (a.size - w, w), a.strides*2)
    return b[::skip, :].copy()


def sum_samples(arr, num_samples):
    """ Sum array by block

    Parameters
    ----------
    arr : np.ndarray
        2D Input array (num_data, data_dimensions)
    num_samples : int
        Number of samples per block to sum

    Returns
    -------
    Reduced array (size will be ~num_data/num_samples x data_dimensions)

    Notes
    -----
    If num_data is not an integer multiple of num_samples, the array will be
    padded with 0.
    """
    num_samples = int(num_samples)
    num_dims = arr.shape[1]
    pad = (arr.shape[0] % num_samples)
    if pad:
        pad = num_samples - pad
    arr_out = np.pad(arr, [[0,pad],[0,0]], 'constant')
    arr_out = arr_out.reshape((-1, num_samples, num_dims))
    arr_out = np.sum(arr_out, 1)
    return arr_out


def reformat_by_sample(arr, num_samples):
    """ Reshape array by block

    Parameters
    ----------
    arr : np.ndarray
        2D Input array (num_data, data_dimensions)
    num_samples : int
        Number of samples per block to sum

    Returns
    -------
    Reshaped array (size will be ~num_data/num_samples x data_dims*num_samples)

    Notes
    -----
    If num_data is not an integer multiple of num_samples, the array will be
    padded by the edge values

    """
    num_samples = int(num_samples)
    num_dims = arr.shape[1]
    pad = (arr.shape[0] % num_samples)
    if pad:
        pad = num_samples - pad
    arr_out = np.pad(arr, [[0,pad],[0,0]], 'edge')
    arr_out = arr_out.reshape((-1, num_dims*num_samples))
    return arr_out