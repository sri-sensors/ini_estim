import numpy as np


def linear_gaussian_mi(w, cs, sigma_v, sigma_d):
    """
    Calculates the mutual information for a linear-Gaussian response model,
    where the input to the system is x=s+v, where s is the "ideal" signal, v is
    noise with variance sigma_v**2, and the output is y=Wx+d, where W is the
    transfer function (ie, retinal recoding weights, or weight matrix, or
    prosthetic transfer function) with noise d with variance sigma_d**2
    :param w: weights from stimulus entries to responses
    :param cs: covariance of stimulus
    :param sigma_v: input noise, ie, noise associated with sensors
    :param sigma_d: output noise, ie, noise associated with spiking
    :return: mutual info, given by http://www.jneurosci.org/content/32/46/16256
    """
    n = np.shape(w)[0]

    # For the spiking noise, use either a mean value or a square matrix for each
    # output channel
    if isinstance(sigma_d, float):
        output_noise = sigma_d ** 2 * np.identity(n)
    else:
        output_noise = sigma_d

    # Compute the values for the mutual information in two terms
    numerator = w@cs@w.T + sigma_v ** 2 * w@w.T + output_noise
    denominator = sigma_v ** 2 * w@w.T + sigma_d ** 2 * np.identity(n)

    # Take the natural log and determinant for each term
    log_num = np.linalg.slogdet(numerator)[1] / np.log(2)
    log_den = np.linalg.slogdet(denominator)[1] / np.log(2)
    return 0.5 * (log_num - log_den)


def gaussian_channel_capacity(Pavg, n_var, bw=None):
    """

    :param Pavg: The average signal power
    :param n_var: The noise variance (noise power)
    :param bw: The bandwidth of the channel in Hz. Let bw=None to compute the
        capacity for the unlimited bandwidth case
    :return: The channel capacity (computed for all amplitude values 0-Pmax
        equally probable -> Pavg = Pmax/3). Note that in the limited bandwidth
        case, the capacity is in units (bits/second), while in the unlimited
        bandwidth case, the capacity is in units (bits/transmission).
    """
    if bw is not None:
        C = bw*np.log2(1+Pavg/n_var)
    else:
        C = 0.5*np.log2(1+Pavg/n_var)
    return C